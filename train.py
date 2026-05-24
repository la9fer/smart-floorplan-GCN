"""
train.py
--------
Training loop for the FloorplanGCN model.

Handles:
    - Class imbalance via weighted cross-entropy loss
    - Early stopping on validation F1
    - Checkpoint saving (best model)
    - Per-epoch logging to results/training_log.csv

Run:
    python train.py

Author: [Your Name]
"""

import os
import csv
import time
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, accuracy_score

from model.dataset import FloorplanDataset
from model.gcn_model import FloorplanGCN, count_parameters

# ── Config ───────────────────────────────────────────────────────────────────
CONFIG = {
    "data_dir"    : "data",
    "results_dir" : "results",
    "checkpoint"  : "results/best_model.pt",
    "batch_size"  : 16,
    "hidden_dim"  : 64,
    "dropout"     : 0.3,
    "lr"          : 1e-3,
    "weight_decay": 1e-4,
    "epochs"      : 80,
    "patience"    : 15,       # early stopping patience
    "device"      : "cuda" if torch.cuda.is_available() else "cpu",
}


# ── Data ─────────────────────────────────────────────────────────────────────
def load_data(cfg):
    train_ds = FloorplanDataset(root=cfg["data_dir"], split="train")
    val_ds   = FloorplanDataset(root=cfg["data_dir"], split="val")
    test_ds  = FloorplanDataset(root=cfg["data_dir"], split="test")

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["batch_size"], shuffle=False)

    # Estimate class weight to handle imbalance
    # (congested nodes may be a minority)
    all_labels = [l for s in train_ds for l in s.y.numpy()]
    pos = sum(all_labels)
    neg = len(all_labels) - pos
    # weight[1] scales up the loss for congested (minority) class
    class_weight = torch.tensor([1.0, neg / max(pos, 1)], dtype=torch.float)
    print(f"[data] train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    print(f"[data] class weight: {class_weight}")
    return train_loader, val_loader, test_loader, class_weight


# ── Train/eval helpers ───────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index)
        loss = criterion(logits, batch.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch.y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    f1  = f1_score(all_labels, all_preds, zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, f1, acc


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index)
        loss = criterion(logits, batch.y)

        total_loss += loss.item() * batch.num_graphs
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch.y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    f1  = f1_score(all_labels, all_preds, zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, f1, acc


# ── Main training loop ───────────────────────────────────────────────────────
def train(cfg=CONFIG):
    os.makedirs(cfg["results_dir"], exist_ok=True)
    device = torch.device(cfg["device"])
    print(f"[train] Using device: {device}")

    train_loader, val_loader, test_loader, class_weight = load_data(cfg)

    model = FloorplanGCN(in_channels=6, hidden_dim=cfg["hidden_dim"], dropout=cfg["dropout"])
    model = model.to(device)
    print(f"[model] Parameters: {count_parameters(model):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=7, factor=0.5)
    criterion = nn.CrossEntropyLoss(weight=class_weight.to(device))

    # ── Logging setup ─────────────────────────────────────────────────────
    log_path = os.path.join(cfg["results_dir"], "training_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_f1", "train_acc",
                         "val_loss", "val_f1", "val_acc", "lr"])

    best_val_f1  = 0.0
    patience_cnt = 0

    print(f"\n{'Epoch':>6} {'TrLoss':>8} {'TrF1':>7} {'VaLoss':>8} {'VaF1':>7}")
    print("-" * 45)

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_f1, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_f1, va_acc = eval_epoch(model, val_loader,   criterion, device)
        scheduler.step(va_loss)

        lr_now = optimizer.param_groups[0]["lr"]

        # ── Log ──────────────────────────────────────────────────────────
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{tr_loss:.4f}", f"{tr_f1:.4f}", f"{tr_acc:.4f}",
                                    f"{va_loss:.4f}", f"{va_f1:.4f}", f"{va_acc:.4f}", lr_now])

        print(f"{epoch:>6} {tr_loss:>8.4f} {tr_f1:>7.4f} {va_loss:>8.4f} {va_f1:>7.4f}  "
              f"({time.time()-t0:.1f}s)")

        # ── Checkpoint ───────────────────────────────────────────────────
        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            torch.save(model.state_dict(), cfg["checkpoint"])
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= cfg["patience"]:
                print(f"\n[train] Early stopping at epoch {epoch}.")
                break

    # ── Final test evaluation ─────────────────────────────────────────────
    model.load_state_dict(torch.load(cfg["checkpoint"], map_location=device))
    te_loss, te_f1, te_acc = eval_epoch(model, test_loader, criterion, device)
    print(f"\n{'='*45}")
    print(f"[test]  Loss={te_loss:.4f}  F1={te_f1:.4f}  Acc={te_acc:.4f}")
    print(f"[test]  Best val F1: {best_val_f1:.4f}")
    print(f"[test]  Checkpoint: {cfg['checkpoint']}")
    print(f"{'='*45}")

    return model


if __name__ == "__main__":
    train()
