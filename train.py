
"""
Training pipeline for the FloorplanGCN model.

Features:
    - Weighted cross-entropy for class imbalance
    - Training and validation monitoring
    - Early stopping based on validation F1
    - Learning-rate reduction on validation loss plateau
    - Best-model checkpoint saving
    - Per-epoch CSV logging
    - Final held-out test evaluation

The test set is used only after model selection.

Run:
    python train.py
"""

import os
import csv
import time
import random

import numpy as np
import torch
import torch.nn as nn

from torch_geometric.loader import DataLoader

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from model.dataset import FloorplanDataset
from model.gcn_model import FloorplanGCN, count_parameters

# Configuration
CONFIG = {
    "data_dir": "data",
    "results_dir": "results",
    "checkpoint": "results/best_model.pt",

    "batch_size": 16,
    "hidden_dim": 64,
    "dropout": 0.3,

    "lr": 1e-3,
    "weight_decay": 1e-4,

    "epochs": 80,
    "patience": 15,

    "seed": 42,

    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# Reproducibility
def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Dataset loading
def load_data(cfg):

    train_ds = FloorplanDataset(
        root=cfg["data_dir"],
        split="train"
    )

    val_ds = FloorplanDataset(
        root=cfg["data_dir"],
        split="val"
    )

    test_ds = FloorplanDataset(
        root=cfg["data_dir"],
        split="test"
    )

    if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
        raise ValueError(
            "Training, validation and test datasets must all be non-empty."
        )

    # Verify binary labels in each partition
    for name, dataset in [
        ("Training", train_ds),
        ("Validation", val_ds),
        ("Test", test_ds),
    ]:

        labels = torch.cat([
            graph.y
            for graph in dataset
        ])

        classes = set(labels.unique().tolist())

        if not classes.issubset({0, 1}) or not classes:
            raise ValueError(
                f"{name} dataset contains invalid labels: {classes}"
            )

        print(
            f"[data] {name}: "
            f"{len(labels)} nodes | "
            f"not congested: {(labels == 0).sum().item()} | "
            f"congested: {(labels == 1).sum().item()}"
        )

    # Reproducible shuffling
    generator = torch.Generator()
    generator.manual_seed(cfg["seed"])

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        generator=generator,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
    )

    # Calculate class weights using training labels only
    all_labels = torch.cat([
        graph.y
        for graph in train_ds
    ])

    class_counts = torch.bincount(
        all_labels,
        minlength=2
    ).float()

    if torch.any(class_counts == 0):
        raise ValueError(
            "Both classes must be present in the training dataset."
        )

    total = class_counts.sum()

    class_weight = total / (2.0 * class_counts)

    print("\n[data] Class weights:", class_weight.tolist())

    print(
        f"[data] Graphs -> "
        f"train={len(train_ds)}, "
        f"val={len(val_ds)}, "
        f"test={len(test_ds)}"
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        class_weight,
    )

# Metric calculation
def calculate_metrics(all_labels, all_preds):

    all_labels = np.asarray(all_labels)
    all_preds = np.asarray(all_preds)

    tn, fp, fn, tp = confusion_matrix(
        all_labels,
        all_preds,
        labels=[0, 1]
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    return {
        "accuracy": accuracy_score(all_labels, all_preds),

        "balanced_accuracy": balanced_accuracy_score(
            all_labels,
            all_preds
        ),

        "precision": precision_score(
            all_labels,
            all_preds,
            zero_division=0
        ),

        "recall": recall_score(
            all_labels,
            all_preds,
            zero_division=0
        ),

        "specificity": specificity,

        "f1": f1_score(
            all_labels,
            all_preds,
            zero_division=0
        ),

        "mcc": matthews_corrcoef(
            all_labels,
            all_preds
        ),
    }

# Training and evaluation
def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
):
    """
    Run one training or evaluation epoch.

    If optimizer is supplied, model parameters are updated.
    Otherwise, the model is evaluated without gradient tracking.
    """

    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_loss_weight = 0.0

    all_preds = []
    all_labels = []

    context = torch.enable_grad() if training else torch.no_grad()

    with context:

        for batch in loader:

            batch = batch.to(device)

            if training:
                optimizer.zero_grad()

            logits = model(
                batch.x,
                batch.edge_index
            )

            loss = criterion(
                logits,
                batch.y
            )

            if training:
                loss.backward()
                optimizer.step()

            # Weighted CrossEntropyLoss with mean reduction
            # divides by the sum of target-class weights
            # Accumulate using that denominator to obtain an
            # appropriately aggregated epoch loss
            batch_loss_weight = criterion.weight[batch.y].sum().item()

            total_loss += loss.item() * batch_loss_weight
            total_loss_weight += batch_loss_weight

            preds = logits.argmax(dim=-1)

            all_preds.extend(
                preds.detach().cpu().numpy()
            )

            all_labels.extend(
                batch.y.detach().cpu().numpy()
            )

    if total_loss_weight == 0:
        raise ValueError("No nodes were processed during the epoch.")

    avg_loss = total_loss / total_loss_weight

    metrics = calculate_metrics(
        all_labels,
        all_preds
    )

    return avg_loss, metrics

# Main training loop
def train(cfg=CONFIG):

    set_seed(cfg["seed"])

    os.makedirs(
        cfg["results_dir"],
        exist_ok=True
    )

    checkpoint_dir = os.path.dirname(
        cfg["checkpoint"]
    )

    if checkpoint_dir:
        os.makedirs(
            checkpoint_dir,
            exist_ok=True
        )

    device = torch.device(cfg["device"])

    print(f"[train] Using device: {device}")

    (
        train_loader,
        val_loader,
        test_loader,
        class_weight,
    ) = load_data(cfg)

    # Model construction
    model = FloorplanGCN(
        in_channels=6,
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg["dropout"]
    ).to(device)

    print(
        f"[model] Trainable parameters: "
        f"{count_parameters(model):,}"
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"]
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=7,
        factor=0.5
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weight.to(device)
    )

    # CSV logging
    log_path = os.path.join(
        cfg["results_dir"],
        "training_log.csv"
    )

    columns = [
        "epoch",
        "train_loss",
        "train_f1",
        "train_accuracy",
        "val_loss",
        "val_f1",
        "val_accuracy",
        "val_precision",
        "val_recall",
        "val_specificity",
        "learning_rate",
    ]

    with open(
        log_path,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)
        writer.writerow(columns)

    best_val_f1 = -float("inf")
    patience_count = 0

    print(
        f"\n{'Epoch':>6} "
        f"{'TrLoss':>9} "
        f"{'TrF1':>8} "
        f"{'VaLoss':>9} "
        f"{'VaF1':>8}"
    )

    print("-" * 55)

    # Epoch loop
    for epoch in range(1, cfg["epochs"] + 1):

        start_time = time.time()

        train_loss, train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
        )

        val_loss, val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
        )

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        # Save epoch metrics
        with open(
            log_path,
            "a",
            newline=""
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                epoch,
                f"{train_loss:.6f}",
                f"{train_metrics['f1']:.6f}",
                f"{train_metrics['accuracy']:.6f}",
                f"{val_loss:.6f}",
                f"{val_metrics['f1']:.6f}",
                f"{val_metrics['accuracy']:.6f}",
                f"{val_metrics['precision']:.6f}",
                f"{val_metrics['recall']:.6f}",
                f"{val_metrics['specificity']:.6f}",
                current_lr,
            ])

        elapsed = time.time() - start_time

        print(
            f"{epoch:>6} "
            f"{train_loss:>9.4f} "
            f"{train_metrics['f1']:>8.4f} "
            f"{val_loss:>9.4f} "
            f"{val_metrics['f1']:>8.4f} "
            f"({elapsed:.1f}s)"
        )

        # Checkpoint selection using validation F1
        if val_metrics["f1"] > best_val_f1:

            best_val_f1 = val_metrics["f1"]

            torch.save(
                model.state_dict(),
                cfg["checkpoint"]
            )

            patience_count = 0

            print(
                f"        [checkpoint] "
                f"Best validation F1: {best_val_f1:.4f}"
            )

        else:

            patience_count += 1

        # Early stopping
        if patience_count >= cfg["patience"]:

            print(
                f"\n[train] Early stopping at epoch {epoch}."
            )

            break

    # Final test evaluation
    # The test set is evaluated only after model selection
    # It is not used for early stopping or checkpoint selection

    if not os.path.exists(cfg["checkpoint"]):
        raise FileNotFoundError(
            "Best model checkpoint was not created."
        )

    state_dict = torch.load(
        cfg["checkpoint"],
        map_location=device
    )

    model.load_state_dict(state_dict)

    test_loss, test_metrics = run_epoch(
        model,
        test_loader,
        criterion,
        device,
    )

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)

    print(f"Test loss:           {test_loss:.4f}")

    for name, value in test_metrics.items():
        print(f"{name:20s}: {value:.4f}")

    print(f"\nBest validation F1: {best_val_f1:.4f}")

    print(
        f"Checkpoint: {cfg['checkpoint']}"
    )

    print(
        f"Training log: {log_path}"
    )

    print("\nClassification report:")

    # Recompute predictions for a readable classification report
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for batch in test_loader:

            batch = batch.to(device)

            logits = model(
                batch.x,
                batch.edge_index
            )

            preds = logits.argmax(dim=-1)

            all_preds.extend(
                preds.cpu().numpy()
            )

            all_labels.extend(
                batch.y.cpu().numpy()
            )

    print(
        classification_report(
            all_labels,
            all_preds,
            labels=[0, 1],
            target_names=[
                "Not congested",
                "Congested"
            ],
            zero_division=0
        )
    )

    print("Confusion matrix:")

    print(
        confusion_matrix(
            all_labels,
            all_preds,
            labels=[0, 1]
        )
    )

    print("=" * 60)

    return model

if __name__ == "__main__":
    train()
