"""
predict.py
----------
Load a trained FloorplanGCN checkpoint and run inference on a single
floorplan sample, then visualise the predicted congestion heatmap.

Usage:
    python predict.py --sample 0 --split test

Output:
    results/prediction_sample_<id>.png   — colour-coded floorplan
    Prints per-macro congestion probabilities to stdout.

Author: [Your Name]
"""

import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from model.dataset import FloorplanDataset
from model.gcn_model import FloorplanGCN

CANVAS_W = 100.0
CANVAS_H = 100.0
CHECKPOINT = "results/best_model.pt"


def load_model(checkpoint: str, device: torch.device) -> FloorplanGCN:
    model = FloorplanGCN(in_channels=6, hidden_dim=64)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model.to(device)


@torch.no_grad()
def predict_sample(model, data, device) -> np.ndarray:
    """
    Run inference on a single PyG Data object.
    Returns congestion probability per node.

    Example:
        >>> probs = predict_sample(model, data, device)
        >>> probs.shape   # (n_macros,)
        >>> probs[3]      # 0.82 → macro 3 is 82% likely to be congested
    """
    data = data.to(device)
    probs = model.predict_proba(data.x, data.edge_index)
    return probs[:, 1].cpu().numpy()  # probability of class 1 (congested)


def visualise(sample_raw: dict, probs: np.ndarray, out_path: str):
    """
    Draw the floorplan with each macro coloured by its congestion probability.
    Red = highly congested, green = clear.

    Also draws net edges as thin grey lines between macro centres.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    cmap = plt.cm.RdYlGn_r  # Red→Yellow→Green (high prob = red)
    norm = Normalize(vmin=0, vmax=1)

    titles = ["Ground Truth", "GCN Prediction"]
    value_sources = [sample_raw["labels"], probs.tolist()]

    for ax, title, values in zip(axes, titles, value_sources):
        ax.set_xlim(0, CANVAS_W)
        ax.set_ylim(0, CANVAS_H)
        ax.set_aspect("equal")
        ax.set_facecolor("#1a1a2e")
        ax.set_title(title, fontsize=14, fontweight="bold", color="white", pad=10)
        ax.tick_params(colors="grey")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

        macros = sample_raw["macros"]

        # Draw net edges
        for e in sample_raw["edges"]:
            i, j = e
            cx_i = macros[i]["x"] + macros[i]["w"] / 2
            cy_i = macros[i]["y"] + macros[i]["h"] / 2
            cx_j = macros[j]["x"] + macros[j]["w"] / 2
            cy_j = macros[j]["y"] + macros[j]["h"] / 2
            ax.plot([cx_i, cx_j], [cy_i, cy_j], color="white", alpha=0.12, lw=0.8, zorder=1)

        # Draw macros
        for idx, m in enumerate(macros):
            v = values[idx]
            # binary label → use 0/1 directly; float probs → use as-is
            color = cmap(norm(float(v)))
            rect = patches.FancyBboxPatch(
                (m["x"], m["y"]), m["w"], m["h"],
                boxstyle="round,pad=0.5",
                linewidth=1.5,
                edgecolor="white",
                facecolor=color,
                alpha=0.85,
                zorder=2,
            )
            ax.add_patch(rect)
            ax.text(
                m["x"] + m["w"] / 2, m["y"] + m["h"] / 2,
                f"M{idx}", ha="center", va="center",
                fontsize=7, color="white", fontweight="bold", zorder=3
            )

    # Shared colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.02, pad=0.02)
    cbar.set_label("Congestion Probability", color="white", fontsize=11)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    fig.patch.set_facecolor("#0f0f1a")
    plt.suptitle(
        f"Floorplan Congestion Map  (Sample #{sample_raw['id']})",
        fontsize=15, fontweight="bold", color="white", y=1.01
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[predict] Saved visualisation → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="FloorplanGCN Inference")
    parser.add_argument("--sample", type=int, default=0, help="Sample index")
    parser.add_argument("--split",  type=str, default="test", choices=["train","val","test"])
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    ds = FloorplanDataset(root="data", split=args.split)
    data = ds[args.sample]

    # Load raw sample for visualisation (we need x,y,w,h)
    with open(f"data/{args.split}.json") as f:
        raw = json.load(f)
    sample_raw = raw[args.sample]

    probs = predict_sample(model, data, device)

    print(f"\n{'Macro':>6}  {'GT':>4}  {'P(congested)':>14}")
    print("-" * 30)
    for i, (gt, p) in enumerate(zip(sample_raw["labels"], probs)):
        flag = "🔴" if p > 0.5 else "🟢"
        print(f"  M{i:>3}  {'C' if gt else ' ':>4}  {p:>12.4f}  {flag}")

    import os
    os.makedirs("results", exist_ok=True)
    out_path = f"results/prediction_sample_{sample_raw['id']}.png"
    visualise(sample_raw, probs, out_path)


if __name__ == "__main__":
    main()
