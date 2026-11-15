
"""
Load a trained FloorplanGCN checkpoint and run inference on a single
floorplan sample, then visualise ground-truth labels and predicted
congestion probabilities

Usage:
    python predict.py --sample 0 --split test
    python predict.py --sample 5 --split val
    python predict.py --sample 0 --split test --checkpoint results/best_model.pt

Output:
    results/prediction_sample_<id>.png

Prints per-macro congestion probabilities to stdout
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from model.dataset import FloorplanDataset
from model.gcn_model import FloorplanGCN

CANVAS_W = 100.0
CANVAS_H = 100.0
CHECKPOINT = "results/best_model.pt"
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")

def load_model(checkpoint: str, device: torch.device) -> FloorplanGCN:
    """Load a trained model from a state-dict checkpoint"""

    checkpoint_path = Path(checkpoint)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
            "Train the model first or provide checkpoint"
        )

    model = FloorplanGCN(in_channels=6, hidden_dim=64)

    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model

@torch.no_grad()
def predict_sample(model, data, device) -> np.ndarray:
    """
    Run inference on one PyTorch Geometric Data object

    Returns:
        NumPy array containing the predicted congestion probability
        for each macro, with shape (number of macros , )
    """

    data = data.to(device)

    probabilities = model.predict_proba(
        data.x,
        data.edge_index,
    )

    return probabilities[:, 1].cpu().numpy()


def visualise(
    sample_raw: dict,
    probs: np.ndarray,
    out_path: str,
):
    """
    Visualise ground-truth congestion labels and predicted probabilities

    Left : Ground truth binary labels
    Right : Predicted congestion probabilities
    """

    macros = sample_raw["macros"]
    labels = sample_raw["labels"]

    if len(macros) != len(labels) or len(macros) != len(probs):
        raise ValueError(
            "Mismatch between macro count, ground-truth labels, "
            "and predicted probabilities"
        )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    cmap = plt.cm.RdYlGn_r
    norm = Normalize(vmin=0, vmax=1)

    titles = [
        "Ground Truth (Binary Labels)",
        "GCN Prediction (Probabilities)",
    ]

    value_sources = [
        labels,
        probs.tolist(),
    ]

    for ax, title, values in zip(axes, titles, value_sources):

        ax.set_xlim(0, CANVAS_W)
        ax.set_ylim(0, CANVAS_H)
        ax.set_aspect("equal")
        ax.set_facecolor("#1a1a2e")

        ax.set_title(
            title,
            fontsize=13,
            fontweight="bold",
            color="white",
            pad=10,
        )

        ax.tick_params(colors="grey")

        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

        # Draw graph edges
        for edge in sample_raw["edges"]:
            i, j = edge

            if not (0 <= i < len(macros) and 0 <= j < len(macros)):
                raise ValueError(f"Invalid edge indices: {edge}")

            macro_i = macros[i]
            macro_j = macros[j]

            cx_i = macro_i["x"] + macro_i["w"] / 2
            cy_i = macro_i["y"] + macro_i["h"] / 2

            cx_j = macro_j["x"] + macro_j["w"] / 2
            cy_j = macro_j["y"] + macro_j["h"] / 2

            ax.plot(
                [cx_i, cx_j],
                [cy_i, cy_j],
                color="white",
                alpha=0.12,
                linewidth=0.8,
                zorder=1,
            )

        # Draw macros
        for idx, macro in enumerate(macros):

            value = float(values[idx])
            color = cmap(norm(value))

            rect = patches.FancyBboxPatch(
                (macro["x"], macro["y"]),
                macro["w"],
                macro["h"],
                boxstyle="round,pad=0.5",
                linewidth=1.5,
                edgecolor="white",
                facecolor=color,
                alpha=0.85,
                zorder=2,
            )

            ax.add_patch(rect)

            ax.text(
                macro["x"] + macro["w"] / 2,
                macro["y"] + macro["h"] / 2,
                f"M{idx}",
                ha="center",
                va="center",
                fontsize=7,
                color="white",
                fontweight="bold",
                zorder=3,
            )

    # Shared colorbar
    scalar_map = ScalarMappable(cmap=cmap, norm=norm)
    scalar_map.set_array([])

    colorbar = fig.colorbar(
        scalar_map,
        ax=axes,
        orientation="vertical",
        fraction=0.02,
        pad=0.02,
    )

    colorbar.set_label(
        "Congestion Score / Probability",
        color="white",
        fontsize=10,
    )

    colorbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(
        colorbar.ax.yaxis.get_ticklabels(),
        color="white",
    )

    fig.patch.set_facecolor("#0f0f1a")

    fig.suptitle(
        f"Floorplan Congestion Map (Sample #{sample_raw['id']})",
        fontsize=15,
        fontweight="bold",
        color="white",
        y=1.01,
    )

    plt.tight_layout()

    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )

    plt.close(fig)

    print(f"[predict] Saved visualisation -> {output_path}")

def main():

    parser = argparse.ArgumentParser(
        description="FloorplanGCN Inference"
    )

    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Sample index within the selected split",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=CHECKPOINT,
        help="Path to trained model checkpoint",
    )

    args = parser.parse_args()

    if args.sample < 0:
        parser.error("--sample must be non-negative")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"[predict] Device: {device}")
    print(f"[predict] Split: {args.split}")
    print(f"[predict] Sample index: {args.sample}")

    # Load model
    model = load_model(args.checkpoint, device)

    # Load processed graph dataset
    dataset = FloorplanDataset(
        root=str(DATA_DIR),
        split=args.split,
    )

    if args.sample >= len(dataset):
        raise IndexError(
            f"Sample index {args.sample} is out of range. "
            f"The {args.split} split contains {len(dataset)} samples."
        )

    data = dataset[args.sample]

    # Load raw sample for geometry and visualisation
    raw_path = DATA_DIR / f"{args.split}.json"

    if not raw_path.is_file():
        raise FileNotFoundError(
            f"Raw dataset file not found: {raw_path}"
        )

    with open(raw_path, "r") as file:
        raw_samples = json.load(file)

    if args.sample >= len(raw_samples):
        raise IndexError(
            "Processed dataset and raw JSON dataset have different lengths."
        )

    sample_raw = raw_samples[args.sample]

    # Run inference
    probabilities = predict_sample(
        model,
        data,
        device,
    )

    if len(probabilities) != len(sample_raw["macros"]):
        raise ValueError(
            "Prediction count does not match the number of macros."
        )

    # Print per-macro predictions
    print(
        f"{'Macro':>6}  {'GT':>4}  "
        f"{'P(congested)':>14}  {'Prediction':>12}"
    )
    print("-" * 48)

    for idx, (ground_truth, probability) in enumerate(
        zip(sample_raw["labels"], probabilities)
    ):

        prediction = int(probability >= 0.5)
        flag = "Congested" if prediction else "Clear"

        print(
            f"M{idx:>3}  "
            f"{int(ground_truth):>4}  "
            f"{probability:>14.4f}  "
            f"{flag:>12}"
        )

    # Generate visualisation
    output_path = (
        RESULTS_DIR
        / f"prediction_sample_{sample_raw['id']}.png"
    )

    visualise(
        sample_raw,
        probabilities,
        str(output_path),
    )

if __name__ == "__main__":
    main()
