"""
generate_placements.py
----------------------
Generates synthetic floorplan placement data for training the GCN model.

Each sample represents a chip floorplan with:
- N macros (rectangular blocks) placed on a 2D canvas
- Connectivity graph (nets between macros)
- Ground truth congestion label per macro (computed from placement geometry)

Author: [Your Name]
"""

import numpy as np
import networkx as nx
import json
import os
import random
from typing import Tuple, List, Dict

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ── canvas & macro config ────────────────────────────────────────────────────
CANVAS_W = 100.0   # normalised chip width  (μm equiv.)
CANVAS_H = 100.0   # normalised chip height
MIN_MACROS = 8
MAX_MACROS = 20
MIN_MACRO_W = 4.0
MAX_MACRO_W = 15.0
MIN_MACRO_H = 4.0
MAX_MACRO_H = 15.0


def place_macros(n: int) -> List[Dict]:
    """
    Place n macros randomly on the canvas.
    Returns list of dicts with keys: id, x, y, w, h
    
    Example:
        >>> macros = place_macros(5)
        >>> macros[0]
        {'id': 0, 'x': 12.3, 'y': 45.1, 'w': 8.2, 'h': 6.4}
    """
    macros = []
    for i in range(n):
        w = np.random.uniform(MIN_MACRO_W, MAX_MACRO_W)
        h = np.random.uniform(MIN_MACRO_H, MAX_MACRO_H)
        x = np.random.uniform(0, CANVAS_W - w)
        y = np.random.uniform(0, CANVAS_H - h)
        macros.append({"id": i, "x": x, "y": y, "w": w, "h": h})
    return macros


def build_connectivity(macros: List[Dict], avg_degree: float = 3.0) -> List[Tuple[int, int]]:
    """
    Build a random connectivity graph over macros using Erdős–Rényi model.
    Edge (i,j) means macro i and macro j share a net.

    avg_degree controls density — higher = more wires = more congestion potential.

    Example:
        >>> edges = build_connectivity(macros, avg_degree=3)
        >>> edges[0]
        (0, 3)
    """
    n = len(macros)
    p = avg_degree / (n - 1) if n > 1 else 0
    G = nx.erdos_renyi_graph(n, p, seed=np.random.randint(0, 10000))
    return list(G.edges())


def compute_congestion(macros: List[Dict], edges: List[Tuple[int, int]]) -> np.ndarray:
    """
    Heuristic congestion score per macro.

    A macro is considered congested if:
    1. It has high net degree (many connections) — wire density
    2. Its neighbours are geometrically close (overlap region)
    3. It partially overlaps with other macros (placement quality)

    Score is in [0, 1]; threshold 0.5 → binary label (congested / not).

    The formula combines:
        - Normalised net degree  (wire demand)
        - Average Manhattan distance to neighbours, inverted  (spatial crowding)
        - Overlap penalty
    """
    n = len(macros)
    scores = np.zeros(n)

    # degree-based component
    degrees = np.zeros(n)
    for u, v in edges:
        degrees[u] += 1
        degrees[v] += 1
    max_deg = degrees.max() if degrees.max() > 0 else 1
    norm_degree = degrees / max_deg  # in [0,1]

    # neighbour proximity component
    # closer neighbours → more crowded routing
    for i in range(n):
        cx_i = macros[i]["x"] + macros[i]["w"] / 2
        cy_i = macros[i]["y"] + macros[i]["h"] / 2
        nbrs = [v for u, v in edges if u == i] + [u for u, v in edges if v == i]
        if not nbrs:
            proximity = 0.0
        else:
            dists = []
            for j in nbrs:
                cx_j = macros[j]["x"] + macros[j]["w"] / 2
                cy_j = macros[j]["y"] + macros[j]["h"] / 2
                dists.append(abs(cx_i - cx_j) + abs(cy_i - cy_j))
            # invert: small distance → high proximity score
            avg_dist = np.mean(dists)
            max_possible = CANVAS_W + CANVAS_H
            proximity = 1.0 - (avg_dist / max_possible)

        # overlap penalty: check AABB overlap with every other macro
        overlap_count = 0
        for j in range(n):
            if i == j:
                continue
            # axis-aligned bounding box overlap check
            xi1, xi2 = macros[i]["x"], macros[i]["x"] + macros[i]["w"]
            yi1, yi2 = macros[i]["y"], macros[i]["y"] + macros[i]["h"]
            xj1, xj2 = macros[j]["x"], macros[j]["x"] + macros[j]["w"]
            yj1, yj2 = macros[j]["y"], macros[j]["y"] + macros[j]["h"]
            if xi1 < xj2 and xi2 > xj1 and yi1 < yj2 and yi2 > yj1:
                overlap_count += 1
        overlap_score = min(overlap_count / 3, 1.0)  # cap at 1

        scores[i] = 0.4 * norm_degree[i] + 0.4 * proximity + 0.2 * overlap_score

    return scores


def generate_sample(sample_id: int) -> Dict:
    """
    Generate one complete floorplan sample.

    Returns a dict with:
        - macros: list of macro dicts
        - edges: connectivity edges
        - congestion_scores: float array [0,1]
        - labels: binary int array (1=congested, 0=not)
    """
    n = np.random.randint(MIN_MACROS, MAX_MACROS + 1)
    macros = place_macros(n)
    edges = build_connectivity(macros, avg_degree=np.random.uniform(2.0, 4.5))
    scores = compute_congestion(macros, edges)
    labels = (scores > 0.45).astype(int)

    return {
        "id": sample_id,
        "n_macros": n,
        "macros": macros,
        "edges": [list(e) for e in edges],
        "congestion_scores": scores.tolist(),
        "labels": labels.tolist(),
    }


def generate_dataset(n_samples: int, split: str = "train", out_dir: str = "data") -> str:
    """
    Generate n_samples floorplan graphs and save as JSON.

    Args:
        n_samples: number of floorplans to generate
        split: 'train', 'val', or 'test'
        out_dir: output directory path

    Returns:
        Path to saved file.
    
    Example:
        >>> generate_dataset(500, split='train', out_dir='data')
        'data/train.json'
    """
    os.makedirs(out_dir, exist_ok=True)
    samples = [generate_sample(i) for i in range(n_samples)]
    out_path = os.path.join(out_dir, f"{split}.json")
    with open(out_path, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"[generate] Saved {n_samples} samples → {out_path}")
    print(f"[generate] Congestion rate: "
          f"{100 * sum(l for s in samples for l in s['labels']) / sum(s['n_macros'] for s in samples):.1f}%")
    return out_path


if __name__ == "__main__":
    generate_dataset(700, split="train", out_dir="data")
    generate_dataset(150, split="val",   out_dir="data")
    generate_dataset(150, split="test",  out_dir="data")
    print("\n✓ Dataset generation complete.")
