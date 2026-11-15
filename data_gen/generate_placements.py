
"""
Generates synthetic floorplan placement data for training a
Graph Convolutional Network (GCN).

Each sample represents a synthetic chip floorplan containing:
    - Randomly placed rectangular macros on a 2D canvas.
    - A connectivity graph generated using an Erdős–Rényi model.
    - A heuristic congestion score for each macro.
    - A binary congestion label derived from the score.

Important:
    Congestion labels are generated using a geometric and
    connectivity-based heuristic. They are not measurements
    from a physical-design or routing engine.

    Macro overlaps are permitted by the placement generator.

    The connectivity graph is synthetic and is not derived
    from an actual chip netlist.
"""

import json
import os
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np

# Reproducibility
SEED = 42

np.random.seed(SEED)

# Canvas and macro configuration
CANVAS_W = 100.0
CANVAS_H = 100.0

MIN_MACROS = 8
MAX_MACROS = 20

MIN_MACRO_W = 4.0
MAX_MACRO_W = 15.0

MIN_MACRO_H = 4.0
MAX_MACRO_H = 15.0

# Congestion heuristic configuration
DEGREE_WEIGHT = 0.4
PROXIMITY_WEIGHT = 0.4
OVERLAP_WEIGHT = 0.2

CONGESTION_THRESHOLD = 0.45

def place_macros(n: int) -> List[Dict]:
    """
    Randomly place n rectangular macros on the floorplan canvas.

    Macro positions and dimensions are sampled independently.

    Note:
        This generator does not enforce non-overlapping placement.

    Returns:
        A list of dictionaries containing:
        id, x, y, w and h.
    """

    if n <= 0:
        raise ValueError("Number of macros must be positive.")

    macros = []

    for i in range(n):

        w = np.random.uniform(
            MIN_MACRO_W,
            MAX_MACRO_W
        )

        h = np.random.uniform(
            MIN_MACRO_H,
            MAX_MACRO_H
        )

        x = np.random.uniform(
            0,
            CANVAS_W - w
        )

        y = np.random.uniform(
            0,
            CANVAS_H - h
        )

        macros.append({
            "id": i,
            "x": float(x),
            "y": float(y),
            "w": float(w),
            "h": float(h)
        })

    return macros

def build_connectivity(
    macros: List[Dict],
    avg_degree: float = 3.0
) -> List[Tuple[int, int]]:
    """
    Generate a synthetic undirected connectivity graph.

    Uses the Erdős–Rényi random graph model G(n, p).

    The probability is selected as:

        p = avg_degree / (n - 1)

    This gives an expected node degree approximately equal
    to avg_degree.

    Each edge (i, j) represents a synthetic connection
    between two macros.

    Returns:
        List of undirected edges represented as node-index pairs.
    """

    n = len(macros)

    if n < 2:
        return []

    if avg_degree < 0:
        raise ValueError("Average degree cannot be negative.")

    p = avg_degree / (n - 1)

    if p > 1:
        raise ValueError(
            "Average degree is too large for the number of nodes."
        )

    graph_seed = int(
        np.random.randint(0, 10000)
    )

    graph = nx.erdos_renyi_graph(
        n,
        p,
        seed=graph_seed
    )

    return list(graph.edges())

def compute_congestion(
    macros: List[Dict],
    edges: List[Tuple[int, int]]
) -> np.ndarray:
    """
    Calculate a heuristic congestion score for every macro.

    The score combines three components:

        1. Normalized node degree:
           Measures the number of synthetic connections.

        2. Neighbour proximity:
           Closer connected macros contribute a higher score.

        3. Geometric overlap:
           Measures the number of overlapping macro rectangles.

    Score formula:

        score = 0.4 * normalized_degree
              + 0.4 * neighbour_proximity
              + 0.2 * overlap_score

    The resulting scores are used to generate binary labels
    using CONGESTION_THRESHOLD.

    Returns:
        NumPy array containing one score per macro.

    Important:
        This is a synthetic heuristic, not a physical routing
        congestion measurement.
    """

    n = len(macros)

    if n == 0:
        raise ValueError("At least one macro is required.")

    scores = np.zeros(n, dtype=np.float64)

    # 1. Degree-based component
    degrees = np.zeros(n, dtype=np.float64)

    for u, v in edges:

        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(
                f"Invalid edge ({u}, {v}) for {n} macros."
            )

        if u == v:
            raise ValueError("Self-loops are not supported.")

        degrees[u] += 1
        degrees[v] += 1

    max_deg = degrees.max()

    if max_deg > 0:
        norm_degree = degrees / max_deg
    else:
        norm_degree = degrees

    # 2. Neighbour proximity component
    neighbours = [[] for _ in range(n)]

    for u, v in edges:
        neighbours[u].append(v)
        neighbours[v].append(u)

    centres = [
        (
            m["x"] + m["w"] / 2,
            m["y"] + m["h"] / 2
        )
        for m in macros
    ]

    max_possible_distance = CANVAS_W + CANVAS_H

    for i in range(n):

        if not neighbours[i]:

            proximity = 0.0

        else:

            distances = []

            cx_i, cy_i = centres[i]

            for j in neighbours[i]:

                cx_j, cy_j = centres[j]

                distance = (
                    abs(cx_i - cx_j)
                    + abs(cy_i - cy_j)
                )

                distances.append(distance)

            avg_distance = np.mean(distances)

            proximity = (
                1.0
                - avg_distance / max_possible_distance
            )

        # 3. Geometric overlap component
        overlap_count = 0

        xi1 = macros[i]["x"]
        xi2 = xi1 + macros[i]["w"]

        yi1 = macros[i]["y"]
        yi2 = yi1 + macros[i]["h"]

        for j in range(n):

            if i == j:
                continue

            xj1 = macros[j]["x"]
            xj2 = xj1 + macros[j]["w"]

            yj1 = macros[j]["y"]
            yj2 = yj1 + macros[j]["h"]

            # Axis-aligned bounding-box intersection.
            if (
                xi1 < xj2
                and xi2 > xj1
                and yi1 < yj2
                and yi2 > yj1
            ):
                overlap_count += 1

        overlap_score = min(
            overlap_count / 3.0,
            1.0
        )

        # 4. Combined heuristic score
        scores[i] = (
            DEGREE_WEIGHT * norm_degree[i]
            + PROXIMITY_WEIGHT * proximity
            + OVERLAP_WEIGHT * overlap_score
        )

    return scores

def generate_sample(sample_id: int) -> Dict:
    """
    Generate one complete synthetic floorplan sample.

    Returns:
        Dictionary containing:
            id
            n_macros
            macros
            edges
            congestion_scores
            labels

    Binary label mapping:
        0 = Not congested
        1 = Congested
    """

    n = int(
        np.random.randint(
            MIN_MACROS,
            MAX_MACROS + 1
        )
    )

    macros = place_macros(n)

    avg_degree = float(
        np.random.uniform(2.0, 4.5)
    )

    edges = build_connectivity(
        macros,
        avg_degree=avg_degree
    )

    scores = compute_congestion(
        macros,
        edges
    )

    labels = (
        scores > CONGESTION_THRESHOLD
    ).astype(int)

    return {
        "id": sample_id,
        "n_macros": n,
        "macros": macros,
        "edges": [
            list(edge)
            for edge in edges
        ],
        "congestion_scores": scores.tolist(),
        "labels": labels.tolist()
    }

def generate_dataset(
    n_samples: int,
    split: str = "train",
    out_dir: str = "data"
) -> str:
    """
    Generate and save a synthetic floorplan dataset.

    Args:
        n_samples: Number of floorplan graphs to generate.
        split: Dataset partition name.
        out_dir: Directory in which the JSON file is saved.

    Returns:
        Path to the generated JSON file.
    """

    if n_samples <= 0:
        raise ValueError(
            "Number of samples must be positive."
        )

    os.makedirs(
        out_dir,
        exist_ok=True
    )

    samples = [
        generate_sample(i)
        for i in range(n_samples)
    ]

    out_path = os.path.join(
        out_dir,
        f"{split}.json"
    )

    with open(
        out_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            samples,
            f,
            indent=2
        )

    total_macros = sum(
        sample["n_macros"]
        for sample in samples
    )

    total_congested = sum(
        sum(sample["labels"])
        for sample in samples
    )

    congestion_rate = (
        100.0 * total_congested / total_macros
    )

    print(
        f"[generate] Saved {n_samples} samples -> {out_path}"
    )

    print(
        f"[generate] Total macros: {total_macros}"
    )

    print(
        f"[generate] Congested macros: {total_congested}"
    )

    print(
        f"[generate] Congestion rate: {congestion_rate:.1f}%"
    )
    return out_path

if __name__ == "__main__":

    generate_dataset(
        700,
        split="train",
        out_dir="data"
    )

    generate_dataset(
        150,
        split="val",
        out_dir="data"
    )

    generate_dataset(
        150,
        split="test",
        out_dir="data"
    )

    print("\nDataset generation complete.")
