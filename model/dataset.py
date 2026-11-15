
"""
dataset.py
----------
PyTorch Geometric Dataset for synthetic chip floorplan graphs.

Each graph contains:

Node features:
    [cx_norm, cy_norm, w_norm, h_norm, area_norm, aspect_ratio]

Graph connectivity:
    edge_index in COO format, with both directions represented.

Edge attributes:
    Normalized Manhattan distance between connected macro centres.

Labels:
    Binary node-level congestion labels:
        0 = Not congested
        1 = Congested

Important:
    Edge attributes are stored in the Data object but are not currently
    consumed by the standard GCNConv architecture in gcn_model.py.

The dataset uses synthetic floorplan data. Its congestion labels
should not be interpreted as physical routing measurements.
"""

import json
import os

import numpy as np
import torch
from torch_geometric.data import Data, Dataset


CANVAS_W = 100.0
CANVAS_H = 100.0
MAX_AREA = CANVAS_W * CANVAS_H


class FloorplanDataset(Dataset):
    """
    PyTorch Geometric Dataset wrapping synthetic floorplan JSON files.

    Expected input structure:
        {
            "macros": [...],
            "edges": [...],
            "labels": [...]
        }

    Args:
        root (str): Directory containing train.json, val.json or
                    the requested split JSON file.
        split (str): Dataset partition name.
        transform: Optional PyTorch Geometric transform.

    Example:
        dataset = FloorplanDataset(root="data", split="train")
        graph = dataset[0]

        graph.x.shape          # (num_nodes, 6)
        graph.edge_index.shape # (2, num_directed_edges)
        graph.edge_attr.shape  # (num_directed_edges, 1)
        graph.y.shape          # (num_nodes,)
    """

    def __init__(self, root: str, split: str = "train", transform=None):

        self.split = split
        self.data_path = os.path.join(root, f"{split}.json")

        if not os.path.isfile(self.data_path):
            raise FileNotFoundError(
                f"Dataset file not found: {self.data_path}"
            )

        with open(self.data_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)

        if not isinstance(self.samples, list):
            raise ValueError(
                "Dataset JSON must contain a list of floorplan samples."
            )

        super().__init__(root, transform)

    def len(self) -> int:
        """Return the number of floorplan graphs."""
        return len(self.samples)

    def get(self, idx: int) -> Data:
        """Convert one JSON floorplan sample into a PyG Data object."""

        sample = self.samples[idx]

        required_keys = {"macros", "edges", "labels"}

        if not isinstance(sample, dict) or not required_keys.issubset(sample):
            raise ValueError(
                f"Sample {idx} must contain macros, edges and labels."
            )

        macros = sample["macros"]
        edges = sample["edges"]
        labels = sample["labels"]

        if not isinstance(macros, list) or not macros:
            raise ValueError(
                f"Sample {idx} must contain at least one macro."
            )

        num_nodes = len(macros)

        if not isinstance(edges, list):
            raise ValueError(
                f"Sample {idx}: edges must be a list."
            )

        if not isinstance(labels, list) or len(labels) != num_nodes:
            raise ValueError(
                f"Sample {idx}: number of labels must match "
                "the number of macros."
            )

        # 1. Node features
        # Features:
        #   1. Normalized macro centre x-coordinate
        #   2. Normalized macro centre y-coordinate
        #   3. Normalized width
        #   4. Normalized height
        #   5. Normalized macro area
        #   6. Aspect ratio
        #
        # The aspect ratio is defined as min(width, height) /
        # max(width, height), giving values in (0, 1]
        # Coordinates and dimensions are normalized relative to
        # the fixed 100 x 100 floorplan canvas

        node_feats = []
        centres = []

        for macro_idx, macro in enumerate(macros):

            try:
                x_pos = float(macro["x"])
                y_pos = float(macro["y"])
                width = float(macro["w"])
                height = float(macro["h"])

            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid macro geometry in sample {idx}, "
                    f"macro {macro_idx}."
                ) from exc

            geometry = np.array(
                [x_pos, y_pos, width, height],
                dtype=np.float64
            )

            if not np.all(np.isfinite(geometry)):
                raise ValueError(
                    f"Non-finite macro geometry in sample {idx}, "
                    f"macro {macro_idx}."
                )

            if width <= 0 or height <= 0:
                raise ValueError(
                    f"Macro dimensions must be positive in sample "
                    f"{idx}, macro {macro_idx}."
                )

            cx = (x_pos + width / 2.0) / CANVAS_W
            cy = (y_pos + height / 2.0) / CANVAS_H

            w_norm = width / CANVAS_W
            h_norm = height / CANVAS_H

            area_norm = (width * height) / MAX_AREA

            aspect_ratio = min(width, height) / max(width, height)

            node_feats.append([
                cx,
                cy,
                w_norm,
                h_norm,
                area_norm,
                aspect_ratio
            ])

            centres.append((
                x_pos + width / 2.0,
                y_pos + height / 2.0
            ))

        x = torch.tensor(
            node_feats,
            dtype=torch.float32
        )

        # 2. Graph connectivity
        # Each edge (i, j) is represented in both directions:
        # i -> j and j -> i
        # This produces an undirected graph representation

        directed_edges = []

        for edge_idx, edge in enumerate(edges):

            if not isinstance(edge, (list, tuple)) or len(edge) != 2:
                raise ValueError(
                    f"Invalid edge at index {edge_idx} in sample {idx}."
                )

            src, dst = edge

            if (
                not isinstance(src, (int, np.integer))
                or not isinstance(dst, (int, np.integer))
            ):
                raise ValueError(
                    f"Edge indices must be integers in sample {idx}."
                )

            if not (0 <= src < num_nodes and 0 <= dst < num_nodes):
                raise ValueError(
                    f"Edge ({src}, {dst}) references a nonexistent "
                    f"node in sample {idx}."
                )

            directed_edges.append((src, dst))
            directed_edges.append((dst, src))

        if directed_edges:

            edge_index = torch.tensor(
                directed_edges,
                dtype=torch.long
            ).t().contiguous()

        else:

            edge_index = torch.empty(
                (2, 0),
                dtype=torch.long
            )

        # 3. Edge attributes
        # Calculate normalized Manhattan distance between macro centres:
        #
        # d(i,j) = (|xi-xj| + |yi-yj|) / (CANVAS_W + CANVAS_H)
        #
        # Both directions receive the same distance
        #
        # These attributes are stored for potential edge-aware models
        # The current GCNConv architecture does not consume them

        max_dist = CANVAS_W + CANVAS_H

        edge_attr_list = []

        for src, dst in directed_edges:

            dx = abs(centres[src][0] - centres[dst][0])
            dy = abs(centres[src][1] - centres[dst][1])

            distance = (dx + dy) / max_dist

            edge_attr_list.append([distance])

        edge_attr = torch.tensor(
            edge_attr_list,
            dtype=torch.float32
        ).reshape(-1, 1)

        # 4. Node-level congestion labels
        try:
            label_array = np.asarray(labels, dtype=np.int64)

        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid labels in sample {idx}."
            ) from exc

        if not np.all(np.isin(label_array, [0, 1])):
            raise ValueError(
                f"Sample {idx}: labels must be binary (0 or 1)."
            )

        y = torch.tensor(
            label_array,
            dtype=torch.long
        )

        # 5. Construct PyTorch Geometric graph
        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y
        )
