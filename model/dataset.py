"""
dataset.py
----------
PyTorch Geometric Dataset for synthetic floorplan graphs.

Each graph:
    - Node features (per macro): [x_norm, y_norm, w_norm, h_norm, area_norm, aspect_ratio]
    - Edge index: connectivity (nets between macros)
    - Edge attr: manhattan distance between macro centres (wire length proxy)
    - Labels: binary congestion per node

Author: [Your Name]
"""

import json
import os
import torch
import numpy as np
from torch_geometric.data import Data, Dataset


CANVAS_W = 100.0
CANVAS_H = 100.0
MAX_AREA  = CANVAS_W * CANVAS_H


class FloorplanDataset(Dataset):
    """
    Custom PyG Dataset wrapping the synthetic floorplan JSON files.

    Usage:
        >>> from dataset import FloorplanDataset
        >>> ds = FloorplanDataset(root='data', split='train')
        >>> g = ds[0]          # first graph
        >>> g.x.shape          # (n_macros, 6)
        >>> g.edge_index.shape # (2, n_edges)
        >>> g.y.shape          # (n_macros,)
    """

    def __init__(self, root: str, split: str = "train", transform=None):
        self.split = split
        self.data_path = os.path.join(root, f"{split}.json")
        with open(self.data_path) as f:
            self.samples = json.load(f)
        super().__init__(root, transform)

    def len(self) -> int:
        return len(self.samples)

    def get(self, idx: int) -> Data:
        sample = self.samples[idx]
        macros = sample["macros"]
        edges  = sample["edges"]
        labels = sample["labels"]

        # ── Node features ─────────────────────────────────────────────────
        # Each node = one macro block
        # Features: [cx_norm, cy_norm, w_norm, h_norm, area_norm, aspect_ratio]
        # Normalised to [0,1] so the GCN doesn't need to worry about scale.
        #
        # Example:
        #   macro at x=20, y=30, w=10, h=5  →  cx=25, cy=32.5
        #   features: [0.25, 0.325, 0.10, 0.05, 0.005, 0.5]
        node_feats = []
        for m in macros:
            cx = (m["x"] + m["w"] / 2) / CANVAS_W
            cy = (m["y"] + m["h"] / 2) / CANVAS_H
            w  = m["w"] / CANVAS_W
            h  = m["h"] / CANVAS_H
            area = (m["w"] * m["h"]) / MAX_AREA
            aspect = min(m["w"], m["h"]) / max(m["w"], m["h"])  # in (0,1]
            node_feats.append([cx, cy, w, h, area, aspect])

        x = torch.tensor(node_feats, dtype=torch.float)  # shape: (N, 6)

        # ── Edge index ────────────────────────────────────────────────────
        if edges:
            src = [e[0] for e in edges]
            dst = [e[1] for e in edges]
            # undirected: add both directions
            edge_index = torch.tensor(
                [src + dst, dst + src], dtype=torch.long
            )
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        # ── Edge attributes: Manhattan wire length ────────────────────────
        # Used as an edge feature; longer wire → more routing resource demand.
        # Example: two macros with centres (10,10) and (30,40) → distance = 50
        edge_attr_list = []
        centres = [(m["x"] + m["w"] / 2, m["y"] + m["h"] / 2) for m in macros]
        max_dist = CANVAS_W + CANVAS_H
        for e in edges:
            i, j = e
            dx = abs(centres[i][0] - centres[j][0])
            dy = abs(centres[i][1] - centres[j][1])
            d  = (dx + dy) / max_dist  # normalised
            edge_attr_list.extend([d, d])  # both directions
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float).unsqueeze(1)

        # ── Labels ────────────────────────────────────────────────────────
        y = torch.tensor(labels, dtype=torch.long)  # shape: (N,)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
