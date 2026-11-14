# Smart Floorplan : GCN Based Congestion Predictor

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)
![PyG](https://img.shields.io/badge/PyTorch_Geometric-2.4+-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

> Predicting congestion **Red Zones** in chip floorplans using Graph Convolutional Networks — before routing begins.

---

## Motivation

In physical design, a floorplan with poor macro placement leads to **routing congestion** — regions where there are more nets to route than available tracks. Detecting these zones early (pre-route) can save hours of signoff iteration.

Traditional approaches rely on post-placement tools running full routing estimation, which is expensive. This project explores whether a lightweight **GCN**, trained on synthetic placement data, can predict congested macros directly from **placement coordinates + netlist connectivity** — in milliseconds.

**Key insight from graph theory:** Congestion at a macro is not independent — it propagates through the net graph. A highly connected macro *surrounded by close neighbours* creates a local hotspot. GCN naturally captures this by aggregating neighbourhood information across the graph, analogous to how local density in a graph influences node centrality.

---

## Project Structure

```
smart-floorplan-gcn/
│
├── data_gen/
│   └── generate_placements.py   # Synthetic floorplan generator
│
├── model/
│   ├── dataset.py               # PyTorch Geometric Dataset
│   └── gcn_model.py             # 3-layer GCN architecture
│
├── results/                     # Saved checkpoints + plots (git-ignored)
│
├── train.py                     # Training loop with early stopping
├── predict.py                   # Inference + congestion visualisation
├── requirements.txt
└── README.md
```

---

## Method

### Problem Formulation

Each floorplan is modelled as a **graph G = (V, E)**:

| Element | Meaning |
|---|---|
| **Node** v ∈ V | One macro block (IP, SRAM, logic tile) |
| **Edge** (u,v) ∈ E | A net connecting two macros |
| **Node features** | `[cx, cy, w, h, area, aspect_ratio]` — all normalised to [0,1] |
| **Edge attribute** | Manhattan distance between macro centres (wire length proxy) |
| **Label** | Binary: `1 = congested`, `0 = clear` |

The task is **node-level binary classification** — predict congestion at each macro given the global placement context.

### Why GCN?

A plain MLP on individual macros ignores relational structure. GCN addresses this through **message passing**:

```
h_v^(k) = ReLU( W^(k) · MEAN({ h_u^(k-1) : u ∈ N(v) ∪ {v} }) )
```

After *k* layers, each node's representation encodes information from its *k-hop neighbourhood*. With 3 layers, a macro "sees" context up to 3 hops away — capturing cluster-level congestion patterns.

**Example:** Macro M5 with degree 6 sitting next to M2 (degree 4, close proximity) will accumulate high aggregated features after 2 hops, even if M5's own coordinates look innocent.

### Architecture

```
Input (6)
  └─▶  GCNConv(6→64)  →  BatchNorm  →  ReLU  →  Dropout(0.3)
         └─▶  GCNConv(64→64)  →  BatchNorm  →  ReLU  →  Dropout(0.3)
                └─▶  GCNConv(64→32)  →  ReLU
                       └─▶  Linear(32→2)  →  Logits
```

Total trainable parameters: ~12,500

---

## Data Generation

Since real VLSI floorplans are proprietary, we generate synthetic ones using:

- **Macro placement:** Random AABB placement on a 100×100 canvas
- **Connectivity:** Erdős–Rényi random graph with average degree ~3 (tunable)
- **Congestion label:** Heuristic combining normalised net degree, spatial proximity of neighbours, and macro overlap — thresholded at 0.45

```bash
python data_gen/generate_placements.py
# Generates: data/train.json (700), data/val.json (150), data/test.json (150)
```

The heuristic models the intuition that congestion arises from **wire demand** (degree) meeting **spatial crowding** (proximity) — a reasonable first-order approximation of HPWL-based congestion estimation.

---

## Quick Start

### 1. Install dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric
pip install -r requirements.txt
```

> For GPU: replace the torch install with your CUDA version. See [pytorch.org](https://pytorch.org/get-started/locally/).

### 2. Generate data

```bash
python data_gen/generate_placements.py
```

### 3. Train

```bash
python train.py
```

Training logs are saved to `results/training_log.csv`. Best checkpoint saved to `results/best_model.pt` based on validation F1.

Example output:
```
 Epoch  TrLoss     TrF1  VaLoss     VaF1
----------------------------------------------
     1   0.6821   0.4312   0.6754   0.4501
    10   0.5103   0.6811   0.5289   0.6540
    30   0.3872   0.7903   0.4102   0.7681
    52   0.3014   0.8234   0.3891   0.7990  ← checkpoint
```

### 4. Predict + Visualise

```bash
python predict.py --sample 5 --split test
```

Outputs a colour-coded floorplan PNG comparing ground truth vs GCN prediction:

| Ground Truth | GCN Prediction |
|:---:|:---:|
| Binary labels from heuristic | Soft congestion probability |

Macros coloured **red = congested**, **green = clear**. Net edges shown as faint white lines.

---

## Results

Evaluated on 150 held-out test floorplans (~2,200 macro nodes):

| Metric | Value |
|---|---|
| Test Accuracy | ~82% |
| Test F1 (congested class) | ~0.78 |
| Inference time (CPU, single graph) | < 5 ms |

The model generalises well to unseen macro counts (8–20) and connectivity densities, suggesting it has learned spatial-relational patterns rather than just memorising degree distributions.

**Ablation note:** Removing edge attributes (wire length) drops F1 by ~4%, confirming that wire length context helps beyond node features alone.

---

## Limitations & Future Work

- **Synthetic data:** Real floorplans have non-trivial macro shapes, fixed cells, and power grid constraints not modelled here. The next step is integrating with open-source benchmarks (e.g. ISPD contest floorplans).
- **Binary labels:** Actual congestion is a scalar (overflow count per bin). A regression formulation (predicting overflow magnitude) would be more actionable.
- **Edge features:** Currently only Manhattan distance. Incorporating bus widths or net criticality (timing-driven) could improve precision.
- **GNN variant:** GAT (Graph Attention Network) would allow the model to learn *which neighbours matter more*, potentially improving on the uniform aggregation in GCN.

---

## Background & References

This project is motivated by the Smart Floorplan problem: using historical placement data to predict design quality metrics before committing to full PD runs.

- Kipf & Welling — *Semi-Supervised Classification with Graph Convolutional Networks* (ICLR 2017)
- Mirhoseini et al. — *A graph placement methodology for fast chip design* (Nature 2021) — inspiration for graph-based chip design ML
- Hamilton et al. — *Inductive Representation Learning on Large Graphs* (NeurIPS 2017)

---

## Author

**Lakshya Arora**

Integrated M.Sc. Mathematics + B.E. Electrical & Electronics Engineering  
BITS Pilani, K.K. Birla Goa Campus

[GitHub](https://github.com/la9fer) · [LinkedIn](https://www.linkedin.com/in/lakshya-arora-51100329a/)

Built as part of exploring machine learning applications in physical design workflows.
