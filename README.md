# Smart Floorplan: GCN-Based Congestion Predictor

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)
![PyG](https://img.shields.io/badge/PyTorch_Geometric-2.4+-orange?style=flat-square)
![Status](https://img.shields.io/badge/Status-Development-blue?style=flat-square)

A graph-based machine learning project exploring node-level congestion classification in synthetic chip floorplans using Graph Convolutional Networks (GCNs)

The objective is to investigate whether macro geometry and graph connectivity can provide useful signals for identifying potentially congested regions in a floorplan.

---

## Motivation

In VLSI physical design, macro placement and net connectivity influence the difficulty of subsequent routing. Poor placement can contribute to routing congestion, where routing demand exceeds the available resources.

Identifying potentially problematic regions early could help guide placement decisions and reduce downstream design iterations.

This project explores a lightweight graph-learning approach. Each floorplan is represented as a graph, with macros as nodes and connectivity relationships as edges. A Graph Convolutional Network performs node-level binary classification using macro geometry and neighbourhood information.

The current implementation uses synthetically generated floorplans and heuristic congestion labels. It is an experimental framework rather than a validated physical-design congestion estimator.

---

## Project Structure

```text
smart-floorplan-gcn/
│
├── data_gen/
│   └── generate_placements.py   # Synthetic floorplan generator
│
├── model/
│   ├── dataset.py               # PyTorch Geometric dataset
│   └── gcn_model.py             # GCN architecture
│
├── results/                     # Checkpoints and outputs (git-ignored)
│
├── train.py                     # Training and validation
├── predict.py                   # Inference and visualisation
├── requirements.txt
└── README.md
```

---

## Methodology

### 1. Graph Representation

Each synthetic floorplan is represented as a graph:

\[
G = (V,E)
\]

| Element | Description |
|---|---|
| Node \(v \in V\) | A macro block |
| Edge \((u,v) \in E\) | A connectivity relationship between two macros |
| Node features | Normalised macro geometry |
| Edge attribute | Normalised Manhattan distance between macro centres |
| Label | Binary congestion class: 0 or 1 |

The six node features are:

- Normalised centre x-coordinate
- Normalised centre y-coordinate
- Normalised width
- Normalised height
- Normalised area
- Aspect ratio

The dataset also stores Manhattan distances between connected macro centres as edge attributes. **The current GCN architecture does not consume these edge attributes**; message passing uses the graph connectivity represented by `edge_index`.

The task is node-level binary classification: predicting whether each macro belongs to the congested class under the synthetic labelling rule.

### 2. Why Graph Convolutional Networks?

An individual macro's geometric features do not describe its connectivity to other macros.

Graph Convolutional Networks address this by aggregating information from neighbouring nodes through message passing.

Conceptually, a node representation is updated using information from its neighbourhood:

\[
H^{(k+1)} =
\sigma\left(
\hat{D}^{-1/2}
\hat{A}
\hat{D}^{-1/2}
H^{(k)}W^{(k)}
\right)
\]

where:

- \(\hat{A}\) is the adjacency matrix with self-loops.
- \(\hat{D}\) is the corresponding degree matrix.
- \(H^{(k)}\) contains node representations at layer \(k\).
- \(W^{(k)}\) is a trainable weight matrix.
- \(\sigma\) is a nonlinear activation.

With three graph convolutional layers, each node can incorporate information from its graph neighbourhood up to three hops away.

### 3. Model Architecture

The model consists of three graph convolutional layers followed by a linear classification layer.

```text
Input Node Features (6)
        |
        v
GCNConv (6 -> 64)
        |
   BatchNorm
        |
      ReLU
        |
   Dropout (0.3)
        |
        v
GCNConv (64 -> 64)
        |
   BatchNorm
        |
      ReLU
        |
   Dropout (0.3)
        |
        v
GCNConv (64 -> 32)
        |
      ReLU
        |
        v
Linear (32 -> 2)
        |
        v
   Class Logits
```

**Trainable parameters:** 7,010

The output contains two logits per macro, corresponding to the clear and congested classes.

---

## Synthetic Data Generation

Real industrial floorplans and routing data are not included in this project. Instead, the dataset generator creates synthetic floorplans.

### Generation process

- **Macro placement:** Randomly generated macro positions and dimensions on a 100 × 100 canvas.
- **Connectivity:** Erdős–Rényi random graphs with a configurable average degree.
- **Macro geometry:** Random macro widths and heights.
- **Dataset splits:** Separate training, validation, and test datasets.

The default generation configuration produces:

| Split | Number of floorplans |
|---|---:|
| Training | 700 |
| Validation | 150 |
| Test | 150 |
| **Total** | **1,000** |

### Congestion Label Generation

Congestion labels are generated using a weighted heuristic based on three components:

1. **Connectivity:** Normalised node degree.
2. **Spatial proximity:** Average Manhattan distance to connected neighbours.
3. **Macro overlap:** Number of overlapping macros, subject to a cap.

The components are combined using the following weights:

\[
S(v) =
0.4D(v) +
0.4P(v) +
0.2O(v)
\]

where \(D(v)\), \(P(v)\), and \(O(v)\) represent the normalised degree, proximity, and overlap scores.

A macro is labelled congested when:

\[
S(v) > 0.45
\]

These labels are synthetic heuristic targets. They are not obtained from a physical-design tool, detailed routing, congestion maps, or measured routing overflow.

---

## Quick Start

### 1. Install dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric
pip install -r requirements.txt
```

For GPU execution, install the PyTorch build compatible with your CUDA environment. Refer to the official [PyTorch installation guide](https://pytorch.org/get-started/locally/).

### 2. Generate the dataset

```bash
python data_gen/generate_placements.py
```

This generates:

```text
data/
├── train.json
├── val.json
└── test.json
```

### 3. Train the model

```bash
python train.py
```

Training includes:

- Class-weighted cross-entropy loss.
- Training and validation evaluation.
- Validation-based checkpoint selection.
- Early stopping based on validation performance.
- Final evaluation on the held-out test split.

The best checkpoint is saved at:

```text
results/best_model.pt
```

Training metrics are saved in:

```text
results/training_log.csv
```

### 4. Run inference

```bash
python predict.py --sample 5 --split test
```

You can also specify a checkpoint:

```bash
python predict.py --sample 5 --split test --checkpoint results/best_model.pt
```

The script prints per-macro congestion probabilities and generates a visualisation:

```text
results/prediction_sample_<id>.png
```

The visualisation compares:

- Ground-truth binary labels generated by the heuristic.
- Predicted congestion probabilities from the GCN.

Red indicates a higher congestion score or probability, while green indicates a lower value.

---

## Evaluation

The model is evaluated as a node-level binary classifier.

The training pipeline tracks classification metrics including:

- Accuracy
- Balanced accuracy
- Precision
- Recall
- Specificity
- F1-score
- Matthews correlation coefficient (MCC)

The validation split is used for checkpoint selection and early stopping. The held-out test split is reserved for final evaluation.

**Numerical performance results are not reported here until verified against an actual completed training run and its saved evaluation outputs**

---

## Limitations and Future Work

The current implementation is an exploratory synthetic-data experiment. Several extensions are needed before it can be considered a practical physical-design congestion predictor

### Current limitations

- **Synthetic floorplans:** The generator does not reproduce the full complexity of industrial chip layouts.
- **Heuristic labels:** Congestion targets are derived from manually defined geometric and graph-based rules rather than actual routing outcomes.
- **Random placement:** The generator does not enforce non-overlapping macro placement.
- **Edge attributes:** Manhattan distances are stored in the dataset but are not currently used by the GCN.
- **Simplified connectivity:** Erdős–Rényi graphs do not capture the full structure of real circuit netlists.
- **No physical-design validation:** Predictions have not been validated against routing overflow or congestion maps from an EDA tool.

### Potential extensions

- Integrating open-source VLSI placement and routing benchmarks.
- Generating labels from actual routing congestion maps.
- Incorporating edge attributes into a message-passing architecture.
- Investigating attention-based graph models such as Graph Attention Networks.
- Exploring regression models for predicting continuous congestion metrics.
- Evaluating generalisation across different floorplan sizes and connectivity distributions.

---

## References

- Kipf, T. N. and Welling, M. (2017). *Semi-Supervised Classification with Graph Convolutional Networks*. ICLR.
- Hamilton, W. L., Ying, R. and Leskovec, J. (2017). *Inductive Representation Learning on Large Graphs*. NeurIPS.
- Mirhoseini, A. et al. (2021). *A graph placement methodology for fast chip design*. Nature.

---

## Author

**Lakshya Arora**

Integrated M.Sc. Mathematics + B.E. Electrical & Electronics Engineering  
BITS Pilani, K.K. Birla Goa Campus

[GitHub](https://github.com/la9fer) · [LinkedIn](https://www.linkedin.com/in/lakshya-arora-51100329a/)

An experimental exploration of graph neural networks for machine learning applications in physical design workflows
