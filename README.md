# AgriEdge-IDS

Reference implementation of:

> **Multimodal edge intelligence with graph attention for resilient
> intrusion detection in agricultural IoT systems**
> Kangde Li, Ziye Wang, Peiling Zhou, Man Ding, Lu Peng, Shouliang Lai
> College of Packaging Design and Art, Hunan University of Technology

This repository is provided as the **reproducibility artifact** for the
manuscript, in response to the reviewers' request for source code,
dataset access, and supporting configuration/scripts.

- **Source code**: this repository (implements Sections 3.2-3.5 in full).
- **Dataset**: [AgriMultiSense v1.0](https://github.com/dunxiandu/AgriMultiSense/releases/tag/v1.0)
  (the raw multimodal sensing assets referenced in Sec. 4.1 as the basis
  for the AgriEdge-MMID benchmark; see `scripts/download_dataset.sh` and
  `docs/DATASET_SCHEMA.md`).
- **Configs / hyperparameters**: `configs/default.yaml` mirrors Table 2
  (training configuration) and Table 3 (model hyperparameters) exactly.

## What is implemented

| Paper section | Module |
|---|---|
| 3.2 Reliability-Aware Edge Visual Perception | `src/models/visual.py` |
| 3.3 IoT Sensor Modeling & Temporal Representation | `src/models/iot.py` |
| 3.4.1-3.4.2 Heterogeneous graph construction & edge weighting | `src/models/graph_build.py` |
| 3.4.3-3.4.4 Heterogeneous graph attention + masked attention | `src/models/gat.py` |
| 3.4.5 Hierarchical reliability-aware temporal readout | `src/models/readout.py` |
| 3.5 Intrusion decision, confidence stabilization, adaptive alerting | `src/models/decision.py` |
| End-to-end streaming model | `src/models/model.py` |
| Training (Table 2: AdamW, focal loss, reduce-on-plateau, early stopping) | `src/train.py`, `src/losses.py` |
| Evaluation (Sec. 4.3: P/R/F1/AUC, latency, false-alarm rate, paired t-test) | `src/evaluate.py` |
| Ablations (Table 37/38: w/o IoT fusion, w/o GAT layer, w/o temporal window) | `src/ablation.py` |

## Installation

```bash
git clone <this-repository-url> agri-edge-ids
cd agri-edge-ids
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.10+, PyTorch 2.1+. A CUDA GPU is used automatically if
available (`src/utils.get_device`); the reference hardware in the paper was
an Intel Core i9 workstation with an NVIDIA RTX 4090 for training and an
NVIDIA Jetson Nano (INT8 quantized) for edge-deployment latency
measurements (Sec. 4.2, Table 39).

## Quick start (synthetic data, no download required)

This runs the entire pipeline end-to-end in under a minute on CPU, so you
can verify the code works before committing to the full dataset download:

```bash
python scripts/make_synthetic_dataset.py --out data/synthetic \
    --num-sequences 4 --windows-per-sequence 20

python -m src.train --config configs/default.yaml \
    --train-manifest data/synthetic/train.jsonl \
    --val-manifest data/synthetic/val.jsonl \
    --image-root data/synthetic \
    --epochs 3

python -m src.evaluate --config configs/default.yaml \
    --checkpoint runs/agri_edge_ids_full/seed0/best_model.pt \
    --test-manifest data/synthetic/test.jsonl \
    --image-root data/synthetic
```

Run the unit tests directly:

```bash
pytest -q
```

## Reproducing the paper's results on the real dataset

1. **Download the raw AgriMultiSense v1.0 release assets:**

   ```bash
   bash scripts/download_dataset.sh data/AgriMultiSense_raw
   ```

2. **Build the AgriEdge-MMID manifests** (JSONL files consumed by
   `src/data/dataset.py`) from the raw assets:

   ```bash
   cp configs/dataset_layout.example.json configs/dataset_layout.json
   # edit configs/dataset_layout.json to match the extracted directory
   # structure on your machine (glob patterns for the IoT/environmental
   # CSVs; see docs/DATASET_SCHEMA.md)
   python scripts/build_manifest.py --layout configs/dataset_layout.json \
       --out data/AgriEdge-MMID
   ```

   The manifest builder assembles the IoT/environmental side of each
   10-second event window automatically (Sec. 4.1.1 synchronization
   strategy: farm-independent splitting, 50 ms max cross-modal offset).
   Image ROI attachment and ground-truth label assignment are
   dataset-release-specific and are marked with `TODO` in
   `scripts/build_manifest.py` for you to complete against the exact
   annotation format shipped with the assets you downloaded.

3. **Train** (repeat once per seed in `configs/default.yaml ->
   experiment.seed_list`; the paper reports mean +/- std over 30 seeds,
   Sec. 4.3):

   ```bash
   for seed in 0 1 2 3 4; do
     python -m src.train --config configs/default.yaml --seed $seed \
       --train-manifest data/AgriEdge-MMID/train.jsonl \
       --val-manifest data/AgriEdge-MMID/val.jsonl \
       --image-root data/AgriEdge-MMID
   done
   ```

4. **Evaluate** each seed's checkpoint and aggregate (mean +/- std) to
   reproduce Table 4 / Figure 4 style results:

   ```bash
   python -m src.evaluate --config configs/default.yaml \
       --checkpoint runs/agri_edge_ids_full/seed0/best_model.pt \
       --test-manifest data/AgriEdge-MMID/test.jsonl \
       --image-root data/AgriEdge-MMID
   ```

5. **Ablations** (Table 37/38):

   ```bash
   python -m src.ablation --config configs/default.yaml \
       --test-manifest data/AgriEdge-MMID/test.jsonl \
       --image-root data/AgriEdge-MMID \
       --checkpoint runs/agri_edge_ids_full/seed0/best_model.pt
   ```

6. **Field-deployment / edge-vs-cloud latency comparison** (Table 39):
   evaluate the same checkpoint on `field.jsonl` (the 30-day, 4,380-event
   field-validation benchmark, held out from training per Sec. 4.1) with
   `model.quantization` in the config set to `int8` for the edge setting;
   compare `mean_latency_ms` from `src/evaluate.py`'s output against an
   FP32 run to reproduce the edge-vs-cloud comparison.

## Notes on batching

Each event window is processed individually because its multimodal graph
has a variable number of nodes (visual ROIs + active IoT sensors + one
environment node); `src/train.py` and `src/evaluate.py` therefore iterate
window-by-window within each sequence while threading the temporal graph
memory (Sec. 3.4.5) and confidence stabilizer (Sec. 3.5) state across
windows, and accumulate a sequence-level loss before a single optimizer
step. `configs/default.yaml`'s `data.batch_size: 16` refers to the number
of event windows aggregated per gradient step at the paper's reported
scale; on limited hardware, reduce sequence length or use gradient
accumulation across multiple sequences to match this effective batch size.

## Repository layout

```
configs/                  Hyperparameters (Table 2/3) + dataset layout template
docs/DATASET_SCHEMA.md     Manifest format reference
scripts/
  download_dataset.sh      Downloads AgriMultiSense v1.0 release assets
  build_manifest.py        Raw assets -> AgriEdge-MMID JSONL manifests
  make_synthetic_dataset.py  Synthetic manifest generator for smoke tests
src/
  data/dataset.py           Manifest-driven PyTorch Dataset
  models/
    visual.py               Sec. 3.2 reliability-aware visual perception
    iot.py                  Sec. 3.3 IoT sensor modeling
    graph_build.py           Sec. 3.4.1-3.4.2 graph construction
    gat.py                   Sec. 3.4.3-3.4.4 heterogeneous graph attention
    readout.py               Sec. 3.4.5 hierarchical temporal readout
    decision.py              Sec. 3.5 decision & alerting
    model.py                 End-to-end streaming model
  losses.py                  Focal loss (Table 2)
  train.py / evaluate.py / ablation.py
tests/test_shapes.py        Shape/smoke tests, no dataset required
```
