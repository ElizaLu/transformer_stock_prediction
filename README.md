# Transformer Stock Prediction

A research-oriented **transformer-based stock prediction** project for cross-sectional market forecasting.  
This repository implements the **MASTER** architecture in PyTorch and is designed around **Qlib-style market data**, **feature gating**, **temporal attention**, **cross-sectional attention**, and **rank-based evaluation metrics**.

> The project is intentionally framed as a reproducible research baseline rather than a simple trading demo.  
> It highlights the kind of work that is valuable in PhD applications: model design, experimental structure, evaluation discipline, and the ability to turn a machine learning idea into a working research pipeline.

---

## Why this project matters for PhD applications

This repository is relevant to PhD-level research in:

- **Machine learning for financial time series**
- **Transformer architectures for structured sequential data**
- **Representation learning with attention and gating**
- **Cross-sectional ranking and return prediction**
- **Reproducible experimental pipelines**
- **Quantitative evaluation with information-theoretic metrics**

From a research perspective, it shows that the project is not limited to training a model once. It includes:

- model architecture design,
- data pipeline assumptions,
- controlled hyperparameters,
- checkpoint loading and evaluation,
- and a clear path for future ablation studies and benchmarking.

---

## Core idea

The model predicts stock returns using a transformer-style architecture that combines:

- **feature gating** to reweight input factors,
- **positional encoding** to preserve sequence order,
- **temporal attention** for within-stock sequence modeling,
- **cross-sectional attention** for market-level interactions,
- and a final regression head for prediction.

This makes the project a strong example of **modern deep learning applied to financial forecasting**.

---

## Repository contents

```text
transformer_stock_prediction/
├── src/
│   ├── base_model.py
│   ├── main.py
│   └── model.py
└── .gitignore
```

---

## Model overview

The implementation is centered on the `MASTER` / `MASTERModel` architecture in `src/model.py`.

### Main components

- **Feature gate**
  - Learns how to weight factor inputs before sequence modeling.
- **Positional encoding**
  - Injects order information into the transformer pipeline.
- **Temporal attention**
  - Aggregates historical information along the time dimension.
- **Spatial / cross-sectional attention**
  - Models relationships across stocks in the batch.
- **Prediction head**
  - Produces the final forecast value.

### Research significance

This architecture is useful for demonstrating:

- attention-based representation learning,
- multi-stage feature transformation,
- market-aware sequence modeling,
- and the separation of prediction logic from evaluation logic.

That separation is exactly what reviewers often look for in strong research code.

---

## Data assumptions

The current entry script expects preprocessed pickle files generated in a Qlib-style workflow.

Typical data layout:

```text
data/
├── original/
├── opensource/
└── ...
```

In `src/main.py`, the script loads data for:

- `csi300`
- `csi800`

The current script also assumes:

- train / validation / test splits are already prepared,
- the feature dimension is fixed,
- and the input files are stored as pickle objects.

This is a good starting point for a research project because it makes the experiment pipeline explicit and easy to extend.

---

## Evaluation metrics

The project evaluates predictions using finance-relevant metrics, including:

- **IC** — Information Coefficient
- **ICIR** — Information Coefficient Information Ratio
- **RIC** — Rank IC
- **RICIR** — Rank IC Information Ratio

These metrics are much more appropriate for academic stock-prediction research than plain classification accuracy, because they reflect **ranking quality** and **predictive association**.

---

## Current script behavior

The `src/main.py` file currently contains both:

- a commented-out training loop, and
- an active checkpoint-loading / testing workflow.

In other words, the repository is already organized like a research experiment:

1. load prepared data,
2. initialize the model,
3. load a saved checkpoint,
4. run prediction,
5. report IC / ICIR / RIC / RICIR.

That structure is useful in a PhD setting because it supports:
- controlled experiments,
- repeatable evaluation,
- and clean comparison across model variants.

---

## How to run

### 1. Install dependencies

Install the packages required by the project environment.

```bash
pip install -r requirements.txt
```

### 2. Prepare Qlib-style data

Ensure the data directory contains the expected pickle files for your selected universe and split.

### 3. Run the entry script

```bash
python src/main.py
```

### 4. Review metrics

The script prints prediction metrics at the end of execution.

---

## Suggested research framing

When presenting this project in a PhD application, describe it as:

> A transformer-based financial forecasting system that combines feature gating, temporal attention, and cross-sectional market modeling to predict stock behavior using structured market data.

This wording helps position the project as:

- a **methodological contribution**,
- a **reproducible experimental baseline**,
- and a **candidate for deeper academic extension**.

---

## What makes this project academically useful

This repository can support a strong academic narrative because it touches several research themes at once:

### 1. Sequence modeling
The model uses transformer-style components for structured time-series learning.

### 2. Cross-sectional finance
It does not treat each stock independently; it considers market interactions.

### 3. Feature selection / gating
It incorporates learned feature reweighting rather than relying on raw factors only.

### 4. Evaluation rigor
It uses rank-based, finance-specific evaluation metrics.

### 5. Reproducibility
The code separates model definition, training logic, and evaluation.

---

## Possible extensions for future papers

To strengthen this repository further for PhD applications, consider adding:

- **Ablation studies**
  - remove feature gating
  - remove temporal attention
  - remove cross-sectional attention
  - compare against LSTM, GRU, MLP, and XGBoost baselines

- **Benchmarking**
  - compare across CSI300 and CSI800
  - report results across multiple random seeds
  - include statistical significance tests

- **Robustness analysis**
  - evaluate different market regimes
  - test sensitivity to sequence length and feature dimensionality
  - compare performance under missing or noisy inputs

- **Interpretability**
  - visualize attention weights
  - inspect factor importance from the gate module

- **Practical deployment**
  - add configuration files
  - package experiments with Docker
  - log results with TensorBoard / Weights & Biases
  - save experiment metadata for reproducibility

---

## A stronger PhD-oriented presentation

If you mention this project in a statement of purpose, CV, or email, the strongest angle is not “I built a stock prediction script.”

A stronger framing is:

- I designed a transformer-based forecasting pipeline for structured financial data.
- I implemented feature gating and attention mechanisms to model both temporal and cross-sectional dependencies.
- I evaluated the system using rank-based market metrics.
- I structured the project so it can be extended into ablation studies, robustness checks, and publishable experiments.

That is the kind of language that reads like research readiness.

---

## Notes

- This repository currently assumes preprocessed data is available.
- If you plan to release the project publicly, add:
  - a `requirements.txt`,
  - a `README` with dataset details,
  - a `license`,
  - and a short experiment table with reported metrics.

---

## License

Add a license file before public release.
