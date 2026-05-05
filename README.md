# Transformer-Based Cross-Sectional Stock Prediction

## Overview

This repository implements a transformer-based architecture for cross-sectional stock return prediction. The model is designed to capture both temporal dependencies within individual assets and cross-sectional interactions across assets at each time step.

The implementation follows a modular structure, separating data handling, model definition, and evaluation, with an emphasis on clarity and reproducibility.

---

## Model Architecture

The core model (MASTER) integrates several components:

- **Feature Gating**  
  A learnable mechanism to reweight input factors before sequence modeling.

- **Positional Encoding**  
  Injects temporal order information into the sequence representation.

- **Temporal Attention**  
  Captures dependencies across historical observations for each asset.

- **Cross-Sectional Attention**  
  Models interactions between assets within the same time slice.

- **Prediction Head**  
  Maps learned representations to scalar return forecasts.

This design enables joint modeling of time-series dynamics and market-wide structure.

---

## Data Interface

The code assumes preprocessed data in a Qlib-style format. Input data are loaded from serialized files (e.g., pickle), with predefined train/validation/test splits.

Typical characteristics:

- Fixed feature dimension  
- Aligned time index across assets  
- Batched cross-sectional samples  

---

## Evaluation

Performance is evaluated using rank-based and correlation-based metrics commonly used in quantitative finance:

- **IC (Information Coefficient)**  
- **ICIR (Information Coefficient Information Ratio)**  
- **Rank IC (RIC)**  
- **Rank ICIR (RICIR)**  

These metrics assess the association between predicted scores and realized returns, focusing on ranking quality rather than absolute error.

---

## Repository Structure

```
transformer_stock_prediction/
├── src/
│   ├── base_model.py
│   ├── model.py
│   └── main.py
└── .gitignore
```

- `model.py`: model architecture definition  
- `base_model.py`: shared components/utilities  
- `main.py`: experiment entry point (loading, inference, evaluation)  

---

## Usage

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run evaluation

```bash
python src/main.py
```

The script loads a trained checkpoint (if provided) and reports evaluation metrics on the test set.

---

## Experimental Setup

The current pipeline supports:

1. Loading preprocessed datasets  
2. Initializing the model with fixed hyperparameters  
3. Loading checkpoints  
4. Running inference  
5. Computing evaluation metrics  

This structure facilitates controlled comparison across model variants.

---

## Extensions

Potential directions for further development include:

- Systematic ablation of architectural components  
- Comparison with alternative baselines (e.g., LSTM, MLP, tree-based models)  
- Sensitivity analysis with respect to sequence length and feature sets  
- Regime-based evaluation across different market conditions  
- Visualization of attention patterns for interpretability  

---

## Notes

- The repository does not include raw data or preprocessing scripts.  
- Users are expected to prepare input data separately.  
- A `requirements.txt` and experiment configuration files are recommended for full reproducibility.

---

## License

To be specified.
