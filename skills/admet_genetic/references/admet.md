# ADMET-AI Usage and Endpoint Aggregation

ADMET-AI is a simple, fast, and accurate web interface for predicting the ADMET properties of molecules using machine learning models.

ADMET-AI predicts ADMET properties using a graph neural network architecture called Chemprop-RDKit (see the Chemprop package for details). ADMET-AI's Chemprop-RDKit models were trained on 41 ADMET datasets from the Therapeutics Data Commons (TDC). ADMET-AI’s Chemprop-RDKit models have the highest average rank on the TDC ADMET Benchmark Group leaderboard. ADMET-AI is also currently the fastest web-based ADMET predictor.

This file summarizes the CLI and Python-module usage needed by the ADMET genetic optimization workflow.

## Installation

`admet-ai` depends on torch, so installation and first import may take longer than ordinary Python packages. Wait for model initialization before deciding that the process is stuck.

```bash
pip install admet-ai
```

## Running it

Command line:

```bash
admet_predict \
    --data_path data.csv \
    --save_path preds.csv \
    --smiles_column smiles
```

Python module:

```python
from admet_ai import ADMETModel

model = ADMETModel()
preds = model.predict(smiles="O(c1ccc(cc1)CCOC)CC(O)CNC(C)C")
```

If a single SMILES string is provided, `preds` is a dictionary mapping property names to values. If a list of SMILES strings is provided, `preds` is a pandas DataFrame indexed by SMILES with endpoint columns.

Inference can use a GPU when available, but CPU-only execution is supported. If GPU inference failed somehow, try `CUDA_DEVICES=""`

## Suggested aggregation policy

Keep raw ADMET-AI outputs unchanged. Add derived fields instead:

```text
admet_score
admet_risk_flags
admet_failed
admet_predictions_json
```

Column-name mapping:

```yaml
admet:
  risk_threshold: 0.5
  positive_keywords:
    - hia
    - caco
    - bioavailability
    - solubility
  negative_keywords:
    - herg
    - ames
    - dili
    - cyp
    - pgp
    - p-gp
    - tox
```

- Positive endpoints: HIA, Caco2, Bioavailability, Solubility.
- Negative endpoints: hERG, AMES, DILI, CYP, Pgp/P-gp, ClinTox or other toxicity-like names.
- Ignore `*_drugbank_approved_percentile` columns for risk scoring. These are reference distribution percentiles, not raw endpoint predictions.
- Unknown endpoints should remain in raw predictions and be reported as ignored.

For numeric outputs in the 0-1 range, interpret values directly. For negative endpoints, high values indicate higher risk. For positive endpoints, low values indicate weakness or risk. For values outside 0-1, use a simple squashing transform only for demo scoring, and state this in the report.

## Runtime behavior

ADMET-AI can print torch/lightning messages such as GPU availability, NVML warnings, progress bars, and dataloader worker hints. These are usually not fatal. Treat actual exceptions from `ADMETModel().predict(...)` as ADMET failure and either stop or use a clearly labeled demo fallback.

Use batch prediction where possible. Cache predictions by canonical SMILES so duplicate evaluation attempts do not reload or recompute endpoints.

## Troubleshooting

| You see | It means / do this |
|---|---|
| ImportError: libXrender.so.1: cannot open shared object file: No such file or directory |  conda install -c conda-forge xorg-libxrender |
| Matplotlib cache warnings | Set `MPLCONFIGDIR` and `XDG_CACHE_HOME` to writable temporary directories. |
| `Can't initialize NVML` | Usually harmless on CPU-only environments. |
| Long import or first prediction | Wait; model and torch initialization can be slow. |
| RuntimeError: The NVIDIA driver on your system is too old (found version xxx). Please update your GPU driver by downloading and installing a new version | setting environment variable CUDA_VISIBLE_DEVICES="" |
