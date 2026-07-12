# ADMET Genetic Optimization Example Report

## Run overview

This report was reconstructed from committed run artifacts. The build script does not rerun the genetic algorithm or ADMET-AI.

- Input seeds: 12
- Evaluated generation-0 seeds: 12
- Invalid and deduplicated input counts: not captured
- Recorded optimization generations: 4
- Generated molecules per generation: 24
- Total generation-log records: 108
- Mutation records: 78
- Crossover records: 18
- Records passing filters: 83
- Final candidates: 4
- Random seed: not captured
- Explicit stop reason: not captured; the artifacts end after generation 4
- Exact dependency versions: not captured
- Final candidate selection: generated molecules that pass a fresh config-based filter check, have no ADMET failure, and score strictly above their best ancestral seed; retain the highest-scoring molecule for each distinct ancestral-seed lineage

## Configuration

### Hard filters

| Setting | Value |
|---|---:|
| `mw_min` | 120 |
| `mw_max` | 500 |
| `logp_min` | -0.5 |
| `logp_max` | 5.0 |
| `tpsa_min` | 20 |
| `tpsa_max` | 140 |
| `hbd_max` | 5 |
| `hba_max` | 10 |
| `rotb_max` | 10 |
| `sa_score_max` | 5.0 |
| `qed_min` | 0.3 |
| `risk_flags_max` | 6 |

### Scoring

| Setting | Value |
|---|---:|
| `qed_weight` | 0.25 |
| `admet_weight` | 0.45 |
| `sa_weight` | 0.25 |
| `property_weight` | 0.05 |
| `sa_transform` | clipped (10 - sa_score) / 9 |
| `property_window_terms` | 6 |

### ADMET mapping

- ADMET-AI requested: `True`
- Risk threshold: `0.5`
- Positive endpoint keywords: `hia`, `caco`, `bioavailability`, `solubility`
- Negative endpoint keywords: `herg`, `ames`, `dili`, `cyp`, `pgp`, `p-gp`, `tox`

## Generation summary

| Generation | Generated | Best score | Mean score | Passed | Population best |
|---:|---:|---:|---:|---:|---:|
| 1 | 24 | 0.8803963 | 0.8216755 | 15 | 0.8822355 |
| 2 | 24 | 0.8819185 | 0.8466638 | 20 | 0.8822355 |
| 3 | 24 | 0.8767579 | 0.8277978 | 19 | 0.8822355 |
| 4 | 24 | 0.8841338 | 0.8495277 | 21 | 0.8841338 |

## Final candidates

| Molecule ID | Operation | Baseline seed(s) | QED | SA score | ADMET score | Total score | Delta total | Risks |
|---|---|---|---:|---:|---:|---:|---:|---|
| `GA_g4_0085` | mutation | N-Propylbenzamide | 0.7584 | 1.3220 | 0.8966 | 0.8841 | +0.0019 | high_CYP1A2_Veith;high_DILI;high_Skin_Reaction |
| `GA_g2_0033` | mutation | Ibuprofen | 0.7675 | 1.4783 | 0.8963 | 0.8819 | +0.0025 | high_DILI |
| `GA_g1_0006` | mutation | Quinoline | 0.5413 | 1.6061 | 0.9086 | 0.8274 | +0.0152 | high_CYP1A2_Veith;high_DILI |
| `GA_g1_0018` | mutation | Acetaminophen | 0.6236 | 2.2939 | 0.9040 | 0.8267 | +0.0116 | high_DILI;low_PAMPA_NCATS |

## Interpretation

The best generation-0 score was 0.8822355 (`N-Propylbenzamide`). The best generated score was 0.8841338 (`GA_g4_0085`), a recorded increase of +0.0018983.

All 4 final candidates passed the configured filters, had successful ADMET evaluation, and improved total score relative to the best seed in their recorded ancestry. Filters were recalculated from the recorded properties and `config.yaml` rather than accepted solely from the log flag. The highest-scoring molecule represents each distinct ancestral-seed lineage. This deterministic rule does not claim an unrecorded fingerprint-diversity calculation.

Final-candidate risk flags were `high_CYP1A2_Veith` (2), `high_DILI` (4), `high_Skin_Reaction` (1), `low_PAMPA_NCATS` (1). These flags are model-derived triage signals, not observed toxicology outcomes.

## Limitations and next steps

- All ADMET values are model predictions and have not been experimentally validated.
- The low-level mutation and crossover operators do not establish chemical feasibility.
- SA score is a heuristic and does not guarantee a practical synthesis route.
- The endpoint aggregation and keyword mapping are task-specific heuristics.
- Reproduce the run with captured package versions and a fixed random seed.
- Review top structures for medicinal-chemistry liabilities and scaffold diversity.
- Validate prioritized endpoints with independent models and experimental assays.
- Assess synthetic routes before advancing a candidate.
