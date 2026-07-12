---
name: admet_genetic
description: ADMET-guided genetic molecule optimization workflow from seed SMILES; use when the agent needs to build or run an RDKit/SA-Score/ADMET-AI GA pipeline for molecule optimization, enforce molecule lineage logs, render optimization-history HTML dashboards, and write candidate triage reports.
origin: openai4s
category: chemistry
metadata:
  display-name: ADMET-driven genetic molecule optimization
  # SKILL.md body: "**License:** MIT (github.com/swansonk14/admet_ai)."
  # github.com/swansonk14/admet_ai/blob/main/LICENSE.txt:
  # MIT (© Kyle Swanson et al.). Verified 2026-07-10.
  third_party:
    - kind: weights
      name: ADMET-AI
      provider: Greenstone Biosciences
      license: MIT
      terms_url: https://github.com/swansonk14/admet_ai/blob/main/LICENSE.txt
---

# ADMET Genetic Optimization

Use this skill to build and run a molecular optimization loop from seed SMILES. The target artifact is a ranked set of optimized candidate molecules with auditable lineage, scores, and report artifacts.

The sidecar deliberately does not provide a fixed GA engine. The agent must
assemble and tune mutation, crossover, evaluation, filtering, and selection for
the user's objective. `kernel.py` provides reusable molecule normalization,
ADMET aggregation, lineage validation, and result visualization.

## Special Reminder
In this skill, when see `references/<file_name>.md` is suggested, use host call to retrieval the complementary material.
```python
host.skills.read("admet_genetic", "references/<file_name>.md")
```

## Prerequisites

```bash
conda create -n admet-sa-ga python=3.11 -y
conda activate admet-sa-ga
python -m pip install pandas pyyaml matplotlib rdkit
python -m pip install admet-ai  # depends on torch; installation/import may take time
```

After creating the environment, select it with `host.env.use("admet-sa-ga")`
before importing this skill's sidecar. Switching environments restarts the
session kernel, so switch before constructing the in-memory pipeline.

See `references/admet.md` for ADMET-AI installation details, endpoint behavior, runtime notes, and troubleshooting.

## Data Contracts
For molecular representation, expected fields, candidate recording and lineage logging, see `references/data_contracts.md`. **Must view these contracts before running the main pipeline.**

## Core Workflow

1. Collect user-provided seed molecules or uploaded files and normalize them into a CSV input. The CSV should contain `smiles`; include `molecule_id` when stable user-facing IDs are available, otherwise synthesize deterministic IDs.
2. Standardize each input using `standardize_smiles(...)`, then use `canonicalize_smiles(...)` from `kernel.py` where a strict canonical string is needed. Molecule ID and canonical SMILES must be one-to-one for all logged records.
3. Design a genetic algorithm that includes molecular mutation and crossover. Match population size, generation count, operators, filters, and scoring weights to the user’s problem scale and constraints. For a starter design and implementation choices, see `references/ga.md`.
4. Evaluate each valid molecule with RDKit descriptors, QED, SA-Score, and ADMET predictions. Aggregate ADMET endpoints into `admet_score` and `admet_risk_flags`; preserve raw endpoint outputs. See `references/data_contracts.md` for required evaluation fields.
5. Apply hard filters, compute total score, select diverse candidates by Morgan fingerprint similarity, and update the population. See `references/ga.md` for starter designs.
6. Assess whether the final candidates improve on the seeds and satisfy the user’s requirements. If they do not, adjust GA parameters, mutation/crossover operators, filters, or scoring weights, then rerun the internal GA workflow before finalizing output.
7. Output final candidates, logs, report, visualization dashboard, and any other produced artifacts. See **Artifacts** for log schema and lineage rules.

## Import

```python
from admet_genetic.kernel import (
    aggregate_admet_predictions,
    canonicalize_smiles,
    classify_admet_columns,
    operation_detail_json,
    standardize_smiles,
    validate_generation_log,
    render_optimization_history,
)
```

## Artifacts

### Lineage Requirements

Treat lineage as a first-class data contract:

- `molecule_id` identifies exactly one canonical `smiles`.
- `smiles` is always the deduplicated canonical SMILES.
- For `operation == mutation`, set `parent` to one parent ID and leave `parents` empty.
- For `operation == crossover`, leave `parent` empty and set `parents` to exactly two parent IDs separated by `;`.
- `operation_detail` should be JSON containing operation name, operator detail, parent IDs, parent SMILES, and child canonical SMILES.

### Record Schema

Before rendering or reporting, run `validate_generation_log(frame)` or equivalent assertions; see `references/data_contracts.md`.

### Required Artifacts

When results are satisfactory, produce:

- `generation_log.csv` with complete lineage and evaluation records.
- `candidates_final.csv` with selected final candidates.
- `report.md` as an audit-friendly report.
- molecule SVGs or embedded drawings when helpful.
- an optimization-history HTML dashboard via `render_optimization_history(log_path, out_path)` from `kernel.py`. The rendered HTML is self-contained and uses embedded SVG molecule depictions and matplotlib-generated SVG plots.
- Other visualized artifacts suggested by system prompt or user requirements.

The visualization workflow expects `generation_log.csv` to follow the lineage contract; for visualization assumptions, see `references/data_contracts.md`.

In `report.md`, include:

- Run goal, input file, seed count, valid seed count, and deduplication/invalid counts.
- Dependency versions, especially RDKit, ADMET-AI, pandas, numpy, and Python.
- GA parameters and stop reason.
- Standardization policy and failure reason summary.
- Scoring formula, hard filters, diversity threshold, and ADMET endpoint mapping.
- Per-generation summary: count, generated count, best score, mean score, pass count.
- Top candidate table with ID, canonical SMILES, parent lineage, operation, QED, SA-Score, ADMET score, risk flags, total score, and pass/fail.
- A short interpretation of what improved, which risks dominate, and whether top hits mainly arise from mutation or crossover.
- Limitations: low-level operators, heuristic ADMET aggregation, model uncertainty, no experimental validation, no synthetic feasibility guarantee beyond SA-Score.
- Next steps: better mutation templates, medicinal chemistry constraints, external validation, improved diversity, and route feasibility checks.

State clearly when ADMET-AI failed or when a fallback was used. Do not present predicted ADMET, toxicity, conditions, or synthesizability as experimental fact.

## Reproducible Example

The committed example under `examples/` is a recorded four-generation test run.
It is an audit and visualization fixture, not evidence of experimental ADMET or
synthetic feasibility:

```text
examples/
|-- seed_molecules.csv
|-- config.yaml
|-- generation_log.csv
|-- generation_summary.csv
|-- candidates_final.csv
|-- optimization_dashboard.html
|-- report.md
`-- build_example.py
```

`generation_log.csv`, `generation_summary.csv`, and `config.yaml` are the source
records used to rebuild the derived dashboard and report. The build does not run
the GA or ADMET-AI:

```bash
python skills/admet_genetic/examples/build_example.py
```

Use alternate output paths when checking reproducibility without replacing the
committed artifacts:

```bash
python skills/admet_genetic/examples/build_example.py \
  --dashboard-output /tmp/admet-dashboard.html \
  --report-output /tmp/admet-report.md
```

The example intentionally reports run metadata that was not captured rather
than inferring it. Exact dependency versions, the random seed, the explicit stop
reason, and invalid-input counts are unknown for this recorded run. Final
candidates are derived reproducibly from `generation_log.csv`: retain generated
molecules that pass a fresh hard-filter check against `config.yaml`, have no
ADMET failure, and strictly improve total score over the best seed in their
recorded ancestry. For each distinct ancestral-seed lineage, retain only its
highest-scoring qualifying molecule.

## Dashboard QA

Before accepting a generated result, manually open the HTML dashboard and:

- move the generation slider through every recorded generation;
- select seed, mutation, and crossover records;
- confirm one-parent and two-parent lineage trees match `generation_log.csv`;
- confirm scores, filter status, structures, and plots agree with the CSV files;
- check desktop and mobile widths for overflow or clipped labels;
- confirm the browser console has no errors.
