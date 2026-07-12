# Data Contracts and Lineage

## Molecule identity

Every logged molecule must satisfy this invariant:

```text
molecule_id <-> canonical smiles
```

Operationally, this means:

- `molecule_id` is unique in `generation_log.csv`.
- `smiles` is unique in `generation_log.csv`.
- `smiles` is canonicalized after RDKit sanitize.
- duplicate canonical SMILES are rejected before assigning a new child ID.

Do not reuse the same canonical SMILES across generations under different IDs. If a molecule reappears, record the operation as a duplicate failure rather than adding another candidate record.

## Required columns for outputs

For any output file, minimum required fields:

```text
molecule_id
generation
smiles
parent
parents
parent_smiles
operation
operation_detail
status
qed
sa_score
admet_score
admet_risk_flags
admet_failed
total_score
passes_filters
MW
LogP
TPSA
HBD
HBA
RotatableBonds
RingCount
```

- `molecule_id` identifies exactly one canonical `smiles`.
- `smiles` is always the deduplicated canonical SMILES.
- For `operation == mutation`, set `parent` to one parent ID and leave `parents` empty.
- For `operation == crossover`, leave `parent` empty and set `parents` to exactly two parent IDs separated by `;`.
- `operation_detail` should be JSON containing operation name, operator detail, parent IDs, parent SMILES, and child canonical SMILES.

Additional raw columns are allowed. Store raw ADMET predictions in `admet_predictions_json` when preserving endpoint-level outputs.

## Parent fields

Use these rules exactly:

- `seed`: `parent == ""`, `parents == ""`.
- `mutation`: `parent == "<single parent molecule_id>"`, `parents == ""`.
- `crossover`: `parent == ""`, `parents == "<parent_a_id>;<parent_b_id>"`.

`parent_smiles` is informational. Use IDs, not SMILES, for all programmatic lineage traversal.

## operation_detail JSON

`operation_detail` should be valid JSON. Recommended shape:

```json
{
  "operation": "mutation",
  "operator_detail": "add_methyl_to_idx_1",
  "parent_ids": ["seed_001"],
  "parent_smiles": ["CCOC(=O)c1ccccc1"],
  "child_canonical_smiles": "CC(C)OC(=O)c1ccccc1"
}
```

For crossover, `operator_detail` may contain:

```json
{
  "operator": "brics_crossover",
  "parent_fragment_counts": [4, 6],
  "selected_fragments": ["[16*]c1ccccc1", "[3*]O[3*]"]
}
```

## Visualization expectations

The optimization-history visualization:

- reads only `generation_log.csv`;
- validates ID/SMILES uniqueness;
- uses `molecule_id`, `parent`, and `parents` for lineage;
- renders mutation as one-parent ancestry;
- renders crossover as two-parent ancestry;
- embeds RDKit SVG depictions in the HTML;
- plots score and throughput trajectories from logged records.
