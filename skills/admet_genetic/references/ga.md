# GA Design Notes

This reference gives a starter design, not fixed parameter advice. Choose population size, offspring count, generation budget, elitism, mutation/crossover rates, filters, and scoring weights from the user’s molecule count, compute budget, endpoint priorities, and tolerance for exploration.

## Starter GA structure

A practical starter structure:

1. Evaluate standardized seed molecules.
2. Keep an elite subset from the current population.
3. Sample parents from high-scoring or diverse candidates.
4. Generate children with molecular mutation and crossover.
5. Sanitize and canonicalize every child.
6. Reject invalid and duplicate canonical SMILES before assigning a child ID.
7. Evaluate valid children.
8. Merge elites and children.
9. Select the next population by hard filters, total score, and diversity.
10. Stop on generation budget, score plateau, compute budget, or enough acceptable candidates.

Scale this structure up or down. Very small problems may need only a few generations and offspring; broad exploration may need larger populations, more mutation templates, or multiple random seeds.

## Mutation operators

Prefer stable local operations first, then add domain-specific operators when the user’s chemistry requires them:

- atom replacement among C/N/O when valence can sanitize;
- halogen replacement such as Cl/Br to F;
- small substituent changes such as methyl addition/removal;
- terminal atom pruning to reduce size or rotatable bonds;
- domain-specific functional-group replacements when the user gives chemistry constraints.

Every mutation must be sanitized and canonicalized. Sanitization failure is normal; count it as a failed operation rather than repairing it silently.

## Crossover operators

Use crossover only when it fits the problem. BRICS crossover is a reasonable starter approach:

- select two distinct parent IDs;
- BRICS-decompose both parents;
- sample fragments from the combined pool;
- attempt BRICSBuild;
- accept only sanitized children that are not equal to either parent and not previously seen.

Record parent fragment counts and selected fragment SMILES in `operation_detail`.

## Filters

Filters are problem-dependent. If the user does not specify them, start from conservative drug-like windows and state them explicitly in the config and report:

- Molecular weight window.
- LogP window.
- TPSA window.
- HBD/HBA caps.
- Rotatable bond cap.
- SA-Score cap.
- QED floor.
- ADMET failure exclusion.
- Endpoint-specific ADMET exclusions or penalties.

Do not treat these windows as universal medicinal chemistry truth. Adjust them for peptides, fragments, CNS programs, macrocycles, covalent warheads, probes, or non-oral programs.

### RDKit SAScorer Usage

```python
from rdkit import Chem
from rdkit.Contrib.SA_Score import sascorer

mol = Chem.MolFromSmiles("CCO")
sa = sascorer.calculateScore(mol)
print(sa)  # 1.9802570386349831
```

## Scoring

The total score should reflect the user’s goal. A starter score can combine:

- QED or target-likeness;
- ADMET aggregate score;
- transformed SA-Score;
- property-window score;
- diversity bonus from Morgan fingerprint distance;
- risk penalties from ADMET flags.

Keep raw endpoint predictions separate from derived scores. Report the scoring formula and all weights. If the user provides a target profile, replace generic weights with target-specific terms.

## Diversity selection

Use diversity selection after score sorting to prevent near-duplicate top lists. Morgan fingerprints with Tanimoto similarity are a reasonable starter. Tighten or loosen the similarity threshold based on whether the user wants analog clustering or broad exploration.
