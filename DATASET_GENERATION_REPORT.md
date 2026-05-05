# Dataset Generation Report

## 1) What Was Implemented
A complete script (`synthetic_data_generator.py`) was implemented to generate a medically constrained synthetic emergency dataset with:

- Exact schema (43 columns)
- Exact class counts (30,000 total rows)
- Class-specific numeric distributions (truncated Gaussian)
- Class-4 bimodal glucose generation
- Class-specific binary Bernoulli generation
- Hard/soft dependency rules
- Contradiction rejection and regeneration logic
- Urgency score computation (capped at 100)
- Top-3 prediction/probability generation from ambiguity map
- Deterministic per-class priority tests
- Human-readable rotating trigger reasons

---

## 2) Class Count Summary (Target)
| Class | Label | Count |
|---|---|---:|
| 1 | Acute Cardiac Event | 5500 |
| 2 | Septic/Infectious Shock | 5500 |
| 3 | Neurological Emergency | 4500 |
| 4 | Metabolic Emergency | 4500 |
| 5 | Drug/Medication Induced Collapse | 5000 |
| 6 | Hemorrhagic/Internal Bleeding Shock | 5000 |
|  | **Total** | **30000** |

---

## 3) Printed Logs (Execution Attempts)
### Command
```bash
python synthetic_data_generator.py
```
### Output
```text
Traceback (most recent call last):
  File "/workspace/Emergency_Forecasting_System/synthetic_data_generator.py", line 1, in <module>
    import numpy as np
ModuleNotFoundError: No module named 'numpy'
```

### Dependency Install Attempt
```bash
python -m pip install numpy pandas -q
```

### Output
```text
ERROR: Could not find a version that satisfies the requirement numpy (from versions: none)
ERROR: No matching distribution found for numpy
```

---

## 4) Sample Rows (5–10) and First 10 Rows
Because runtime dependencies are unavailable in the current environment (no `numpy`/`pandas` install possible due network/proxy restrictions), the CSV could not be generated here.

Once dependencies are available, run:

```bash
python synthetic_data_generator.py
python - <<'PY'
import pandas as pd

df = pd.read_csv('emergency_forecasting_dataset.csv')
print('Class distribution:')
print(df['primary_emergency_class'].value_counts())
print('\n5-10 sample rows:')
print(df.sample(10, random_state=42))
print('\nFirst 10 rows:')
print(df.head(10))
PY
```

---

## 5) Class Distribution (Expected from Generator)
The generator is configured to emit exactly:

- Acute Cardiac Event: 5500
- Septic/Infectious Shock: 5500
- Neurological Emergency: 4500
- Metabolic Emergency: 4500
- Drug/Medication Induced Collapse: 5000
- Hemorrhagic/Internal Bleeding Shock: 5000
