# Emergency Forecasting System

## Overview
This repository contains a medically constrained synthetic dataset generator:

- `synthetic_data_generator.py`

The script generates a final CSV file named:

- `emergency_forecasting_dataset.csv`

with exactly **30,000 rows** and **43 columns** (37 input features + 6 output columns), aligned to the Emergency Forecasting System requirements.

---

## Files
- `synthetic_data_generator.py`: full synthetic data generation script.
- `DATASET_GENERATION_REPORT.md`: structured run/report document including class distribution targets, sample output section, and logs.

---

## How to Run
```bash
python synthetic_data_generator.py
```

On successful execution, the script will:
1. Save `emergency_forecasting_dataset.csv`
2. Print class counts
3. Print null counts
4. Print sample rows

---

## Required Python Packages
- `numpy`
- `pandas`

Install with:
```bash
python -m pip install numpy pandas
```

---

## Expected Class Distribution
- Class 1 Acute Cardiac Event: 5500
- Class 2 Septic/Infectious Shock: 5500
- Class 3 Neurological Emergency: 4500
- Class 4 Metabolic Emergency: 4500
- Class 5 Drug/Medication Induced Collapse: 5000
- Class 6 Hemorrhagic/Internal Bleeding Shock: 5000

Total: 30,000
