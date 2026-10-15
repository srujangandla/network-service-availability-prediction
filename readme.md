# Network Service Availability Prediction

**ML framework for anomaly classification and service-risk estimation on telecom KPI time series**

Telecom operators log high-frequency KPI telemetry (RSRP, BLER, MCS, PRB usage, traffic counters) to keep services within their SLAs. This project classifies whether a telemetry trace contains an anomaly, using statistical features extracted from multi-dimensional KPI time series. It is built to work under severe class imbalance (about 3.86% anomalous traces) while keeping the false alarm rate close to zero.

> **Scope note:** This is **retrospective anomaly classification**. The model labels a trace using telemetry from the same window in which the anomaly occurs. It does **not** forecast outages in advance. See [Limitations](#limitations).

---

## Table of Contents

- [Highlights](#highlights)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Features](#features)
- [Models](#models)
- [Results](#results)
- [Most Influential Features](#most-influential-features)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Authors](#authors)
- [Acknowledgements](#acknowledgements)

---

## Highlights

- 32,000 telemetry traces, 18 KPIs per trace, 128 samples each (10 Hz, about 12.7 s)
- 110 engineered features: 106 numeric and 4 categorical
- Leakage safeguards: free-text, anomaly descriptions, QnA pairs and the labels dictionary (apart from the 4 context fields) are excluded
- Class-weighted Random Forest reaches **F1 = 0.9577, precision = 1.0000, FAR = 0.0000** on the held-out test set
- A majority-class baseline is included to show the accuracy paradox (96.15% accuracy, 0 recall)
- Top predictors are physical-layer signals: **RSRP, UL_SNR, UL_BLER**

---

## Dataset

`telecom_ts_dataset` contains 32,000 records. Each record is one trace window.

| Field | Type | Description |
|---|---|---|
| `start_time`, `end_time` | string (timestamp) | Trace collection window |
| `sampling_rate` | int | Fixed at 10 Hz |
| `KPIs` | string (encoded dict) | 18 parallel KPI series, 128 floats each |
| `description` | string | Natural-language summary (**excluded**, leakage) |
| `anomalies` | string (encoded dict) | Fault diagnosis and ticket text (**excluded**, leakage) |
| `statistics` | string (encoded dict) | Precomputed aggregates per KPI |
| `labels` | string (encoded dict) | zone, application, mobility, congestion, `anomaly_present` |
| `QnA` | string (encoded dict) | Synthetic Q&A pairs (**excluded**, leakage) |

**Target:** `labels['anomaly_present']` (Yes/No). The class split is 30,765 normal (96.14%) and 1,235 anomalous (3.86%).

**KPIs (18):** RSRP, DL_BLER, UL_BLER, DL_MCS, UL_MCS, UL_NPRB, UL_SNR, TX_Bytes, RX_Bytes, Estimated_UL_Buffer, PRBs_DL_Current, PRBs_UL_Current, PRB_Utilization_DL, PRB_Utilization_UL, UL_NumberOfPackets, DL_NumberOfPackets, UL_Protocol, DL_Protocol.

> The dataset file is **not included** in this repository. Place `telecom_ts_dataset.csv` (or `.parquet`) where the notebook expects it (see [Getting Started](#getting-started)).

**Data quality findings:**
- 4,419 records have future-dated start times (as far as 2037), with multi-year gaps between traces. The timestamps are non-chronological and synthetic, so a **stratified split** is used instead of a chronological one.
- No outage or availability label exists. The task is therefore anomaly classification, not outage prediction.

---

## Pipeline

```
Raw records (32,000)
  → Parse nested string fields (safe literal eval + regex clean-up)
  → Validate shapes & audit timestamps
  → Extract window statistics per KPI
  → Filter constant & highly correlated features
  → Remove leakage sources
  → Stratified 70/15/15 split
  → Train-only pipeline (impute → scale → one-hot → class-weighted model)
```

All preprocessing (median imputation, standardisation, one-hot encoding) lives inside a single scikit-learn `Pipeline` fitted **only on the training partition**, so nothing leaks from validation or test data.

---

## Features

| Group | Count | Description |
|---|---|---|
| KPI-derived statistics | 90 | mean, std, min, max, median for each of the 18 KPIs |
| Supplied statistics | 44 | Flattened from the `statistics` field. 20 redundant ones with \|r\| > 0.95 against KPI features were dropped |
| Context (categorical) | 4 | zone, application, mobility, congestion |

Constant columns and near-duplicates (\|r\| > 0.98) are pruned, leaving **110 features** (106 numeric, 4 categorical).

---

## Models

| Model | Notes |
|---|---|
| Dummy classifier | Majority-class baseline |
| Logistic Regression | L2, `class_weight="balanced"`, standardised inputs, `max_iter=1000` |
| Random Forest | 200 trees, `min_samples_leaf=2`, `class_weight="balanced_subsample"` |

The best model is selected by **validation F1** (not accuracy), and the test set is used only for the final report. `RANDOM_STATE = 42`.

---

## Results

### Data split (stratified 70/15/15)

| Partition | Records | Anomalous | Prevalence |
|---|---|---|---|
| Train | 22,400 | 865 | 3.86% |
| Validation | 4,800 | 185 | 3.85% |
| Test | 4,800 | 185 | 3.85% |

### Test-set performance

| Model | Acc. | Prec. | Recall | F1 | FAR | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|
| Dummy (baseline) | 0.9615 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 4,615 | 185 |
| Logistic Regression | 0.9694 | 0.5605 | **0.9514** | 0.7054 | 0.0299 | 176 | 138 | 4,477 | 9 |
| **Random Forest** | **0.9969** | **1.0000** | 0.9189 | **0.9577** | **0.0000** | 170 | 0 | 4,615 | 15 |

- **Random Forest:** zero false alarms and the best F1. Choose it when analyst time is the scarce resource.
- **Logistic Regression:** slightly higher recall, but about 3% FAR would flood an OSS on a large network. Its Wilson 95% recall interval overlaps the forest's, so the recall gap should not be over-interpreted.
- **Uncertainty:** with 185 test anomalies, the RF recall of 0.9189 has a 95% Wilson interval of roughly 0.871–0.950. Zero false positives in 4,615 normals gives a one-sided 95% FAR upper bound of about 0.065%.

### Ablation (validation set, Random Forest)

| Feature set | Count | F1 | FAR |
|---|---|---|---|
| Full model | 110 | 0.9665 | 0.0000 |
| KPI statistics only | 90 | 0.9579 | 0.0002 |
| Supplied statistics only | 44 | 0.8804 | 0.0032 |
| Context features only | 4 | 0.0000 | 0.0000 |

Context labels alone carry no predictive signal. The KPI-derived statistics account for more than 98% of the full model's F1.

---

## Most Influential Features

Random Forest mean-decrease-in-impurity (MDI), top 10:

| Rank | Feature | MDI |
|---|---|---|
| 1 | `RSRP_mean` | 0.0568 |
| 2 | `stat_RSRP_variance` | 0.0554 |
| 3 | `RSRP_min` | 0.0518 |
| 4 | `UL_SNR_mean` | 0.0517 |
| 5 | `RSRP_max` | 0.0493 |
| 6 | `RSRP_std` | 0.0472 |
| 7 | `UL_BLER_mean` | 0.0369 |
| 8 | `RX_Bytes_median` | 0.0337 |
| 9 | `UL_BLER_max` | 0.0310 |
| 10 | `RX_Bytes_min` | 0.0308 |

A drop in RSRP or UL_SNR together with rising error rates matches what hardware faults, cell-edge degradation and interference would produce. MDI indicates association, not causation. Permutation importance or SHAP would be a useful complement.

---

## Repository Structure

```
.
├── Network_Service_Availability_Prediction.ipynb   # Full pipeline: EDA → features → models → evaluation
├── Network_Service_Availability_paper.pdf          # Accompanying research paper
├── README.md
└── data/
    └── telecom_ts_dataset.csv                      # Not included; add your own copy
```

The notebook is organised into these sections:

1. Imports and data loading
2. Data understanding and parsing of the nested string fields
3. Data cleaning (duplicates, timestamps, trace-length checks)
4. Exploratory data analysis
5. Feature engineering (KPI statistics, supplied statistics, categorical context)
6. Target preparation and leakage guard
7. Splitting and model training
8. Evaluation (confusion matrices, precision-recall curves)
9. Model comparison and feature importance
10. Example predictions and final report

---

## Getting Started

### Requirements

- Python 3.10+
- `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Run on Google Colab (as written)

1. Upload `telecom_ts_dataset.csv` to your Google Drive.
2. Open the notebook in Colab.
3. Edit the data path in the loading cell:
```python
   DATA_PATH = "/content/drive/MyDrive/telecom_ts_dataset.csv"
```
4. Run all cells.

### Run locally

The notebook mounts Google Drive via `google.colab`. To run it locally, remove these lines:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Then point `DATA_PATH` at your local file (`.csv` and `.parquet` are supported):

```python
DATA_PATH = "data/telecom_ts_dataset.csv"
```

### Predict a single held-out record

```python
predict_record(test_row_index)
# {'anomaly_probability': 0.97, 'predicted': 'Anomaly', 'actual': 'Anomaly'}
```

---

## Limitations

1. **Retrospective, not prospective.** Features come from the anomaly window itself, so the model does not forecast outages.
2. **Synthetic timestamp drift.** Future-dated and non-chronological timestamps rule out chronological validation and true time-series forecasting.
3. **No outage-severity ground truth.** The target is anomaly presence, not SLA breach or service unavailability.
4. **Fixed 0.5 decision threshold.** No cost-based threshold tuning was done.
5. **Single stratified split.** No repeated resampling or cross-validation, so variance is only partly captured.

---

## Future Work

- Collect chronologically valid telemetry linked to confirmed outage tickets in the OSS
- Use sliding-window sequence models (LSTM, TCN) to predict failures Δt minutes ahead
- Tune the decision threshold with cost-sensitive optimisation
- Add permutation and SHAP explanations plus repeated cross-validation

---

## Authors

- Charagundla Hemanth Sai
- Gandla Srujan
- Cheemala Chaitanya Kumar Reddy

Department of Artificial Intelligence and Data Science, Chaitanya Bharathi Institute of Technology, Hyderabad, Telangana, India

---

## Acknowledgements

Thanks to the open-source communities behind Python, scikit-learn, pandas and NumPy.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

The `telecom_ts_dataset` is not covered by this license and is subject to its own terms.
