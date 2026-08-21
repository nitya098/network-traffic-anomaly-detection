# Network Traffic Anomaly Detection using NSL-KDD

Five-class classification of NSL-KDD connection records for network intrusion detection. The locked model is a **Random Forest (500 trees)** selected on a **KDDTrain+ validation** split. **KDDTest+ was not used for model selection.**

The packaged artifact is a scikit-learn `Pipeline` (preprocessing + classifier), intended for later backend integration. A full-stack application is **not** implemented in this repository.

---

## 1. Project Overview

Network intrusion detection systems (NIDS) inspect traffic records and flag abnormal connections. This project treats that task as **supervised multiclass classification**: each row is one NSL-KDD **connection record** with 41 traffic features.

Attack names in the raw `label` column are mapped to five project classes:

| Class | Role |
|---|---|
| **Normal** | Benign traffic |
| **DoS** | Denial-of-service |
| **Probe** | Reconnaissance / scanning |
| **R2L** | Remote-to-local (unauthorized remote access) |
| **U2R** | User-to-root (privilege escalation) |

R2L and U2R are rare in KDDTrain+ and are the main source of error on KDDTest+.

---

## 2. Project Objectives

- Explore NSL-KDD (schema, imbalance, categorical vs numeric features).
- Establish baseline classifiers (Logistic Regression and related Day 2-3 controls).
- Investigate class imbalance (class weights; SMOTE on Logistic Regression).
- Evaluate Random Forest, including validation sensitivity to `n_estimators`.
- **Select a final model using KDDTrain+ validation only** (Macro F1).
- Evaluate generalization on held-out **KDDTest+** after lock.
- Analyze errors and class/feature distribution differences (post-hoc).
- Package the locked model for later deployment (joblib pipeline).

---

## 3. Dataset

**NSL-KDD** is used as provided in this project as two headerless files:

| Split | File | Role |
|---|---|---|
| KDDTrain+ | `data/raw/KDDTrain+.txt` | Training and **validation** (80/20 stratified split) |
| KDDTest+ | `data/raw/KDDTest+.txt` | **Held-out** final evaluation / reporting only |

Raw files have **43 columns**: 41 features + `label` + `difficulty`. The five-class target is `attack_category`, obtained by mapping original attack names (standard NSL-KDD grouping used throughout Days 1-10).

**KDDTest+ was not used for training, validation, hyperparameter choice, or model selection.** Local copies of the raw files are gitignored (`data/raw/*`).

No official dataset URL is recorded in this repository, so none is listed here.

---

## 4. Repository Structure

```
network-traffic-anomaly-detection/
|-- notebooks/
|   |-- 01_eda.ipynb
|   |-- 02_baseline_model.ipynb
|   |-- 04_smote_experiments.ipynb
|   |-- 05_random_forest.ipynb
|   |-- 06_model_analysis.ipynb
|   |-- 07_final_model_comparison.ipynb
|   |-- 08_final_evaluation.ipynb
|   |-- 09_reproducibility_audit.ipynb
|   `-- 10_finalization_and_packaging.ipynb
|-- models/
|   |-- nsl_kdd_random_forest_500.joblib
|   |-- nsl_kdd_random_forest_500_metadata.json
|   `-- nsl_kdd_random_forest_500_input_contract.json
|-- data/raw/          # local NSL-KDD files; contents gitignored
|-- requirements.txt
|-- .gitignore
`-- README.md
```

There is **intentionally no `03_*.ipynb`**. Day 2-3 work (preprocessing, baselines, class weighting, validation split) is in `notebooks/02_baseline_model.ipynb`.

| Notebook | Description |
|---|---|
| `01_eda.ipynb` | Load NSL-KDD, five-class map, class imbalance, feature types |
| `02_baseline_model.ipynb` | Preprocessor, LR/RF/XGB baselines, class weights, KDDTrain+ validation split |
| `04_smote_experiments.ipynb` | Logistic Regression with vs without SMOTE (validation, then KDDTest+) |
| `05_random_forest.ipynb` | RF validation, tree-count comparison, **model lock**, KDDTest+ evaluation |
| `06_model_analysis.ipynb` | Post-hoc error and distribution-shift analysis of the locked RF |
| `07_final_model_comparison.ipynb` | Reporting-only comparison of LR and locked RF |
| `08_final_evaluation.ipynb` | Final evidence record of locked-model metrics |
| `09_reproducibility_audit.ipynb` | Integrity/consistency audit (no modeling) |
| `10_finalization_and_packaging.ipynb` | Joblib pipeline packaging and validation smoke test |

---

## 5. Methodology

### Data preparation

- **41 model features**; `label` and `difficulty` are not used as predictors.
- **Categorical:** `protocol_type`, `service`, `flag` - `OneHotEncoder(handle_unknown="ignore")`.
- **Numerical:** remaining 38 columns - `StandardScaler`.
- **ColumnTransformer** fitted on the **training partition only**.

### Train/validation split (Day 5, used for selection and packaging)

| Item | Setting |
|---|---|
| Source | KDDTrain+ only |
| Split | 80% / 20%, **stratified**, `random_state=42` |
| Fit | `X_train_fit`, `y_train_fit` |
| Validation | `X_validation`, `y_validation` |
| Test | KDDTest+ held out until after lock |

### Models investigated

| Experiment | Notebook | Notes |
|---|---|---|
| Logistic Regression (including class-weighted) | Days 2-3 (`02_baseline_model.ipynb`) | Day 2 also compared Random Forest and XGBoost **configurations that are not the locked model** |
| Logistic Regression - no SMOTE | Day 4 | Validation control |
| Logistic Regression - SMOTE | Day 4 | SMOTE on the **fit** subset after numeric preprocess; not applied to the locked RF |
| Random Forest, `n_estimators` in {100, 300, **500**} | Day 5 | No `class_weight`, no SMOTE, no threshold tuning |

---

## 6. Model Selection

**Selection criterion:** KDDTrain+ **validation Macro F1**.

**Selected model:** Random Forest - **500 trees**.

| Setting | Value |
|---|---|
| `n_estimators` | 500 |
| `random_state` | 42 |
| `class_weight` | `None` |
| SMOTE | No |
| Threshold tuning | No |

Among the reported validation candidates (class-weighted LR, LR no SMOTE, LR SMOTE, RF 500), RF 500 had the highest validation Macro F1 (**0.951031**).

**KDDTest+ was not used for model selection.** Logistic Regression with SMOTE later showed a **higher KDDTest+ Macro F1** than the locked RF; that result is **reporting only** and did **not** replace the locked model.

---

## 7. Validation Results

Locked Random Forest - 500 trees, KDDTrain+ validation split:

| Metric | Random Forest - 500 |
|---|---:|
| Accuracy | 0.998809 |
| Macro Precision | 0.974485 |
| Macro Recall | 0.931597 |
| Macro F1 | 0.951031 |
| R2L F1 | 0.982097 |
| U2R F1 | 0.777778 |

After packaging (Day 10), the joblib pipeline reproduced these validation metrics to rounding (absolute differences on the order of \(10^{-7}\)).

---

## 8. KDDTest+ Results

**Held-out evaluation / reporting only** (Day 5, after lock). Not used to select or tune the model.

| Metric | Random Forest - 500 |
|---|---:|
| Accuracy | 0.7447 |
| Macro Precision | 0.8198 |
| Macro Recall | 0.4896 |
| Macro F1 | 0.5061 |
| R2L Precision | 0.978571 |
| R2L Recall | 0.047487 |
| R2L F1 | 0.090579 |
| U2R Precision | 0.666667 |
| U2R Recall | 0.059701 |
| U2R F1 | 0.109589 |

---

## 9. Generalization Findings

| Quantity | Value |
|---|---|
| Validation Macro F1 | 0.951031 |
| KDDTest+ Macro F1 | 0.5061 |
| Absolute Macro F1 drop | 0.4449 |
| Relative Macro F1 drop | 46.78% |
| R2L recall | 0.9648 -> 0.0475 |
| U2R recall | 0.7000 -> 0.0597 |

The drop is a **substantial validation-to-test generalization gap**, especially for rare classes R2L and U2R. High validation scores on a KDDTrain+ split do not imply comparable KDDTest+ performance.

---

## 10. Error Analysis

Dominant KDDTest+ actual -> predicted counts (locked RF, existing Day 5 confusion matrix):

| Actual -> predicted | Count |
|---|---:|
| R2L -> Normal | 2744 |
| DoS -> Normal | 1649 |
| Probe -> Normal | 812 |
| U2R -> Normal | 60 |

Most missed **R2L** (2744 of 2885) and **U2R** (60 of 67) rows were labeled **Normal**. R2L precision on KDDTest+ remained high while recall was very low (conservative R2L predictions). This describes the error pattern; it does **not** prove that any particular feature caused the mistakes.

---

## 11. Distribution Shift

Day 6 post-hoc comparison of the validation split vs KDDTest+ (not used for selection):

| Class | Validation prevalence | KDDTest+ prevalence |
|---|---:|---:|
| R2L | 0.7898% | 12.7972% |
| U2R | 0.0397% | 0.2972% |

Largest reported overall numerical standardized mean difference (|SMD|) = **0.5060**. R2L/U2R **conditional** feature distributions showed larger shifts than the overall table.

The class-mix and numerical-feature differences are **consistent with dataset/domain shift** and **help explain** why validation performance did not transfer to KDDTest+. They are **not** shown to be the sole cause, and **no individual feature is claimed to have caused** the errors.

---

## 12. Final Model Artifact

| File | Role |
|---|---|
| [`models/nsl_kdd_random_forest_500.joblib`](models/nsl_kdd_random_forest_500.joblib) | `Pipeline`: `ColumnTransformer` (OHE + `StandardScaler`) + `RandomForestClassifier` |
| [`models/nsl_kdd_random_forest_500_metadata.json`](models/nsl_kdd_random_forest_500_metadata.json) | Locked config, split, recorded validation metrics, library versions |
| [`models/nsl_kdd_random_forest_500_input_contract.json`](models/nsl_kdd_random_forest_500_input_contract.json) | Feature names, order, output classes |

Load:

```python
import joblib
model = joblib.load("models/nsl_kdd_random_forest_500.joblib")
# y_hat = model.predict(X)  # X must match the input contract
```

The pipeline was fitted on `X_train_fit` / `y_train_fit` only. After serialization it was **smoke-tested on the validation split** and reproduced the Day 5 validation metrics above. **KDDTest+ was not scored during packaging.**

The joblib file is on the order of **56 MiB**. Prefer Git LFS or other artifact storage if the binary should not live as a plain Git blob.

---

## 13. Deployment Readiness

The locked model is a **deployment-usable artifact** (packaged for later integration; not a production service). Conceptual inference flow:

```
Client request
  -> backend API
  -> validate input against the feature contract
  -> joblib.load(...)
  -> pipeline.predict(...)
  -> return one of: Normal, DoS, Probe, R2L, U2R
```

No backend, API, or UI is implemented here. Future application code must supply the **same 41 feature names and types** (and column order) as Day 5. Output classes: **Normal, DoS, Probe, R2L, U2R**.

---

## 14. Reproducibility

Recorded environment (Day 10 packaging / `requirements.txt`):

| Component | Version |
|---|---|
| Python | 3.10.11 |
| scikit-learn | 1.7.2 |
| joblib | 1.5.2 |
| pandas | 2.3.3 |
| numpy | 2.2.6 |

Other listed dependencies: matplotlib, seaborn, xgboost, ipykernel. Split and model settings are recorded in notebooks and metadata. This documents the experimental configuration; **bit-for-bit identical runs across machines are not claimed.**

---

## 15. Experimental Integrity

- **KDDTest+ was evaluation-only.**
- **It was not used for model selection.**
- No post-test tuning.
- No threshold tuning.
- No SMOTE on the locked Random Forest.
- The locked Random Forest was **not replaced** after KDDTest+ evaluation.
- Days 6-9 are reporting, diagnostic, and audit work.
- Day 10 **packages the already-selected model**.

---

## 16. Limitations

- Large KDDTest+ Macro F1 drop (0.951031 -> 0.5061).
- Very low KDDTest+ **R2L** and **U2R** recall.
- Class prevalence and feature distributions differ between the KDDTrain+ validation slice and KDDTest+.
- U2R validation support is small (10 rows), so validation U2R scores are unstable.
- NSL-KDD is a **benchmark**, not a guarantee of performance on live production traffic.
- Integration into an application still requires schema validation, monitoring, and evaluation on **representative** operational data.

This project is **not** described as production-ready.

---

## 17. Project Status

| Phase | Status |
|---|---|
| Machine-learning experimentation | COMPLETE |
| Model selection | COMPLETE |
| Model evaluation | COMPLETE |
| Model analysis | COMPLETE |
| Reproducibility / integrity audit | COMPLETE |
| Model packaging | COMPLETE |
| Documentation | COMPLETE |
| Full-stack integration | **NOT YET IMPLEMENTED** |

The next phase is **application / backend integration**, not additional model search or KDDTest+-based tuning.

---

## 18. License / Citation

This repository does not currently include a license file or a formal citation record. Add a license of your choice before public redistribution. Cite NSL-KDD according to the dataset's original authors and distributors when using these results.
