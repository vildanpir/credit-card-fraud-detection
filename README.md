# Credit Card Fraud Detection

A supervised Machine Learning project that predicts whether a credit card transaction is fraudulent.

*Ironhack Data Analytics Bootcamp — Machine Learning Project Week (2026)*

---

## 🎯 Project Overview

**What am I predicting?** The column `fraud` — `1` if the transaction is fraudulent, `0` if it is legitimate. Two possible outcomes, so this is a **binary classification** problem.

**Why this dataset?** I want to work in the banking/finance sector after the bootcamp, and fraud detection is one of the most common Machine Learning use cases in that industry.

**The catch:** only **8.7%** of transactions are fraud. The data is strongly **imbalanced**, which shapes every decision in the project — especially how I measure success.

---

## 📊 Dataset

- **Source:** [Credit Card Fraud — Kaggle](https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud)
- **Size:** 1,000,000 rows × 8 columns · no missing values
- **Class balance:** 91.3% legitimate / 8.7% fraud (imbalanced)
- **Sample used:** 50,000 random rows (`random_state=17`), because KNN is slow on 1M rows. The sample keeps the same class balance.
- **Features:** `distance_from_home`, `distance_from_last_transaction`, `ratio_to_median_purchase_price`, `repeat_retailer`, `used_chip`, `used_pin_number`, `online_order`

*The full file is not in this repo because of its size — download it from the Kaggle link above. A 50k sample is included.*

---

## 🛠️ Tools & Libraries

`Python` · `pandas` · `numpy` · `scikit-learn` · `imbalanced-learn` (SMOTE) · `matplotlib` · `seaborn` · `Jupyter`

---

## 🔍 Methodology

**1. Cleaning & EDA**
No missing values, no duplicates. Used `groupby` to find which features separate fraud from non-fraud.

**2. Feature engineering**
Created new signals from the raw data:
- **log transforms** (`log1p`) to tame the skewed distance/ratio columns
- **`unusual_amount`** — purchase > 2× the customer's median
- **`security_score`** — chip + PIN used (in-person signals)
- **`online_no_pin`** — classic card-not-present risk pattern

> Example finding: a normal transaction is fraud ~1.2% of the time, but an **online order with no PIN is fraud ~13.8%** of the time — a very strong signal.

**3. Feature selection**
Used a correlation matrix to drop redundant raw columns (kept the `log_` versions).

**4. Scaling**
`MinMaxScaler`, **fit on the training set only** (to avoid data leakage). KNN is distance-based, so scaling matters.

**5. Models — 10 compared on the same test set**
KNN (baseline / tuned / + SMOTE), Logistic Regression, Decision Tree, Random Forest, Random Forest (balanced), Bagging, AdaBoost, Gradient Boosting.

**6. Metric choice**
On imbalanced data **accuracy is misleading** (a model that always predicts "not fraud" scores ~91%). So I judged models on **recall, precision and F1 for the fraud class**, plus the confusion matrix.

**7. Tuning & imbalance handling**
- Hyperparameter tuning with **GridSearchCV** and **RandomizedSearchCV**
- **5-fold Stratified Cross-Validation** (scoring on fraud-class F1)
- Compared **Oversampling / Undersampling / SMOTE** on the training set only

---

## 📈 Results — model leaderboard (fraud class)

| Model | Recall | Precision | F1 |
|---|---|---|---|
| **Decision Tree** | 0.999 | 0.999 | **0.999** |
| Random Forest | 0.998 | 1.000 | 0.999 |
| Random Forest (balanced) | 0.999 | 0.999 | 0.999 |
| Bagging | 0.998 | 1.000 | 0.999 |
| AdaBoost | 0.999 | 0.999 | 0.999 |
| Gradient Boosting | 0.998 | 0.999 | 0.998 |
| KNN (tuned) | 0.951 | 0.987 | 0.968 |
| KNN (baseline) | 0.950 | 0.971 | 0.960 |
| KNN + SMOTE | 0.996 | 0.885 | 0.937 |
| Logistic Regression | 0.914 | 0.415 | 0.571 |

**Final model — Decision Tree.** On the test set it caught **855 of 856 real frauds** (1 missed), with almost no false alarms.

**Key findings**
- The **engineered features** (`log_ratio_to_median`, `online_no_pin`, `unusual_amount`) are the most important drivers — feature engineering did the heavy lifting.
- Rebalancing (SMOTE) lifted recall from **0.63 → 0.91** on a Logistic Regression baseline, but lowered precision — the classic **recall/precision trade-off**.
- The very high score (~99.9%) is honest: this dataset follows near-deterministic rules that my features captured. **No data leakage** — I split first, fit the scaler and SMOTE on training data only, and judged on the untouched test set.

---

## 🌍 Real-world application & limitations

This model works as a **first-pass filter**: it flags risky transactions for a human to review — it does **not** block cards on its own. For a bank, a **missed fraud (false negative) is more costly** than a false alarm, so I optimise for recall.

**Limitations:** the dataset is simplified (no transaction amount or time); fraud patterns drift over time, so the model would need retraining; and it must stay fair to customers.

---

## 📁 Repository contents

- `fraud_detection.ipynb` — full analysis, code and visualizations
- `card_transdata_sample.csv` — 50,000-row sample of the dataset
- `README.md` — this file

## ▶️ How to run

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
```
Open `fraud_detection.ipynb` in Jupyter and run all cells (the sample CSV is in the repo).

---

## 👤 Author

**Vildan Pirpiroglu** — Ironhack Data Analytics Bootcamp, 2026
