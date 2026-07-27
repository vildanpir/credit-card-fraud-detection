# Credit Card Fraud Detection

A supervised Machine Learning project that predicts whether a credit card transaction is fraudulent.

Ironhack Data Analytics Bootcamp — Machine Learning Project Week.

## Project Overview

**What am I predicting?** The column `fraud` — `1` if the transaction is fraudulent, `0` if it is legitimate.
Two possible outcomes, so this is a **binary classification** problem.

**With what?** 7 transaction features: `distance_from_home`, `distance_from_last_transaction`,
`ratio_to_median_purchase_price`, `repeat_retailer`, `used_chip`, `used_pin_number`, `online_order`.

**Why this dataset?** I want to work in the banking/finance sector after the bootcamp, and fraud
detection is one of the most common Machine Learning use cases in that industry.

## Dataset

- **Source:** [Credit Card Fraud — Kaggle](https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud)
- **Size:** 1,000,000 rows × 8 columns · no missing values
- **Class balance:** 91.3% legitimate / 8.7% fraud — imbalanced
- **Sample used:** 50,000 random rows (`random_state=17`), because KNN is slow on 1M rows.
  The sample keeps the same class balance.

The full file is not in this repo because of its size — download it from the Kaggle link above.

## Author

**Vildan Pirpiroglu** — Ironhack, 2026
