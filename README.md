# Customer Campaign Response Prediction – ML Classification Assignment

## 1. Problem Statement
Predict whether a customer will **accept the company's latest marketing campaign offer** (`Response` = 1) using demographic, purchase-behaviour and past-campaign data. Targeting likely responders reduces wasted marketing spend. This is a **binary classification** problem; five models are trained and compared on the same data.

## 2. Dataset Description
- **Source:** Kaggle – *Customer Personality Analysis* (`marketing_campaign.csv`, tab-separated).
- **Instances:** 2,240 raw rows (2,055 after removing duplicates and impossible ages). **Features:** 28 raw (25 used after cleaning/engineering) – meets the ≥12 features / ≥1,000 instances requirement.
- **Why chosen:** realistic business problem, mixed numeric + categorical features, genuine missing values (`Income`), and a clearly imbalanced target.
- **Target:** `Response` (1 = accepted last campaign, 0 = did not).
- **Feature types:**
  - Numeric: Income, Kidhome, Teenhome, Recency, Mnt* (6 spend columns), Num* purchase/visit counts, AcceptedCmp1–5, Complain, Age*, Customer_Tenure_Days*
  - Categorical: Education, Marital_Status
  - (*engineered from Year_Birth and Dt_Customer)
- **Class distribution (raw):** 0 → 1,906 (85.1 %), 1 → 334 (14.9 %) → **imbalanced**, so accuracy alone is misleading; AUC, recall, F1 and MCC are emphasised. The 80:20 split is stratified (train 15.2 % positives, test 15.3 %).

### Preprocessing
| Step | Action |
|---|---|
| Missing values | `Income` has 24 NaN → median imputation (fitted on train only) |
| Duplicates | 182 duplicate customers (ignoring ID) removed |
| Dropped columns | `ID` (identifier), `Z_CostContact`, `Z_Revenue` (constant), `Year_Birth`, `Dt_Customer` (replaced by Age and tenure) |
| Invalid data | 3 customers with Age > 100 removed; `Alone`→`Single`, `Absurd`/`YOLO`→`Other` |
| Encoding | One-hot (drop first) for Education, Marital_Status |
| Scaling | StandardScaler for Logistic Regression, kNN, Naive Bayes (not needed for tree models) |
| Split | 80:20, `stratify=y`, `random_state=42` |
| Leakage control | Imputer/scaler/encoder live inside a scikit-learn `Pipeline`, fitted on the training set only |

## 3. GitHub Repository Link
https://github.com/2026CT05057-ShrutiSinha/Machine-Learning/tree/main

## 4. Models Used
Configurations: Logistic Regression (max_iter=1000) · Decision Tree (max_depth=5, min_samples_leaf=10) · kNN (k=7) · Gaussian Naive Bayes · Random Forest (300 trees). All with seed 42.

### Comparison table (test set, 411 samples)
| ML Model Name            |   Accuracy |    AUC |   Precision |   Recall |     F1 |    MCC |
|:-------------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| Logistic Regression      |     0.8783 | 0.8736 |      0.6383 |   0.4762 | 0.5455 | 0.4838 |
| Decision Tree            |     0.854  | 0.7483 |      0.5429 |   0.3016 | 0.3878 | 0.3299 |
| kNN                      |     0.8613 | 0.7938 |      0.6071 |   0.2698 | 0.3736 | 0.3406 |
| Naive Bayes              |     0.8321 | 0.8071 |      0.4643 |   0.619  | 0.5306 | 0.4375 |
| Random Forest (Ensemble) |     0.8613 | 0.8542 |      0.6071 |   0.2698 | 0.3736 | 0.3406 |

### Visualisations
![metrics by model](results/metrics_by_model.png)
![metrics by metric](results/metrics_by_metric.png)
![heatmap](results/metrics_heatmap.png)
![confusion matrices](results/confusion_matrices.png)

### Observations
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall: highest accuracy (0.878), AUC (0.874), precision-recall balance (F1 0.546) and MCC (0.484). The relationship between features (past campaign acceptance, recency, spend) and response is largely linear, and scaling plus a simple model generalises well on a small dataset. |
| Decision Tree | Weakest AUC (0.748) and low recall (0.302). A depth-5 tree is easy to interpret but unstable and overfits small training data; it misses most responders. |
| kNN | Accuracy 0.861 looks fine but recall is only 0.270 – with 85 % negatives, neighbours are mostly non-responders, so the minority class is under-predicted. AUC 0.794. |
| Naive Bayes | Highest recall (0.619) but lowest precision (0.464) and accuracy (0.832): it flags many customers as responders (many false positives). Useful if missing a responder is costlier than contacting a non-responder. |
| Random Forest (Ensemble) | Second-best AUC (0.854), much better than a single tree, showing the benefit of ensembling. With the default 0.5 threshold its recall is low (0.270) because of imbalance; class weights or threshold tuning would likely help. |
| **Overall Winner** | **Logistic Regression** – best on Accuracy, AUC, F1 and MCC, and simple/interpretable. |

## 5. How to Run
```bash
pip install -r requirements.txt
python model/train_models.py
```
Outputs: trained pipelines (`model/*.pkl`), `test_data.csv`, and `results/` (metrics + plots).

## 6. Repository Structure
```
├── requirements.txt
├── README.md
├── marketing_campaign.csv
├── test_data.csv
├── model/   (train_models.py + saved *.pkl pipelines)
└── results/ (metrics.csv, plots)
```
