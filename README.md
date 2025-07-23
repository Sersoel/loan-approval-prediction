# Loan Approval Prediction using Random Forest

> Predict loan approval outcomes using applicant data and a tuned Random Forest classifier

---

## Overview

Financial institutions often struggle with manual, inconsistent, and biased loan approval processes, leading to:
- Risky approvals increasing default rates  
- Rejections of eligible applicants, hurting business growth

This project presents a **Random Forest–based machine learning model** trained on historical loan data to accurately predict loan approval outcomes.  
- **Original dataset**: 36 columns  
- **Post-preprocessing**: 45 numerical features (after encoding)

---

## Dataset

- **Source**: Kaggle – [Financial Risk for Loan Approval](https://www.kaggle.com/datasets/lorenzozoppelletto/financial-risk-for-loan-approval/?select=Loan.csv)    
- **Records**: 20,000 samples  
- **Target Variable**: `LoanApproved` (0 = Rejected, 1 = Approved)  
- **Class Balance**: Nearly balanced

---

## Exploratory Data Analysis (EDA)

- Visualized key numerical features (e.g., `ApplicantIncome`, `RiskScore`, `TotalDebtToIncomeRatio`)
- Analyzed class distribution
- Verified dataset is suitable for classification with minimal imbalance

---

## Data Preprocessing

- Started with **36 raw columns**
- Removed missing values
- Dropped irrelevant or high-cardinality columns (e.g., IDs, emails)
- One-hot encoded low-cardinality categorical features
- Resulting dataset: **45 numeric features**
- Applied an 80/20 **train-test split** with stratification

---

## Modeling Approach

I used a **Random Forest Classifier** due to its robustness, explainability, and ability to model non-linear financial behaviors.

> Learns complex patterns  
> Reduces variance via tree ensemble  
> Provides interpretable feature importance

---

## Hyperparameter Tuning

We fine-tuned the model using `GridSearchCV` with 5-fold cross-validation.

### Tuned Parameters:
| Hyperparameter | Values Tried                        | Best |
|----------------|-------------------------------------|------|
| `n_estimators` | [100, 300, 500, 1000]               | 100  |
| `max_depth`    | [10, 15, 20, None]                  | 15   |
| `max_features` | ['sqrt', 'log2', 0.5, 1.0]          | 0.5  |

**Scoring Metric**: Accuracy  
**Final Accuracy**: **99.6%**

---

## Evaluation

Confusion Matrix (original model):

| Actual / Predicted | Rejected | Approved |
|--------------------|----------|----------|
| **Rejected**       | 2,247    | 6        |
| **Approved**       | 11       | 1,736    |

- High precision and recall  
- Low false negatives – crucial in financial approval systems

---

## Feature Importance

Top features by model weight:
1. **RiskScore** – ~74% of decision power  
2. **TotalDebtToIncomeRatio** – ~10%  
3. Remaining features contributed marginally

---

## Impact Analysis (Ablation Study)

After dropping `RiskScore`:
- Accuracy dropped from **99.6% → 90.2%**
- False Negatives increased from 11 to 191
- False Positives rose from 6 to 200

Despite the drop, the model still retained decent performance, showing robustness.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/sersoel/loan-approval-prediction.git
cd loan-approval-prediction
```

### 2. Set Up the Environment

You can install the dependencies in two ways:

### Option A – Using requirements.txt (Recommended):

```bash
pip install -r requirements.txt
```
### or Option B – Manual Installation:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### 3. (Optional) Use a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # On macOS/Linux
venv\Scripts\activate         # On Windows
```

### 4. Run the Project

Launch Jupyter Notebook:
```bash
jupyter notebook
```

Then open `notebooks/loan_classification.ipynb` and run the cells.

---

## Contact

- shl.smdn@gmail.com
- 5893603@studenti.unige.it
- University of Genoa – MSc. Computer Engineering (AI & Human-Centered Computing)