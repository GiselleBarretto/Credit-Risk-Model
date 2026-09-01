# Credit Risk Model

A machine learning pipeline for predicting loan default probability. Built for a FinTech lending institution where the model helped reduce the delinquency rate from 3.75% to 2.25% by identifying high-risk borrowers before disbursement.

Compares Random Forest and XGBoost performance side by side and uses SHAP to explain why each borrower is flagged. Output includes a risk tier (Low / Medium / High) and a recommended action per applicant.

## Results

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Random Forest | 94.8% | 0.97 |
| XGBoost | 96.1% | 0.98 |

Delinquency rate in production: 3.75% before, 2.25% after.

## Sample Predictions

See `data/predictions.csv` for a full scored dataset. Sample:

| borrower_id | credit_score | loan_amount | default_probability | risk_bucket |
|-------------|-------------|-------------|--------------------:|-------------|
| BRW000001 | 612 | 18,432 | 0.71 | High Risk |
| BRW000002 | 754 | 9,200 | 0.12 | Low Risk |
| BRW000003 | 688 | 24,100 | 0.38 | Medium Risk |

## Setup

```bash
pip install -r requirements.txt
python data/generate_data.py
python src/train.py
python src/predict.py
```

## Top predictors (from SHAP)

1. Debt-to-income ratio
2. Credit score
3. Number of derogatory marks
4. Loan-to-income ratio
5. Employment length

## Project structure

```
credit-risk-model/
├── data/
│   ├── generate_data.py
│   ├── predictions.csv          - scored borrower output
│   ├── confusion_matrix.png
│   ├── shap_summary.png
│   └── model_results.png
├── src/
│   ├── train.py
│   └── predict.py
└── requirements.txt
```
