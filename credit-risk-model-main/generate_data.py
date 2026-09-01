import pandas as pd
import numpy as np

np.random.seed(42)
N = 50_000

print("Generating synthetic loan data...")

age = np.random.randint(22, 65, N)
income = np.random.normal(65000, 25000, N).clip(20000, 250000)
loan_amount = np.random.normal(15000, 8000, N).clip(1000, 80000)
loan_term = np.random.choice([12, 24, 36, 48, 60], N)
credit_score = np.random.normal(680, 80, N).clip(300, 850)
debt_to_income = np.random.beta(2, 5, N)
employment_years = np.random.exponential(5, N).clip(0, 40)
num_open_accounts = np.random.randint(1, 20, N)
num_derogatory_marks = np.random.poisson(0.3, N)
late_payments_2yr = np.random.poisson(0.4, N)
loan_purpose = np.random.choice(["debt_consolidation","home_improvement","business","education","personal"], N)

# Default probability
log_odds = (
    -3.5
    + 0.015 * (700 - credit_score) / 100
    + 1.2 * debt_to_income
    + 0.5 * num_derogatory_marks
    + 0.4 * late_payments_2yr
    - 0.3 * (employment_years / 10)
    - 0.2 * (income / 100000)
    + 0.1 * (loan_amount / income)
    + np.random.normal(0, 0.3, N)
)
prob_default = 1 / (1 + np.exp(-log_odds))
default = (np.random.uniform(0, 1, N) < prob_default).astype(int)

df = pd.DataFrame({
    "age": age,
    "annual_income": income.round(2),
    "loan_amount": loan_amount.round(2),
    "loan_term_months": loan_term,
    "credit_score": credit_score.round(0).astype(int),
    "debt_to_income_ratio": debt_to_income.round(4),
    "employment_years": employment_years.round(1),
    "num_open_accounts": num_open_accounts,
    "num_derogatory_marks": num_derogatory_marks,
    "late_payments_2yr": late_payments_2yr,
    "loan_purpose": loan_purpose,
    "default": default,
})

df.to_csv("data/loan_data.csv", index=False)
print(f"Done. {N:,} records. Default rate: {default.mean():.2%}")
