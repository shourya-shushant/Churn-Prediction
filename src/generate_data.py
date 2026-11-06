"""Generate a synthetic telecom-style customer dataset (data/raw/customers.csv).

Replace this file's output with a real dataset (e.g. the Kaggle
'Telco Customer Churn' CSV) if you like - just keep the same column names
or adjust src/features.py.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 7000

tenure = rng.integers(0, 73, N)
contract = rng.choice(["Month-to-month", "One year", "Two year"], N, p=[0.55, 0.21, 0.24])
internet = rng.choice(["DSL", "Fiber optic", "No"], N, p=[0.34, 0.44, 0.22])
payment = rng.choice(
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], N, p=[0.34, 0.23, 0.22, 0.21]
)
paperless = rng.choice(["Yes", "No"], N, p=[0.59, 0.41])
senior = rng.choice([0, 1], N, p=[0.84, 0.16])


def addon(p_with_internet=0.45):
    vals = rng.choice(["Yes", "No"], N, p=[p_with_internet, 1 - p_with_internet])
    return np.where(internet == "No", "No internet service", vals)


online_security = addon(0.40)
tech_support = addon(0.40)
streaming_tv = addon(0.50)
online_backup = addon(0.45)

base = np.select([internet == "No", internet == "DSL", internet == "Fiber optic"], [20, 45, 75])
extras = sum((a == "Yes") * 6 for a in [online_security, tech_support, streaming_tv, online_backup])
monthly = np.round(base + extras + rng.normal(0, 4, N), 2).clip(18, None)

logit = (
    -0.9
    + 1.3 * (contract == "Month-to-month")
    - 1.0 * (contract == "Two year")
    + 0.7 * (internet == "Fiber optic")
    - 0.035 * tenure
    + 0.012 * (monthly - 65)
    + 0.45 * (payment == "Electronic check")
    + 0.25 * (paperless == "Yes")
    + 0.3 * senior
    - 0.6 * (tech_support == "Yes")
    - 0.5 * (online_security == "Yes")
    + rng.normal(0, 0.6, N)
)
churn = rng.random(N) < 1 / (1 + np.exp(-logit))

total = np.round(monthly * tenure + rng.normal(0, 15, N).clip(-30, 30) * (tenure > 0), 2)
total = np.where(total < 0, 0, total).astype(object)
total[tenure == 0] = " "  # blank strings, like the real Telco file -> must be cleaned

df = pd.DataFrame(
    {
        "customer_id": [f"CUST-{i:05d}" for i in range(N)],
        "gender": rng.choice(["Male", "Female"], N),
        "senior_citizen": senior,
        "partner": rng.choice(["Yes", "No"], N),
        "dependents": rng.choice(["Yes", "No"], N, p=[0.3, 0.7]),
        "tenure": tenure,
        "contract": contract,
        "internet_service": internet,
        "online_security": online_security,
        "tech_support": tech_support,
        "streaming_tv": streaming_tv,
        "online_backup": online_backup,
        "paperless_billing": paperless,
        "payment_method": payment,
        "monthly_charges": monthly,
        "total_charges": total,
        "churn": np.where(churn, "Yes", "No"),
    }
)
# add a few duplicate rows so the cleaning step has something to do
df = pd.concat([df, df.sample(25, random_state=1)], ignore_index=True)
df.to_csv("data/raw/customers.csv", index=False)
print(f"Saved data/raw/customers.csv  rows={len(df)}  churn rate={churn.mean():.1%}")
