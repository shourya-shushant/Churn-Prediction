# Customer Churn Prediction & Business Analytics

**Python · Pandas · Scikit-learn · SQL · Streamlit**

An end-to-end churn prediction system: data cleaning, EDA, feature engineering,
model comparison (Logistic Regression, Decision Tree, Random Forest), SQL analysis
and an interactive Streamlit dashboard.

## Project structure

```
churn-prediction/
├── app.py                      # Streamlit dashboard
├── requirements.txt
├── data/
│   ├── raw/customers.csv       # raw (synthetic) data
│   └── processed/              # cleaned + scored data
├── models/best_model.joblib    # saved best pipeline
├── reports/                    # metrics.csv, feature_importance.csv, figures/
├── sql/analysis.sql            # business-analytics queries
└── src/
    ├── generate_data.py        # creates the dataset
    ├── features.py             # cleaning + feature engineering
    ├── train.py                # EDA, training, evaluation
    └── run_sql.py              # loads SQLite DB and runs the SQL
```

## Run it

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/generate_data.py        # 1. create data (skip if using your own CSV)
python -m src.train                # 2. clean, EDA, train, evaluate
python -m src.run_sql              # 3. SQL business analysis
streamlit run app.py               # 4. launch dashboard
```

## Approach

1. **Cleaning** – dropped duplicates, fixed blank `total_charges`, encoded target.
2. **EDA** – churn by contract, internet service, tenure (`reports/figures/eda.png`).
3. **Feature engineering** – tenure groups, average monthly spend, number of add-on services.
4. **Modeling** – three classifiers compared on accuracy, precision, recall, F1 and confusion matrix
   (`reports/metrics.csv`, `reports/figures/confusion_matrices.png`).
5. **SQL** – churn by contract / tenure / payment method and revenue lost (`sql/analysis.sql`).
6. **Dashboard** – KPIs, filters, model comparison, high-risk customer list and a single-customer predictor.

## Key insights
- Month-to-month customers churn far more than annual/two-year customers.
- Churn is highest in the first year of tenure.
- Fiber customers without tech support are the riskiest segment.
