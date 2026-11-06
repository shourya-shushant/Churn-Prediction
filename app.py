"""Streamlit dashboard:  streamlit run app.py"""
import joblib
import pandas as pd
import streamlit as st

from src.features import FEATURES, add_features, clean

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")


@st.cache_data
def load_data():
    return (pd.read_csv("data/processed/customers_scored.csv"),
            pd.read_csv("reports/metrics.csv"),
            pd.read_csv("reports/feature_importance.csv"))


@st.cache_resource
def load_model():
    return joblib.load("models/best_model.joblib")


df, metrics, importance = load_data()
model = load_model()

st.title("📉 Customer Churn Prediction & Business Analytics")

# ---- sidebar filters ----
st.sidebar.header("Filters")
contracts = st.sidebar.multiselect("Contract", sorted(df.contract.unique()), default=sorted(df.contract.unique()))
internet = st.sidebar.multiselect("Internet service", sorted(df.internet_service.unique()),
                                  default=sorted(df.internet_service.unique()))
view = df[df.contract.isin(contracts) & df.internet_service.isin(internet)]

tab1, tab2, tab3 = st.tabs(["Business overview", "Model performance", "Predict a customer"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(view):,}")
    c2.metric("Churn rate", f"{view.churn.mean():.1%}")
    c3.metric("Monthly revenue lost", f"${view.loc[view.churn == 1, 'monthly_charges'].sum():,.0f}")
    high = view[view.churn_probability >= 0.6]
    c4.metric("High-risk customers (p≥0.6)", f"{len(high):,}")

    a, b = st.columns(2)
    a.subheader("Churn rate by contract")
    a.bar_chart(view.groupby("contract").churn.mean())
    b.subheader("Churn rate by tenure group")
    b.bar_chart(view.groupby("tenure_group").churn.mean())
    a.subheader("Churn rate by payment method")
    a.bar_chart(view.groupby("payment_method").churn.mean())
    b.subheader("Churn rate by internet service")
    b.bar_chart(view.groupby("internet_service").churn.mean())

    st.subheader("Top 20 customers most likely to churn")
    st.dataframe(
        view[view.churn == 0].nlargest(20, "churn_probability")
        [["customer_id", "contract", "tenure", "monthly_charges", "churn_probability"]],
        use_container_width=True,
    )

with tab2:
    st.subheader("Model comparison (hold-out test set)")
    st.dataframe(metrics, use_container_width=True)
    st.image("reports/figures/confusion_matrices.png", caption="Confusion matrices")
    st.subheader("Top drivers of churn (Random Forest)")
    st.bar_chart(importance.set_index("feature"))
    st.image("reports/figures/eda.png", caption="Exploratory analysis")

with tab3:
    st.subheader("Enter customer details")
    c1, c2, c3 = st.columns(3)
    contract = c1.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    tenure = c1.slider("Tenure (months)", 0, 72, 12)
    monthly = c1.number_input("Monthly charges ($)", 18.0, 150.0, 70.0)
    internet_s = c2.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    payment = c2.selectbox("Payment method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
    paperless = c2.selectbox("Paperless billing", ["Yes", "No"])
    tech = c3.selectbox("Tech support", ["Yes", "No"])
    security = c3.selectbox("Online security", ["Yes", "No"])
    senior = c3.selectbox("Senior citizen", [0, 1])

    if st.button("Predict churn risk"):
        row = pd.DataFrame([{
            "customer_id": "NEW", "gender": "Male", "senior_citizen": senior, "partner": "No",
            "dependents": "No", "tenure": tenure, "contract": contract, "internet_service": internet_s,
            "online_security": security, "tech_support": tech, "streaming_tv": "No",
            "online_backup": "No", "paperless_billing": paperless, "payment_method": payment,
            "monthly_charges": monthly, "total_charges": monthly * tenure,
        }])
        prob = model.predict_proba(add_features(clean(row))[FEATURES])[0, 1]
        st.metric("Churn probability", f"{prob:.1%}")
        st.progress(float(prob))
        st.write("🔴 High risk" if prob >= 0.6 else "🟠 Medium risk" if prob >= 0.35 else "🟢 Low risk")
