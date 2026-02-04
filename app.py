import streamlit as st
import subprocess
import pandas as pd
import sys
import os

st.set_page_config(page_title="Revenue Forecasting App", layout="wide")

st.title("📊 Revenue Forecasting & Trend Analysis")

def run_script(script_name):
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True
        )
        return result.stdout, result.stderr
    except Exception as e:
        return "", str(e)

menu = st.sidebar.selectbox(
    "Select Module",
    [
        "Dataset Generation",
        "Data Preprocessing",
        "Baseline Models",
        "ARIMA Model",
        "ML Regression",
        "Hybrid Model",
        "Trend Detection"
    ]
)

if menu == "Dataset Generation":
    st.subheader("Generate Synthetic Revenue Data")

    if st.button("Run data.py"):
        out, err = run_script("data.py")
        st.code(out)
        if err:
            st.error(err)


elif menu == "Data Preprocessing":
    st.subheader("Preprocess Daily Revenue")

    if st.button("Run processed_data.py"):
        out, err = run_script("processed_data.py")
        st.code(out)
        if err:
            st.error(err)

    if os.path.exists("processed_daily_revenue.csv"):
        df = pd.read_csv("processed_daily_revenue.csv")
        st.dataframe(df)
        st.line_chart(df["revenue"])


elif menu == "Baseline Models":
    st.subheader("Naive & Moving Average Forecasts")

    if st.button("Run fund_models.py"):
        out, err = run_script("fund_models.py")
        st.code(out)
        if err:
            st.error(err)


elif menu == "ARIMA Model":
    st.subheader("ARIMA Forecasting")

    if st.button("Run arima_model.py"):
        out, err = run_script("arima_model.py")
        st.code(out)
        if err:
            st.error(err)


elif menu == "ML Regression":
    st.subheader("Machine Learning Regression")

    if st.button("Run regression.py"):
        out, err = run_script("regression.py")
        st.code(out)
        if err:
            st.error(err)

elif menu == "Hybrid Model":
    st.subheader("Adaptive Hybrid Forecasting")

    if st.button("Run hybrid.py"):
        out, err = run_script("hybrid.py")
        st.code(out)
        if err:
            st.error(err)


elif menu == "Trend Detection":
    st.subheader("Trend Regime Detection")

    if st.button("Run trend.py"):
        out, err = run_script("trend.py")
        st.code(out)
        if err:
            st.error(err)
