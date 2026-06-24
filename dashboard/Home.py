import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(os.path.dirname(__file__))
import api_client

st.set_page_config(
    page_title="Dengue Forecasting",
    page_icon="🦟",
    layout="wide",
)

st.title("Dengue Brazil — Forecasting Dashboard")
st.caption(
    "Real-time forecasting powered by LSTM and PLE models trained on SINAN data."
)

st.divider()

# -------------------------------------------------------
# Summary cards
# -------------------------------------------------------
try:
    summary = api_client.get_summary()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Notifications", f"{summary['total_notifications']:,}")
    col2.metric("Total Deaths", f"{summary['total_deaths']:,}")
    col3.metric("Hospitalization Rate", f"{summary['hospitalization_rate'] * 100:.2f}%")
    col4.metric("Mortality Rate", f"{summary['mortality_rate'] * 100:.2f}%")

except Exception as e:
    st.error(f"Could not load summary statistics: {e}")

st.divider()

# -------------------------------------------------------
# Top municipalities bar chart
# -------------------------------------------------------
st.subheader("Top Municipalities by Notifications")

try:
    limit = st.slider("Number of municipalities", min_value=5, max_value=20, value=10)
    top_municipalities = api_client.get_top_municipalities(limit)
    df = pd.DataFrame(top_municipalities).sort_values(
        "total_notifications", ascending=True
    )
    df["municipality_code"] = df["municipality_code"].astype(str)

    fig = px.bar(
        df,
        x="total_notifications",
        y="municipality_code",
        orientation="h",
        labels={
            "total_notifications": "Total Notifications",
            "municipality_code": "Municipality (IBGE code)",
        },
        color="total_notifications",
        color_continuous_scale="Reds",
    )
    fig.update_layout(coloraxis_showscale=False, height=400)
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Could not load municipality data: {e}")

st.divider()

# -------------------------------------------------------
# Confidence interval for daily notifications
# -------------------------------------------------------
st.subheader("Confidence Interval — Daily Notifications")

try:
    confidence = st.select_slider(
        "Confidence level",
        options=[0.80, 0.85, 0.90, 0.95, 0.99],
        value=0.95,
    )
    ci = api_client.get_confidence_interval_daily_cases(confidence)

    c1, c2, c3 = st.columns(3)
    c1.metric("Mean", f"{ci['mean']:,.2f}")
    c2.metric("Lower bound", f"{ci['lower']:,.2f}")
    c3.metric("Upper bound", f"{ci['upper']:,.2f}")
    st.caption(f"Based on {ci['n']:,} days at {int(confidence * 100)}% confidence.")

except Exception as e:
    if "404" in str(e):
        st.info(
            "No data available yet. Run the ETL pipeline first: `docker compose run --rm training python main_workflow.py --states CE`"
        )
    else:
        st.error(f"Could not load confidence interval: {e}")
