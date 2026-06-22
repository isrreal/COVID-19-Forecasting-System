import streamlit as st
import plotly.express as px
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(__file__))
import api_client

st.set_page_config(
    page_title="COVID-19 Forecasting",
    page_icon="🦠",
    layout="wide"
)

st.title("COVID-19 Brazil — Forecasting Dashboard")
st.caption("Real-time forecasting powered by LSTM and PLE models.")

st.divider()

# -------------------------------------------------------
# Summary cards
# -------------------------------------------------------
try:
    summary = api_client.get_summary()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{summary['total_records']:,}")
    col2.metric("Total Confirmed", f"{summary['total_confirmed']:,.0f}")
    col3.metric("Total Deaths", f"{summary['total_deaths']:,.0f}")
    col4.metric("Avg Daily Cases", f"{summary['avg_new_confirmed_per_day']:,.1f}")

except Exception as e:
    st.error(f"Could not load summary statistics: {e}")

st.divider()

# -------------------------------------------------------
# Top cities bar chart
# -------------------------------------------------------
st.subheader("Top Cities by Confirmed Cases")

try:
    limit = st.slider("Number of cities", min_value=5, max_value=20, value=10)
    top_cities = api_client.get_top_cities(limit)
    df = pd.DataFrame(top_cities).sort_values("total_confirmed", ascending=True)

    fig = px.bar(
        df,
        x="total_confirmed",
        y="city",
        orientation="h",
        labels={"total_confirmed": "Total Confirmed Cases", "city": "City"},
        color="total_confirmed",
        color_continuous_scale="Reds",
    )
    fig.update_layout(coloraxis_showscale=False, height=400)
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Could not load city data: {e}")

st.divider()

# -------------------------------------------------------
# Confidence interval for daily cases
# -------------------------------------------------------
st.subheader("Confidence Interval — Daily New Cases")

try:
    confidence = st.select_slider(
        "Confidence level",
        options=[0.80, 0.85, 0.90, 0.95, 0.99],
        value=0.95
    )
    ci = api_client.get_confidence_interval_cases(confidence)

    c1, c2, c3 = st.columns(3)
    c1.metric("Mean", f"{ci['mean']:,.2f}")
    c2.metric("Lower bound", f"{ci['lower']:,.2f}")
    c3.metric("Upper bound", f"{ci['upper']:,.2f}")
    st.caption(f"Based on {ci['n']:,} observations at {int(confidence * 100)}% confidence.")

except Exception as e:
    if "404" in str(e):
        st.info("No data available yet. Run the ETL pipeline first: `docker compose run --rm training python main_workflow.py --states CE`")
    else:
        st.error(f"Could not load confidence interval: {e}")
