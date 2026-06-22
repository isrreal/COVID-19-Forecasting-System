import streamlit as st
import plotly.express as px
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import api_client

st.set_page_config(page_title="City Rankings", page_icon="🏙️", layout="wide")

st.title("City Rankings")
st.caption("Mortality rates and confirmed case counts by city.")

st.divider()

# -------------------------------------------------------
# Most deadly cities
# -------------------------------------------------------
st.subheader("Most Deadly Cities")
st.caption("Ranked by mortality rate (deaths / confirmed cases).")

try:
    limit_deadly = st.slider("Number of cities", min_value=5, max_value=20, value=10, key="deadly")
    deadly = api_client.get_most_deadly_cities(limit_deadly)
    df_deadly = pd.DataFrame(deadly).sort_values("mortality_rate", ascending=False)
    df_deadly["mortality_rate"] = (df_deadly["mortality_rate"] * 100).round(2)
    df_deadly = df_deadly.rename(columns={
        "city": "City", "state": "State",
        "mortality_rate": "Mortality Rate (%)",
        "total_deaths": "Total Deaths",
        "total_confirmed": "Total Confirmed"
    })

    fig = px.bar(
        df_deadly,
        x="Mortality Rate (%)", y="City",
        orientation="h",
        color="Mortality Rate (%)",
        color_continuous_scale="Reds",
        text="State"
    )
    fig.update_layout(coloraxis_showscale=False, height=400)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df_deadly, use_container_width=True, hide_index=True)

except Exception as e:
    st.error(f"Could not load data: {e}")

st.divider()

# -------------------------------------------------------
# Least affected cities
# -------------------------------------------------------
st.subheader("Least Affected Cities")
st.caption("Ranked by lowest mortality rate among cities with confirmed cases.")

try:
    limit_least = st.slider("Number of cities", min_value=5, max_value=20, value=10, key="least")
    least = api_client.get_least_affected_cities(limit_least)
    df_least = pd.DataFrame(least)
    df_least["mortality_rate"] = (df_least["mortality_rate"] * 100).round(2)
    df_least = df_least.rename(columns={
        "city": "City", "state": "State",
        "mortality_rate": "Mortality Rate (%)",
        "total_deaths": "Total Deaths",
        "total_confirmed": "Total Confirmed"
    })
    st.dataframe(df_least, use_container_width=True, hide_index=True)

except Exception as e:
    st.error(f"Could not load data: {e}")
