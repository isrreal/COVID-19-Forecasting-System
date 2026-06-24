import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import api_client

st.set_page_config(page_title="Municipality Rankings", page_icon="🏙️", layout="wide")

st.title("Municipality Rankings")
st.caption("Mortality rates and notification counts by municipality (IBGE code).")

st.divider()

# -------------------------------------------------------
# Most deadly municipalities
# -------------------------------------------------------
st.subheader("Most Deadly Municipalities")
st.caption("Ranked by mortality rate (deaths / total notifications).")

try:
    limit_deadly = st.slider(
        "Number of municipalities", min_value=5, max_value=20, value=10, key="deadly"
    )
    deadly = api_client.get_most_deadly_municipalities(limit_deadly)
    df_deadly = pd.DataFrame(deadly).sort_values("mortality_rate", ascending=False)
    df_deadly["mortality_rate"] = (df_deadly["mortality_rate"] * 100).round(2)
    df_deadly["municipality_code"] = df_deadly["municipality_code"].astype(str)
    df_deadly = df_deadly.rename(
        columns={
            "municipality_code": "Municipality (IBGE)",
            "state_code": "State (IBGE)",
            "mortality_rate": "Mortality Rate (%)",
            "total_deaths": "Total Deaths",
            "total_notifications": "Total Notifications",
        }
    )

    fig = px.bar(
        df_deadly,
        x="Mortality Rate (%)",
        y="Municipality (IBGE)",
        orientation="h",
        color="Mortality Rate (%)",
        color_continuous_scale="Reds",
        text="State (IBGE)",
    )
    fig.update_layout(coloraxis_showscale=False, height=400)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df_deadly, use_container_width=True, hide_index=True)

except Exception as e:
    st.error(f"Could not load data: {e}")

st.divider()

# -------------------------------------------------------
# Least affected municipalities
# -------------------------------------------------------
st.subheader("Least Affected Municipalities")
st.caption("Ranked by lowest mortality rate among municipalities with notifications.")

try:
    limit_least = st.slider(
        "Number of municipalities", min_value=5, max_value=20, value=10, key="least"
    )
    least = api_client.get_least_affected_municipalities(limit_least)
    df_least = pd.DataFrame(least)
    df_least["mortality_rate"] = (df_least["mortality_rate"] * 100).round(2)
    df_least["municipality_code"] = df_least["municipality_code"].astype(str)
    df_least = df_least.rename(
        columns={
            "municipality_code": "Municipality (IBGE)",
            "state_code": "State (IBGE)",
            "mortality_rate": "Mortality Rate (%)",
            "total_deaths": "Total Deaths",
            "total_notifications": "Total Notifications",
        }
    )
    st.dataframe(df_least, use_container_width=True, hide_index=True)

except Exception as e:
    st.error(f"Could not load data: {e}")
