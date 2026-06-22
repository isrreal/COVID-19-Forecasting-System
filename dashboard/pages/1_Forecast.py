import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import api_client

st.set_page_config(page_title="State Forecast", page_icon="📈", layout="wide")

st.title("State Forecast")
st.caption("Multi-step COVID-19 case forecasting by Brazilian state.")

STATES = [
    "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO",
    "MA", "MG", "MS", "MT", "PA", "PB", "PE", "PI", "PR",
    "RJ", "RN", "RO", "RR", "RS", "SC", "SE", "SP", "TO"
]

col1, col2, col3 = st.columns([2, 2, 3])
state_code = col1.selectbox("State", STATES, index=STATES.index("CE"))
days = col2.slider("Days to forecast", min_value=1, max_value=30, value=7)
show_ci = col3.checkbox("Show confidence interval", value=True)

confidence = 0.95
if show_ci:
    confidence = col3.select_slider(
        "Confidence level",
        options=[0.80, 0.85, 0.90, 0.95, 0.99],
        value=0.95
    )

st.divider()

if st.button("Generate Forecast", type="primary"):
    with st.spinner("Loading model and generating forecast..."):
        try:
            if show_ci:
                data = api_client.get_forecast_confidence(state_code, days, confidence)
                items = data["forecast_with_confidence"]
                df = pd.DataFrame(items)

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df["date"], y=df["upper_bound"],
                    mode="lines", line=dict(width=0),
                    showlegend=False, name="Upper bound"
                ))
                fig.add_trace(go.Scatter(
                    x=df["date"], y=df["lower_bound"],
                    mode="lines", line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(99, 110, 250, 0.15)",
                    showlegend=True, name=f"{int(confidence * 100)}% CI"
                ))
                fig.add_trace(go.Scatter(
                    x=df["date"], y=df["predicted_mean"],
                    mode="lines+markers",
                    line=dict(color="#636EFA", width=2),
                    name="Forecast"
                ))
            else:
                data = api_client.get_forecast_state(state_code, days)
                items = data["forecast"]
                df = pd.DataFrame(items)

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df["date"], y=df["predicted_value"],
                    mode="lines+markers",
                    line=dict(color="#636EFA", width=2),
                    name="Forecast"
                ))

            fig.update_layout(
                title=f"Forecast — {state_code} ({days} days)",
                xaxis_title="Date",
                yaxis_title="Predicted New Cases",
                hovermode="x unified",
                height=450
            )
            st.plotly_chart(fig, use_container_width=True)

            st.caption(f"Model run ID: `{data['model_run_id']}`")
            st.dataframe(df, use_container_width=True)

        except Exception as e:
            st.error(f"Could not generate forecast: {e}")
