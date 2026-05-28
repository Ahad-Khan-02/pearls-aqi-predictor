import sys
import os
import tempfile

# Windows-safe temp directory for Hopsworks
tempfile.tempdir = os.path.join(os.getcwd(), "tmp")

# Create tmp directory if missing
os.makedirs(tempfile.tempdir, exist_ok=True)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import shap

st.set_page_config(
    page_title="...",
    page_icon="...",
    layout="wide",
    initial_sidebar_state="expanded"   # ← this is critical
)

css_path = os.path.join(os.path.dirname(__file__), "..", "assets", "style.css")
with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

import joblib

@st.cache_resource
def load_model():

    model_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "models",
        "best_aqi_model.pkl"
    )

    return joblib.load(model_path)

model = load_model()

from src.utils.fetch_latest_features import get_latest_features
FEATURE_COLUMNS = [
    "temperature", "humidity", "wind_speed", "pressure",
    "pm25", "pm10", "co", "no2", "o3",
    "hour", "day", "month", "day_of_week",
    "previous_aqi", "aqi_lag_3", "aqi_lag_6", "aqi_lag_12",
    "rolling_avg_3", "rolling_avg_6", "aqi_change"
]

raw_data = get_latest_features()

payload = {
    k: raw_data[k]
    for k in FEATURE_COLUMNS
}

st.markdown("""
<div class="page-header">
    <h2>🔬 Explainability</h2>
    <p>SHAP-based feature importance — understand what drives the prediction</p>
</div>
""", unsafe_allow_html=True)

# =========================
# COMPUTE SHAP
# =========================

@st.cache_data(ttl=3600)
def compute_shap(_model, _payload):
    explainer   = shap.TreeExplainer(_model)
    input_df    = pd.DataFrame([_payload])
    shap_vals   = explainer.shap_values(input_df)
    base_val    = float(explainer.expected_value)
    shap_arr    = shap_vals[0] if len(shap_vals.shape) > 1 else shap_vals
    return shap_arr, base_val, list(input_df.columns), input_df.iloc[0].to_dict()

with st.spinner("Computing SHAP values..."):
    shap_arr, base_val, feature_names, feature_vals = compute_shap(model, payload)

importance      = np.abs(shap_arr)
sorted_idx      = np.argsort(importance)[::-1]
top_n           = 10
top_idx         = sorted_idx[:top_n]
top_features    = [feature_names[i] for i in top_idx]
top_shap        = [shap_arr[i] for i in top_idx]
top_importance  = [importance[i] for i in top_idx]
top_vals        = [feature_vals[f] for f in top_features]

# =========================
# SUMMARY STATS
# =========================

predicted_aqi = st.session_state.get("predicted_aqi", base_val)
top_contributor = top_features[0]
top_impact      = top_shap[0]

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f"""
    <div class="stat-block">
        <div class="stat-block-num" style="color:#3b82f6;">{predicted_aqi}</div>
        <div class="stat-block-label">Predicted AQI</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="stat-block">
        <div class="stat-block-num" style="font-size:1.4rem; color:#06b6d4;">{top_contributor}</div>
        <div class="stat-block-label">Top Driver</div>
    </div>""", unsafe_allow_html=True)
with c3:
    impact_clr = "#f44336" if top_impact > 0 else "#00e676"
    sign       = "+" if top_impact > 0 else ""
    st.markdown(f"""
    <div class="stat-block">
        <div class="stat-block-num" style="color:{impact_clr};">{sign}{top_impact:.2f}</div>
        <div class="stat-block-label">Top Feature Impact</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =========================
# SHAP BAR CHART
# =========================

col_bar, col_waterfall = st.columns([1, 1])

with col_bar:
    st.markdown("""
    <div class="info-card-title" style="font-size:0.7rem; letter-spacing:2px; color:#8899aa;
         text-transform:uppercase; margin-bottom:0.5rem;">Feature Importance (SHAP)</div>
    """, unsafe_allow_html=True)

    # horizontal bars sorted ascending
    sorted_bar_idx = np.argsort(top_importance)
    bar_features   = [top_features[i] for i in sorted_bar_idx]
    bar_vals       = [top_importance[i] for i in sorted_bar_idx]
    bar_colors     = ["#3b82f6" if top_shap[i] > 0 else "#06b6d4" for i in sorted_bar_idx]

    fig_bar = go.Figure(go.Bar(
        x=bar_vals,
        y=bar_features,
        orientation="h",
        marker=dict(
            color=bar_colors,
            opacity=0.85,
            line=dict(color="rgba(0,0,0,0)", width=0)
        ),
        hovertemplate="<b>%{y}</b><br>|SHAP|: %{x:.3f}<extra></extra>"
    ))
    fig_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,20,32,0.6)",
        height=360,
        margin=dict(l=0, r=20, t=10, b=30),
        xaxis=dict(
            title="Mean |SHAP value|",
            color="#8899aa", gridcolor="#1e2d45",
            tickfont=dict(size=9, family="Space Mono")
        ),
        yaxis=dict(color="#e8f0fe", tickfont=dict(size=10, family="DM Sans")),
        font=dict(color="#e8f0fe")
    )
    st.plotly_chart(fig_bar, width="stretch")

with col_waterfall:
    st.markdown("""
    <div class="info-card-title" style="font-size:0.7rem; letter-spacing:2px; color:#8899aa;
         text-transform:uppercase; margin-bottom:0.5rem;">SHAP Waterfall (Top 8)</div>
    """, unsafe_allow_html=True)

    n_wf      = 8
    wf_feats  = top_features[:n_wf]
    wf_shap   = top_shap[:n_wf]
    wf_clrs   = ["#ef5350" if v > 0 else "#42a5f5" for v in wf_shap]

    fig_wf = go.Figure(go.Waterfall(
        orientation="h",
        measure=["relative"] * n_wf + ["total"],
        x=wf_shap + [sum(wf_shap)],
        y=wf_feats + ["Total"],
        connector={"line": {"color": "#2a3f5f", "width": 1}},
        decreasing={"marker": {"color": "#42a5f5"}},
        increasing={"marker": {"color": "#ef5350"}},
        totals={"marker": {"color": "#fbbf24"}},
        hovertemplate="<b>%{y}</b><br>SHAP: %{x:.3f}<extra></extra>"
    ))
    fig_wf.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,20,32,0.6)",
        height=360,
        margin=dict(l=0, r=20, t=10, b=30),
        xaxis=dict(
            title="SHAP value",
            color="#8899aa", gridcolor="#1e2d45",
            tickfont=dict(size=9, family="Space Mono")
        ),
        yaxis=dict(color="#e8f0fe", tickfont=dict(size=10, family="DM Sans")),
        showlegend=False,
        font=dict(color="#e8f0fe")
    )
    st.plotly_chart(fig_wf, width="stretch")

# =========================
# FEATURE CONTRIBUTOR TABLE
# =========================

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown("""
<div class="info-card-title" style="font-size:0.7rem; letter-spacing:2px; color:#8899aa;
     text-transform:uppercase; margin-bottom:1rem;">Top Feature Breakdown</div>
""", unsafe_allow_html=True)

st.markdown('<div class="leaderboard">', unsafe_allow_html=True)
rank_labels = ["gold", "silver", "bronze"] + [""] * 7

for i, (feat, sv, fv, imp) in enumerate(zip(top_features, top_shap, top_vals, top_importance)):
    direction = "↑ Raises AQI" if sv > 0 else "↓ Lowers AQI"
    dir_color = "#f44336" if sv > 0 else "#00e676"
    rank_class = rank_labels[i] if i < len(rank_labels) else ""
    st.markdown(f"""
    <div class="leaderboard-row">
        <div class="lb-rank {rank_class}">#{i+1}</div>
        <div class="lb-name">{feat}</div>
        <div style="font-family:'Space Mono',monospace; font-size:0.85rem; color:#8899aa; width:70px;">
            {round(float(fv), 3)}
        </div>
        <div style="font-size:0.8rem; color:{dir_color}; width:120px; font-weight:600;">
            {direction}
        </div>
        <div style="font-family:'Space Mono',monospace; font-size:0.85rem;
                    color:{dir_color}; width:70px; text-align:right;">
            {"+" if sv > 0 else ""}{sv:.3f}
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)