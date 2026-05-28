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
import joblib


st.set_page_config(
    page_title="...",
    page_icon="...",
    layout="wide",
    initial_sidebar_state="expanded"   # ← this is critical
)

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

css_path = os.path.join(os.path.dirname(__file__), "..", "assets", "style.css")
with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


from src.utils.fetch_latest_features import get_latest_features
FEATURE_COLUMNS = [
    "temperature", "humidity", "wind_speed", "pressure",
    "pm25", "pm10", "co", "no2", "o3",
    "hour", "day", "month", "day_of_week","is_weekend", "is_rush_hour",
    "previous_aqi", "aqi_lag_3", "aqi_lag_6", "aqi_lag_12",
    "rolling_avg_3", "rolling_avg_6","rolling_avg_24", "aqi_change","aqi_trend", "pollution_index"
]

raw_data = get_latest_features()

payload = {
    k: raw_data[k]
    for k in FEATURE_COLUMNS
}


predicted_aqi = st.session_state.get("predicted_aqi", 0)
color         = st.session_state.get("color", "#3b82f6")

st.markdown("""
<div class="page-header">
    <h2>📈 72-Hour Forecast</h2>
    <p>Autoregressive AQI projection for the next 3 days</p>
</div>
""", unsafe_allow_html=True)

# =========================
# GENERATE FORECAST
# =========================

@st.cache_data(ttl=3600)
def generate_forecast(_model, _payload):
    forecast_values = []
    current = dict(_payload)
    for i in range(72):
        current.pop("timestamp", None)
        df = pd.DataFrame([current])
        next_aqi = round(float(_model.predict(df)[0]), 2)
        forecast_values.append(next_aqi)
        prev = current["previous_aqi"]
        current["aqi_change"]    = next_aqi - prev
        current["previous_aqi"]  = next_aqi
        current["aqi_lag_3"]     = next_aqi
        current["aqi_lag_6"]     = next_aqi
        current["aqi_lag_12"]    = next_aqi
        current["rolling_avg_3"] = (current["rolling_avg_3"] + next_aqi) / 2
        current["rolling_avg_6"] = (current["rolling_avg_6"] + next_aqi) / 2
        current["hour"]          = (current["hour"] + 1) % 24
    return forecast_values

with st.spinner("Generating 72-hour forecast..."):
    forecast_values = generate_forecast(model, payload)

hours   = list(range(72))
fdf     = pd.DataFrame({"Hour": hours, "AQI": forecast_values})
peak    = max(forecast_values)
best    = min(forecast_values)
avg_72  = round(np.mean(forecast_values), 1)
peak_h  = forecast_values.index(peak)
best_h  = forecast_values.index(best)

# =========================
# SUMMARY STATS
# =========================

c1, c2, c3, c4 = st.columns(4)
stats = [
    (c1, "Current AQI",   f"{predicted_aqi}", color),
    (c2, "72h Average",   f"{avg_72}",         "#3b82f6"),
    (c3, f"Peak (H+{peak_h})", f"{peak}",      "#f44336"),
    (c4, f"Best (H+{best_h})", f"{best}",      "#00e676"),
]

for col, label, val, clr in stats:
    with col:
        st.markdown(f"""
        <div class="stat-block" style="border-color:{clr}33;">
            <div class="stat-block-num" style="color:{clr};">{val}</div>
            <div class="stat-block-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =========================
# MAIN FORECAST CHART
# =========================

def aqi_color_for(v):
    if v <= 50:  return "#00e676"
    if v <= 100: return "#ffeb3b"
    if v <= 150: return "#ff9800"
    if v <= 200: return "#f44336"
    return "#9c27b0"

fig = go.Figure()

# Threshold zones
threshold_bands = [
    (0,   50,  "#00e676", "Good"),
    (50,  100, "#ffeb3b", "Moderate"),
    (100, 150, "#ff9800", "Unhealthy (Sensitive)"),
    (150, 200, "#f44336", "Unhealthy"),
    (200, 300, "#9c27b0", "Very Unhealthy"),
]
for lo, hi, clr, name in threshold_bands:
    fig.add_hrect(y0=lo, y1=hi, fillcolor=clr, opacity=0.04,
                  layer="below", line_width=0, annotation_text=name,
                  annotation_position="right", annotation_font_size=9,
                  annotation_font_color=clr)

# Forecast line
fig.add_trace(go.Scatter(
    x=hours, y=forecast_values,
    mode="lines",
    name="AQI Forecast",
    line=dict(color="#3b82f6", width=2.5, shape="spline", smoothing=0.6),
    fill="tozeroy",
    fillcolor="rgba(59,130,246,0.06)",
    hovertemplate="Hour +%{x}<br>AQI: <b>%{y:.1f}</b><extra></extra>"
))

# Peak marker
fig.add_trace(go.Scatter(
    x=[peak_h], y=[peak],
    mode="markers+text",
    name="Peak",
    marker=dict(color="#f44336", size=10, symbol="triangle-up",
                line=dict(color="#fff", width=1.5)),
    text=[f"Peak {peak}"],
    textposition="top center",
    textfont=dict(color="#f44336", size=10, family="Space Mono"),
    hovertemplate="Peak AQI: <b>%{y}</b><extra></extra>"
))

# Best marker
fig.add_trace(go.Scatter(
    x=[best_h], y=[best],
    mode="markers+text",
    name="Best",
    marker=dict(color="#00e676", size=10, symbol="triangle-down",
                line=dict(color="#fff", width=1.5)),
    text=[f"Best {best}"],
    textposition="bottom center",
    textfont=dict(color="#00e676", size=10, family="Space Mono"),
    hovertemplate="Best AQI: <b>%{y}</b><extra></extra>"
))

# Day separators
for d in [24, 48]:
    fig.add_vline(x=d, line=dict(color="#2a3f5f", width=1, dash="dot"))
    fig.add_annotation(x=d, y=max(forecast_values)*1.05,
                       text=f"Day {d//24 + 1}", showarrow=False,
                       font=dict(color="#556677", size=10, family="Space Mono"))

fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(13,20,32,0.6)",
    height=380,
    margin=dict(l=10, r=80, t=20, b=40),
    xaxis=dict(
        title="Hours from Now",
        color="#8899aa",
        gridcolor="#1e2d45",
        tickfont=dict(family="Space Mono", size=10),
        showgrid=True,
        zeroline=False
    ),
    yaxis=dict(
        title="AQI",
        color="#8899aa",
        gridcolor="#1e2d45",
        tickfont=dict(family="Space Mono", size=10),
        range=[0, max(max(forecast_values) * 1.15, 60)],
        zeroline=False
    ),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
        font=dict(color="#8899aa", size=10)
    ),
    hovermode="x unified",
    font=dict(color="#e8f0fe")
)

st.plotly_chart(fig, width="stretch")

# =========================
# HOURLY BREAKDOWN TABLE
# =========================

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown("""
<div class="info-card-title" style="font-size:0.7rem; letter-spacing:2px; color:#8899aa; text-transform:uppercase; margin-bottom:1rem;">
    Hourly Breakdown — Next 24 Hours
</div>
""", unsafe_allow_html=True)

cols = st.columns(8)
for i, col in enumerate(cols):
    aqi_val = forecast_values[i * 3]
    clr = aqi_color_for(aqi_val)
    with col:
        st.markdown(f"""
        <div style="text-align:center; padding:0.8rem 0.3rem; background:#111827;
                    border:1px solid #1e2d45; border-radius:10px;">
            <div style="font-size:0.7rem; color:#556677; margin-bottom:0.4rem;">
                H+{i*3}
            </div>
            <div style="font-family:'Space Mono',monospace; font-size:1.1rem;
                        color:{clr}; font-weight:700;">
                {aqi_val}
            </div>
            <div style="width:8px; height:8px; border-radius:50%; background:{clr};
                        margin:0.4rem auto 0; box-shadow:0 0 6px {clr};"></div>
        </div>
        """, unsafe_allow_html=True)