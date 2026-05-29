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
import plotly.express as px
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="...",
    page_icon="...",
    layout="wide",
    initial_sidebar_state="expanded"   # ← this is critical
)

css_path = os.path.join(os.path.dirname(__file__), "..", "assets", "style.css")
with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

raw_data = st.session_state.get("raw_data", {})

st.markdown("""
<div class="page-header">
    <h2>🌐 Air Quality Insights</h2>
    <p>Analytical deep-dive into Karachi's air quality patterns</p>
</div>
""", unsafe_allow_html=True)

# =========================
# LOAD FEATURE DATA
# =========================

@st.cache_data(ttl=3600)
def load_feature_data():
    path = os.path.join(
        os.path.dirname(__file__), "../..", "data", "processed", "featured_aqi_data.csv"
    )
    if os.path.exists(path):
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    return None

df = load_feature_data()

if df is None:
    st.error(
        "Historical feature dataset not found."
    )
    st.stop()

# =========================
# KPI ROW
# =========================

c1, c2, c3, c4 = st.columns(4)
kpis = [
    (c1, "Avg AQI",     f"{df['aqi'].mean():.1f}",  "#3b82f6"),
    (c2, "Max AQI",     f"{df['aqi'].max():.0f}",   "#f44336"),
    (c3, "Min AQI",     f"{df['aqi'].min():.0f}",   "#00e676"),
    (c4, "Std Dev",     f"{df['aqi'].std():.1f}",   "#fbbf24"),
]
for col, label, val, clr in kpis:
    with col:
        st.markdown(f"""
        <div class="stat-block">
            <div class="stat-block-num" style="color:{clr};">{val}</div>
            <div class="stat-block-label">{label}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =========================
# ROW 1: AQI DISTRIBUTION + HOURLY PATTERN
# =========================

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    <div class="info-card-title" style="font-size:0.7rem; letter-spacing:2px; color:#8899aa;
         text-transform:uppercase; margin-bottom:0.5rem;">AQI Distribution</div>
    """, unsafe_allow_html=True)

    fig_hist = go.Figure(go.Histogram(
        x=df["aqi"],
        nbinsx=40,
        marker=dict(
            color="#3b82f6",
            opacity=0.8,
            line=dict(color="rgba(0,0,0,0)", width=0)
        ),
        hovertemplate="AQI: %{x:.0f}<br>Count: %{y}<extra></extra>"
    ))
    # Add AQI threshold lines
    for val, clr, lbl in [(50, "#00e676", "Good"), (100, "#ffeb3b", "Moderate"),
                           (150, "#ff9800", "Sensitive"), (200, "#f44336", "Unhealthy")]:
        fig_hist.add_vline(x=val, line=dict(color=clr, width=1, dash="dot"),
                           annotation_text=lbl, annotation_font_size=9,
                           annotation_font_color=clr)

    fig_hist.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,20,32,0.6)",
        height=280,
        margin=dict(l=10, r=10, t=10, b=30),
        xaxis=dict(title="AQI", color="#8899aa", gridcolor="#1e2d45",
                   tickfont=dict(family="Space Mono", size=9)),
        yaxis=dict(title="Count", color="#8899aa", gridcolor="#1e2d45",
                   tickfont=dict(family="Space Mono", size=9)),
        font=dict(color="#e8f0fe")
    )
    st.plotly_chart(fig_hist, width="stretch")

with col2:
    st.markdown("""
    <div class="info-card-title" style="font-size:0.7rem; letter-spacing:2px; color:#8899aa;
         text-transform:uppercase; margin-bottom:0.5rem;">Average AQI by Hour of Day</div>
    """, unsafe_allow_html=True)

    hourly = df.groupby("hour")["aqi"].mean().reset_index()

    fig_hour = go.Figure(go.Scatter(
        x=hourly["hour"],
        y=hourly["aqi"],
        mode="lines+markers",
        line=dict(color="#06b6d4", width=2.5, shape="spline", smoothing=0.8),
        marker=dict(color="#06b6d4", size=5),
        fill="tozeroy",
        fillcolor="rgba(6,182,212,0.08)",
        hovertemplate="Hour %{x}:00<br>Avg AQI: <b>%{y:.1f}</b><extra></extra>"
    ))
    fig_hour.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,20,32,0.6)",
        height=280,
        margin=dict(l=10, r=10, t=10, b=30),
        xaxis=dict(title="Hour", color="#8899aa", gridcolor="#1e2d45",
                   tickvals=list(range(0, 24, 4)), tickfont=dict(family="Space Mono", size=9)),
        yaxis=dict(title="Avg AQI", color="#8899aa", gridcolor="#1e2d45",
                   tickfont=dict(family="Space Mono", size=9)),
        font=dict(color="#e8f0fe")
    )
    st.plotly_chart(fig_hour, width="stretch")

# =========================
# ROW 2: WEEKDAY + MONTHLY
# =========================

col3, col4 = st.columns([1, 1])

with col3:
    st.markdown("""
    <div class="info-card-title" style="font-size:0.7rem; letter-spacing:2px; color:#8899aa;
         text-transform:uppercase; margin-bottom:0.5rem;">Weekday vs Weekend AQI</div>
    """, unsafe_allow_html=True)

    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    daily = df.groupby("day_of_week")["aqi"].mean().reset_index()
    daily["day_name"] = daily["day_of_week"].apply(lambda x: days[x] if x < 7 else "?")
    day_clrs = ["#3b82f6" if i < 5 else "#fbbf24" for i in range(len(daily))]

    fig_day = go.Figure(go.Bar(
        x=daily["day_name"],
        y=daily["aqi"],
        marker=dict(color=day_clrs, opacity=0.85),
        hovertemplate="%{x}<br>Avg AQI: <b>%{y:.1f}</b><extra></extra>"
    ))
    fig_day.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,20,32,0.6)",
        height=260,
        margin=dict(l=10, r=10, t=10, b=30),
        xaxis=dict(color="#8899aa", tickfont=dict(family="DM Sans", size=10)),
        yaxis=dict(color="#8899aa", gridcolor="#1e2d45",
                   tickfont=dict(family="Space Mono", size=9)),
        annotations=[dict(
            x=5.5, y=daily["aqi"].max() * 1.05,
            text="Weekend", showarrow=False,
            font=dict(color="#fbbf24", size=10)
        )],
        font=dict(color="#e8f0fe")
    )
    st.plotly_chart(fig_day, width="stretch")

with col4:
    st.markdown("""
    <div class="info-card-title" style="font-size:0.7rem; letter-spacing:2px; color:#8899aa;
         text-transform:uppercase; margin-bottom:0.5rem;">Monthly AQI Trend</div>
    """, unsafe_allow_html=True)

    months    = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    monthly   = df.groupby("month")["aqi"].mean().reset_index()
    mon_names = monthly["month"].apply(lambda x: months[x-1] if 1 <= x <= 12 else str(x))

    fig_mon = go.Figure(go.Scatter(
        x=mon_names,
        y=monthly["aqi"],
        mode="lines+markers",
        line=dict(color="#a78bfa", width=2.5, shape="spline", smoothing=0.6),
        marker=dict(color="#a78bfa", size=7),
        fill="tozeroy",
        fillcolor="rgba(167,139,250,0.08)",
        hovertemplate="%{x}<br>Avg AQI: <b>%{y:.1f}</b><extra></extra>"
    ))
    fig_mon.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,20,32,0.6)",
        height=260,
        margin=dict(l=10, r=10, t=10, b=30),
        xaxis=dict(color="#8899aa", tickfont=dict(family="DM Sans", size=9)),
        yaxis=dict(color="#8899aa", gridcolor="#1e2d45",
                   tickfont=dict(family="Space Mono", size=9)),
        font=dict(color="#e8f0fe")
    )
    st.plotly_chart(fig_mon, width="stretch")

# =========================
# CORRELATION HEATMAP
# =========================

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown("""
<div class="info-card-title" style="font-size:0.7rem; letter-spacing:2px; color:#8899aa;
     text-transform:uppercase; margin-bottom:0.5rem;">Pollutant Correlation Matrix</div>
""", unsafe_allow_html=True)

corr_cols = ["aqi", "pm25", "pm10", "no2", "o3", "co", "temperature", "humidity", "wind_speed"]
corr_cols = [c for c in corr_cols if c in df.columns]
corr_matrix = df[corr_cols].corr()

fig_heat = go.Figure(go.Heatmap(
    z=corr_matrix.values,
    x=corr_cols,
    y=corr_cols,
    colorscale=[
        [0.0, "#1565c0"],
        [0.5, "#0d1420"],
        [1.0, "#b71c1c"]
    ],
    zmid=0,
    text=np.round(corr_matrix.values, 2),
    texttemplate="%{text}",
    textfont=dict(size=9, color="#e8f0fe"),
    hovertemplate="%{x} × %{y}<br>r = %{z:.3f}<extra></extra>"
))
fig_heat.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(13,20,32,0.6)",
    height=380,
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis=dict(color="#8899aa", tickfont=dict(family="DM Sans", size=10)),
    yaxis=dict(color="#8899aa", tickfont=dict(family="DM Sans", size=10)),
    font=dict(color="#e8f0fe")
)
st.plotly_chart(fig_heat, width="stretch")
