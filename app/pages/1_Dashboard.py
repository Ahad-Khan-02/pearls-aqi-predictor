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

st.set_page_config(
    page_title="...",
    page_icon="...",
    layout="wide",
    initial_sidebar_state="expanded"   
)

# Load CSS
css_path = os.path.join(os.path.dirname(__file__), "..", "assets", "style.css")
with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Pull from session state (set by streamlit_app.py)
model         = st.session_state.get("model")
raw_data      = st.session_state.get("raw_data", {})
predicted_aqi = st.session_state.get("predicted_aqi", 0)
status        = st.session_state.get("status", "Unknown")
color         = st.session_state.get("color", "#ffffff")
icon          = st.session_state.get("icon", "")



# PAGE HEADER

st.markdown("""
<div class="page-header">
    <h2> Dashboard</h2>
    <p>Real-time air quality overview for Karachi, Pakistan</p>
</div>
""", unsafe_allow_html=True)


# AQI GAUGE + STATUS

col_gauge, col_info = st.columns([1, 1])

with col_gauge:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=predicted_aqi,
        domain={"x": [0, 1], "y": [0, 1]},
        number={"font": {"size": 56, "color": color, "family": "Space Mono"}},
        gauge={
            "axis": {
                "range": [0, 300],
                "tickwidth": 1,
                "tickcolor": "#2a3f5f",
                "tickvals": [50, 100, 150, 200, 300],
                "tickfont": {"color": "#8899aa", "size": 11}
            },
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#0d1420",
            "borderwidth": 0,
            "steps": [
                {"range": [0,   50],  "color": "#00251a"},
                {"range": [50,  100], "color": "#1a1600"},
                {"range": [100, 150], "color": "#1a0d00"},
                {"range": [150, 200], "color": "#1a0000"},
                {"range": [200, 300], "color": "#12001a"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.8,
                "value": predicted_aqi
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=280,
        margin=dict(l=30, r=30, t=20, b=10),
        font={"color": "#e8f0fe"}
    )
    st.plotly_chart(fig, width="stretch")

with col_info:
    aqi_levels = [
        ("Good",                     "0–50",   "#00e676"),
        ("Moderate",                 "51–100",  "#ffeb3b"),
        ("Unhealthy for Sensitive",  "101–150", "#ff9800"),
        ("Unhealthy",                "151–200", "#f44336"),
        ("Very Unhealthy",           "201–300", "#9c27b0"),
    ]

    st.markdown("""
    <div class="info-card">
        <div class="info-card-title">AQI Scale Reference</div>
    """, unsafe_allow_html=True)

    for level, rng, clr in aqi_levels:
        active = "opacity:1;" if status in level or level in status else "opacity:0.45;"
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:10px; padding:5px 0; {active}">
            <div style="width:10px; height:10px; border-radius:50%; background:{clr}; flex-shrink:0;"></div>
            <span style="color:#e8f0fe; font-size:0.88rem; flex:1;">{level}</span>
            <span style="color:#8899aa; font-size:0.8rem; font-family:'Space Mono',monospace;">{rng}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Health recommendation
    rec_map = {
        "Good":                   "Air quality is satisfactory. Enjoy outdoor activities freely.",
        "Moderate":               "Unusually sensitive people should consider limiting prolonged outdoor exertion.",
        "Unhealthy for Sensitive":"People with heart/lung disease, elderly and children should reduce prolonged exertion.",
        "Unhealthy":              "Everyone may begin to experience health effects. Limit prolonged outdoor exertion.",
        "Very Unhealthy":         "Health alert — everyone may experience serious effects. Avoid outdoor activity.",
        "Hazardous":              "Emergency conditions. Everyone should avoid all outdoor activity.",
    }
    rec = rec_map.get(status, rec_map["Moderate"])

    st.markdown(f"""
    <div class="health-rec" style="margin-top:1rem;">
        <div class="health-rec-title">Health Recommendation</div>
        <div class="health-rec-text">{rec}</div>
    </div>
    """, unsafe_allow_html=True)



# POLLUTANT CARDS

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown("""
<div class="info-card-title" style="font-size:0.7rem; letter-spacing:2px; color:#8899aa; text-transform:uppercase; margin-bottom:1rem;">
    Pollutant Levels
</div>
""", unsafe_allow_html=True)

pollutants = [
    ("PM2.5", raw_data.get("pm25", 0),  75,  "#ef5350", "µg/m³"),
    ("PM10",  raw_data.get("pm10", 0),  150, "#ff9800", "µg/m³"),
    ("NO₂",   raw_data.get("no2", 0),   100, "#ab47bc", "µg/m³"),
    ("CO",    raw_data.get("co", 0),    400, "#26c6da", "µg/m³"),
    ("O₃",    raw_data.get("o3", 0),    120, "#66bb6a", "µg/m³"),
]

cols = st.columns(5)
for col, (name, val, limit, clr, unit) in zip(cols, pollutants):
    pct = min(float(val) / limit * 100, 100)
    with col:
        st.markdown(f"""
        <div class="info-card" style="text-align:center;">
            <div class="info-card-title">{name}</div>
            <div style="font-family:'Space Mono',monospace; font-size:1.6rem; color:{clr}; margin:0.4rem 0;">
                {round(float(val), 1)}
            </div>
            <div style="font-size:0.7rem; color:#556677; margin-bottom:0.8rem;">{unit}</div>
            <div style="background:#1e2d45; height:5px; border-radius:3px; overflow:hidden;">
                <div style="width:{pct}%; height:100%; background:{clr}; border-radius:3px; 
                            box-shadow:0 0 8px {clr}66;"></div>
            </div>
            <div style="font-size:0.65rem; color:#556677; margin-top:0.4rem;">
                {pct:.0f}% of limit
            </div>
        </div>
        """, unsafe_allow_html=True)


# WEATHER SNAPSHOT

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown("""
<div class="info-card-title" style="font-size:0.7rem; letter-spacing:2px; color:#8899aa; text-transform:uppercase; margin-bottom:1rem;">
    Weather Snapshot — Karachi
</div>
""", unsafe_allow_html=True)

w1, w2, w3, w4 = st.columns(4)
weather_items = [
    (w1, "🌡️ Temperature",  f"{raw_data.get('temperature', 'N/A')}°C"),
    (w2, "💧 Humidity",     f"{raw_data.get('humidity', 'N/A')}%"),
    (w3, "💨 Wind Speed",   f"{raw_data.get('wind_speed', 'N/A')} km/h"),
    (w4, "🔵 Pressure",     f"{raw_data.get('pressure', 'N/A')} hPa"),
]

for col, label, val in weather_items:
    with col:
        st.markdown(f"""
        <div class="stat-block">
            <div style="font-size:0.8rem; color:#8899aa; margin-bottom:0.5rem;">{label}</div>
            <div class="stat-block-num" style="font-size:1.8rem;">{val}</div>
        </div>
        """, unsafe_allow_html=True)