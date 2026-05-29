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
import json

st.set_page_config(
    page_title="...",
    page_icon="...",
    layout="wide",
    initial_sidebar_state="expanded"   # ← this is critical
)

css_path = os.path.join(os.path.dirname(__file__), "..", "assets", "style.css")
with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("""
<div class="page-header">
    <h2>🏆 Model Analytics</h2>
    <p>Performance comparison across all trained models</p>
</div>
""", unsafe_allow_html=True)

# =========================
# MODEL METRICS DATA
# =========================


metrics_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "models",
    "model_metrics.json"
)

if not os.path.exists(metrics_path):
    st.error("Model metrics file not found.")
    st.stop()

with open(metrics_path, "r") as f:
    metrics_json = json.load(f)

best_model = metrics_json["best_model"]

models_data = {}

for model_name, vals in metrics_json.items():

    if model_name == "best_model":
        continue

    models_data[model_name] = {
        "MAE": vals["MAE"],
        "RMSE": vals["RMSE"],
        "R2": vals["R2"],
        "Params": "Dynamic",
        "Status": (
            "Champion"
            if model_name == best_model
            else "Runner-up"
        )
    }



# =========================
# LEADERBOARD
# =========================

st.markdown("""
<div class="info-card-title" style="font-size:0.7rem; letter-spacing:2px; color:#8899aa;
     text-transform:uppercase; margin-bottom:1rem;">Model Leaderboard</div>
""", unsafe_allow_html=True)

sorted_models = sorted(models_data.items(), key=lambda x: x[1]["R2"], reverse=True)

st.markdown('<div class="leaderboard">', unsafe_allow_html=True)
rank_labels = ["gold", "silver", "bronze"]

for i, (name, metrics) in enumerate(sorted_models):
    rank_cls  = rank_labels[i] if i < 3 else ""
    is_best   = name == best_model
    badge_clr = "#fbbf24" if is_best else "#3b82f6"
    badge_bg  = "#fbbf2422" if is_best else "#3b82f622"
    badge_txt = "🏆 Champion" if is_best else "Runner-up"

    st.markdown(f"""
    <div class="leaderboard-row" style="{'background:#1a1f0d;' if is_best else ''}">
        <div class="lb-rank {rank_cls}">#{i+1}</div>
        <div class="lb-name">{name}</div>
        <div style="font-size:0.8rem; color:#8899aa; width:90px;">
            {metrics['Params']}
        </div>
        <div style="font-family:'Space Mono',monospace; font-size:0.85rem; color:#00e676; width:80px;">
            R² {metrics['R2']:.3f}
        </div>
        <div style="font-family:'Space Mono',monospace; font-size:0.85rem; color:#ef9a9a; width:80px;">
            MAE {metrics['MAE']}
        </div>
        <div style="font-family:'Space Mono',monospace; font-size:0.85rem; color:#8899aa; width:80px;">
            RMSE {metrics['RMSE']}
        </div>
        <div class="lb-badge" style="background:{badge_bg}; color:{badge_clr};
                                      border:1px solid {badge_clr}44;">
            {badge_txt}
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =========================
# METRIC COMPARISON CHARTS
# =========================

col_bar, col_radar = st.columns([1, 1])

with col_bar:
    st.markdown("""
    <div class="info-card-title" style="font-size:0.7rem; letter-spacing:2px; color:#8899aa;
         text-transform:uppercase; margin-bottom:0.5rem;">MAE & RMSE Comparison</div>
    """, unsafe_allow_html=True)

    model_names = [m for m, _ in sorted_models]
    maes        = [models_data[m]["MAE"]  for m in model_names]
    rmses       = [models_data[m]["RMSE"] for m in model_names]
    # 1. Map base colors
    clrs        = ["#fbbf24" if m == best_model else "#3b82f6" for m in model_names]
    
    # 2. Convert standard hex strings safely into rgba strings for alpha control
    # #fbbf24 -> rgba(251, 191, 36, 0.8)  and  #3b82f6 -> rgba(59, 130, 246, 0.8)
    clrs_with_alpha = [
        "rgba(251, 191, 36, 0.8)" if m == best_model else "rgba(59, 130, 246, 0.8)" 
        for m in model_names
    ]

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        name="MAE",
        x=model_names,
        y=maes,
        marker_color=clrs_with_alpha,  # <-- Use the valid rgba array here
        text=[f"{v}" for v in maes],
        textposition="outside",
        textfont=dict(color="#e8f0fe", size=11, family="Space Mono")
    ))

    fig_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,20,32,0.6)",
        height=320,
        barmode="group",
        margin=dict(l=10, r=10, t=20, b=30),
        legend=dict(font=dict(color="#8899aa")),
        xaxis=dict(color="#8899aa", tickfont=dict(family="DM Sans", size=11)),
        yaxis=dict(
            color="#8899aa", gridcolor="#1e2d45",
            tickfont=dict(family="Space Mono", size=9),
            title="Error (lower is better)"
        ),
        font=dict(color="#e8f0fe")
    )
    st.plotly_chart(fig_bar, width="stretch")

with col_radar:
    st.markdown("""
    <div class="info-card-title" style="font-size:0.7rem; letter-spacing:2px; color:#8899aa;
         text-transform:uppercase; margin-bottom:0.5rem;">Performance Radar</div>
    """, unsafe_allow_html=True)

    # Normalize metrics to 0-1 (higher = better)
    def normalize(val, lo, hi, invert=False):
        n = (val - lo) / (hi - lo)
        return 1 - n if invert else n

    categories = ["R² Score", "Low MAE", "Low RMSE", "Stability", "Generalization"]

    radar_data = {}

    for model_name, vals in models_data.items():

        radar_data[model_name] = [
            vals["R2"],
            1 / (1 + vals["MAE"]),
            1 / (1 + vals["RMSE"])
        ]

    # Define primary colors and matching translucent RGBA fills for the radar plots
    model_colors = {
        "Random Forest": "#fbbf24", 
        "Ridge": "#3b82f6"
    }
    model_fills = {
        "Random Forest": "rgba(251, 191, 36, 0.13)",  # Matches #fbbf24 with low opacity
        "Ridge": "rgba(59, 130, 246, 0.13)"          # Matches #3b82f6 with low opacity
    }

    fig_radar = go.Figure()
    for mname, vals in radar_data.items():
        vals_closed = vals + [vals[0]]
        cats_closed = categories + [categories[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals_closed,
            theta=cats_closed,
            fill="toself",
            name=mname,
            line=dict(color=model_colors[mname], width=2),
            fillcolor=model_fills[mname],  # <-- Pass the safe, explicit RGBA string here
            hovertemplate="%{theta}: %{r:.2f}<extra>" + mname + "</extra>"
        ))

    fig_radar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        polar=dict(
            bgcolor="rgba(13,20,32,0.6)",
            radialaxis=dict(
                visible=True, range=[0, 1],
                color="#556677",
                gridcolor="#1e2d45",
                tickfont=dict(size=8, color="#556677")
            ),
            angularaxis=dict(
                color="#8899aa",
                tickfont=dict(size=10, family="DM Sans")
            )
        ),
        height=320,
        margin=dict(l=30, r=30, t=20, b=30),
        legend=dict(font=dict(color="#8899aa")),
        font=dict(color="#e8f0fe")
    )
    st.plotly_chart(fig_radar, width="stretch")

# =========================
# R² SCORE CHART
# =========================

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown("""
<div class="info-card-title" style="font-size:0.7rem; letter-spacing:2px; color:#8899aa;
     text-transform:uppercase; margin-bottom:0.5rem;">R² Score Comparison (Higher is Better)</div>
""", unsafe_allow_html=True)

r2_vals  = [models_data[m]["R2"] for m in model_names]
r2_clrs  = ["#fbbf24" if m == best_model else "#3b82f6" for m in model_names]

fig_r2 = go.Figure(go.Bar(
    x=model_names,
    y=r2_vals,
    marker=dict(
        color=r2_clrs,
        line=dict(color="rgba(0,0,0,0)", width=0)
    ),
    text=[f"{v:.4f}" for v in r2_vals],
    textposition="outside",
    textfont=dict(color="#e8f0fe", size=13, family="Space Mono"),
    hovertemplate="<b>%{x}</b><br>R²: %{y:.4f}<extra></extra>"
))

fig_r2.add_hline(
    y=1.0,
    line_dash="dot",
    line_color="green"
)

fig_r2.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(13,20,32,0.6)",
    height=240,
    margin=dict(l=10, r=10, t=20, b=30),
    xaxis=dict(color="#8899aa", tickfont=dict(family="DM Sans", size=12)),
    yaxis=dict(
        range=[0.8, 1.02],
        color="#8899aa", gridcolor="#1e2d45",
        tickfont=dict(family="Space Mono", size=9)
    ),
    font=dict(color="#e8f0fe")
)
st.plotly_chart(fig_r2, width="stretch")
