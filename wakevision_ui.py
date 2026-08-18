
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import glob
from datetime import timedelta

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WakeVision Dashboard",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;`
}

.main { background: #0a0c10; }
.block-container { padding: 1.5rem 2rem; max-width: 1600px; }

/* Header */
.wv-header {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
    border: 1px solid #21262d;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
}
.wv-logo {
    font-size: 3rem;
    filter: drop-shadow(0 0 12px rgba(0,240,180,0.5));
}
.wv-title { font-size: 2rem; font-weight: 700; color: #e6edf3; margin: 0; }
.wv-subtitle { font-size: 0.95rem; color: #7d8590; margin: 0; font-weight: 400; }

/* Metric cards */
.metric-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #388bfd; }
.metric-value { font-size: 2.2rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.metric-label { font-size: 0.78rem; color: #7d8590; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.25rem; }

/* Level badge */
.badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    font-family: 'JetBrains Mono', monospace;
}
.badge-alert    { background: #0d2119; color: #3fb950; border: 1px solid #3fb950; }
.badge-mild     { background: #0d2626; color: #39c5cf; border: 1px solid #39c5cf; }
.badge-moderate { background: #0d1b2e; color: #58a6ff; border: 1px solid #58a6ff; }
.badge-severe   { background: #1a1230; color: #a371f7; border: 1px solid #a371f7; }
.badge-critical { background: #2d1117; color: #ff7b72; border: 1px solid #ff7b72; }

/* Section header */
.section-header {
    font-size: 1.05rem;
    font-weight: 600;
    color: #e6edf3;
    border-left: 3px solid #388bfd;
    padding-left: 0.75rem;
    margin: 1.5rem 0 0.75rem;
}

/* Timeline table */
.timeline-row { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #21262d;
}
section[data-testid="stSidebar"] .block-container { padding: 1rem; }
</style>
""", unsafe_allow_html=True)

LEVEL_COLORS = {
    "ALERT":    "#3fb950",
    "MILD":     "#39c5cf",
    "MODERATE": "#58a6ff",
    "SEVERE":   "#a371f7",
    "CRITICAL": "#ff7b72",
}
LEVEL_ORDER = ["ALERT", "MILD", "MODERATE", "SEVERE", "CRITICAL"]

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_telemetry(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["elapsed_sec"] = pd.to_numeric(df["elapsed_sec"], errors="coerce")
    df["attention_score"] = pd.to_numeric(df["attention_score"], errors="coerce")
    df["perclos"] = pd.to_numeric(df["perclos"], errors="coerce")
    df["ear"] = pd.to_numeric(df["ear"], errors="coerce")
    df["blink_rate_pm"] = pd.to_numeric(df["blink_rate_pm"], errors="coerce")
    df["yawn_rate_pm"] = pd.to_numeric(df["yawn_rate_pm"], errors="coerce")
    df["pitch"] = pd.to_numeric(df["pitch"], errors="coerce")
    df["yaw"] = pd.to_numeric(df["yaw"], errors="coerce")
    df["gaze_dev"] = pd.to_numeric(df["gaze_dev"], errors="coerce")
    df["micro_sleep"] = pd.to_numeric(df["micro_sleep"], errors="coerce")
    df.dropna(subset=["elapsed_sec", "attention_score"], inplace=True)
    return df


def format_duration(seconds):
    return str(timedelta(seconds=int(seconds)))


def level_badge(level: str) -> str:
    cls = f"badge-{level.lower()}"
    return f'<span class="badge {cls}">{level}</span>'


def get_telemetry_files():
    patterns = [
        "wakevision_telemetry_*.csv",
        os.path.join(os.path.dirname(__file__), "wakevision_telemetry_*.csv"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    return sorted(set(files), reverse=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Session Files")
    telem_files = get_telemetry_files()

    if not telem_files:
        st.info("No telemetry files found.\nRun `wakevision_enhanced.py` first.")
        uploaded = st.file_uploader("Or upload a CSV", type="csv")
        if uploaded:
            import io
            df_raw = pd.read_csv(uploaded)
            df_raw.to_csv("/tmp/_wv_upload.csv", index=False)
            telem_files = ["/tmp/_wv_upload.csv"]
        else:
            st.stop()

    labels = [os.path.basename(f).replace("wakevision_telemetry_", "").replace(".csv", "")
              for f in telem_files]
    sel_label = st.selectbox("Session", labels)
    sel_path  = telem_files[labels.index(sel_label)]

    st.markdown("---")
    st.markdown("### ⚙️ Display Options")
    smooth_window = st.slider("Smoothing window (sec)", 1, 30, 10)
    show_raw      = st.checkbox("Show raw signal", value=False)

    st.markdown("---")
    st.markdown("### 🎯 Drowsiness Levels")
    for lv in LEVEL_ORDER:
        st.markdown(
            f'<span class="badge badge-{lv.lower()}">{lv}</span>',
            unsafe_allow_html=True)
    st.markdown("""
<small style='color:#7d8590'>
ALERT ≥ 80 · MILD ≥ 62<br>
MODERATE ≥ 45 · SEVERE ≥ 28<br>
CRITICAL < 28
</small>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
try:
    df = load_telemetry(sel_path)
except Exception as e:
    st.error(f"Failed to load telemetry: {e}")
    st.stop()

if df.empty:
    st.warning("Telemetry file is empty.")
    st.stop()

# Rolling smoothed columns
df["att_smooth"] = df["attention_score"].rolling(smooth_window, min_periods=1, center=True).mean()
df["perclos_smooth"] = df["perclos"].rolling(smooth_window, min_periods=1, center=True).mean()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="wv-header">
  <div class="wv-logo">👁️</div>
  <div>
    <div class="wv-title">WakeVision Dashboard</div>
    <div class="wv-subtitle">Session: {sel_label} &nbsp;·&nbsp; {len(df)} data points</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPI row ───────────────────────────────────────────────────────────────────
ride_dur   = df["elapsed_sec"].max()
avg_att    = df["attention_score"].mean()
max_perc   = df["perclos"].max()
total_bl   = int(df["blink_count"].max()) if "blink_count" in df.columns else "—"
total_yn   = int(df["yawn_total"].max()) if "yawn_total" in df.columns else "—"
micro_cnt  = int(df["micro_sleep"].sum()) if "micro_sleep" in df.columns else 0
dom_level  = df["level"].value_counts().idxmax() if "level" in df.columns else "—"

att_color = LEVEL_COLORS.get(
    "ALERT" if avg_att >= 80 else
    "MILD"  if avg_att >= 62 else
    "MODERATE" if avg_att >= 45 else
    "SEVERE" if avg_att >= 28 else "CRITICAL",
    "#58a6ff"
)

c1, c2, c3, c4, c5, c6 = st.columns(6)
kpis = [
    (c1, format_duration(ride_dur), "Ride Duration", "#e6edf3"),
    (c2, f"{avg_att:.0f}", "Avg Attention", att_color),
    (c3, f"{max_perc:.1f}%", "Peak PERCLOS", "#ff7b72" if max_perc > 40 else "#58a6ff"),
    (c4, str(total_bl), "Total Blinks", "#e6edf3"),
    (c5, str(total_yn), "Total Yawns", "#e6edf3"),
    (c6, str(micro_cnt), "Micro-sleeps", "#ff7b72" if micro_cnt > 0 else "#3fb950"),
]
for col, val, lbl, color in kpis:
    with col:
        st.markdown(f"""
<div class="metric-card">
  <div class="metric-value" style="color:{color}">{val}</div>
  <div class="metric-label">{lbl}</div>
</div>""", unsafe_allow_html=True)

# ── Main charts ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Attention & Drowsiness Timeline</div>', unsafe_allow_html=True)

fig_att = go.Figure()

# Coloured background bands per level (fill between thresholds)
band_ranges = [
    (80, 100, "rgba(63,185,80,0.06)", "Alert"),
    (62,  80, "rgba(57,197,207,0.07)", "Mild"),
    (45,  62, "rgba(88,166,255,0.08)", "Moderate"),
    (28,  45, "rgba(163,113,247,0.10)", "Severe"),
    (0,   28, "rgba(255,123,114,0.12)", "Critical"),
]
for lo, hi, color, name in band_ranges:
    fig_att.add_shape(type="rect",
        x0=df["elapsed_sec"].min(), x1=df["elapsed_sec"].max(),
        y0=lo, y1=hi,
        fillcolor=color, line_width=0, layer="below")

# Raw attention
if show_raw:
    fig_att.add_trace(go.Scatter(
        x=df["elapsed_sec"], y=df["attention_score"],
        mode="lines", name="Raw",
        line=dict(color="rgba(88,166,255,0.25)", width=1),
        showlegend=True
    ))

# Smoothed attention
fig_att.add_trace(go.Scatter(
    x=df["elapsed_sec"], y=df["att_smooth"],
    mode="lines", name="Attention score",
    line=dict(color="#58a6ff", width=2.5),
    fill="tozeroy", fillcolor="rgba(88,166,255,0.07)"
))

# Micro-sleep markers
ms = df[df["micro_sleep"] == 1] if "micro_sleep" in df.columns else pd.DataFrame()
if not ms.empty:
    fig_att.add_trace(go.Scatter(
        x=ms["elapsed_sec"], y=ms["attention_score"],
        mode="markers", name="Micro-sleep",
        marker=dict(color="#ff7b72", size=10, symbol="x", line=dict(width=2))
    ))

# Threshold lines
for lv, thr in [("MILD", 62), ("MODERATE", 45), ("SEVERE", 28)]:
    fig_att.add_hline(y=thr, line_dash="dot",
                      line_color=LEVEL_COLORS[lv], opacity=0.45,
                      annotation_text=lv, annotation_position="right")

fig_att.update_layout(
    height=280, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font_color="#c9d1d9", margin=dict(t=10, b=10, l=10, r=80),
    xaxis=dict(title="Elapsed (s)", gridcolor="#21262d", color="#7d8590"),
    yaxis=dict(title="Score", range=[0, 105], gridcolor="#21262d", color="#7d8590"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font_size=11),
)
st.plotly_chart(fig_att, use_container_width=True, config={"displayModeBar": False})

# ── Second row: PERCLOS + EAR ──────────────────────────────────────────────────
st.markdown('<div class="section-header">Biometric Signals</div>', unsafe_allow_html=True)
col_l, col_r = st.columns(2)

with col_l:
    fig_perc = go.Figure()
    if show_raw:
        fig_perc.add_trace(go.Scatter(
            x=df["elapsed_sec"], y=df["perclos"],
            mode="lines", name="PERCLOS raw",
            line=dict(color="rgba(163,113,247,0.25)", width=1)))
    fig_perc.add_trace(go.Scatter(
        x=df["elapsed_sec"], y=df["perclos_smooth"],
        mode="lines", name="PERCLOS",
        line=dict(color="#a371f7", width=2.5),
        fill="tozeroy", fillcolor="rgba(163,113,247,0.07)"))
    fig_perc.add_hline(y=20, line_dash="dot", line_color="#ff7b72", opacity=0.5,
                       annotation_text="Danger 20%", annotation_position="right")
    fig_perc.update_layout(
        title=dict(text="PERCLOS (% eyes closed / 30s)", font_size=13, x=0),
        height=230, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#c9d1d9", margin=dict(t=40, b=10, l=10, r=80),
        xaxis=dict(title="Elapsed (s)", gridcolor="#21262d", color="#7d8590"),
        yaxis=dict(title="%", range=[0, 105], gridcolor="#21262d", color="#7d8590"),
        showlegend=False,
    )
    st.plotly_chart(fig_perc, use_container_width=True, config={"displayModeBar": False})

with col_r:
    fig_ear = go.Figure()
    fig_ear.add_trace(go.Scatter(
        x=df["elapsed_sec"], y=df["ear"],
        mode="lines", name="EAR",
        line=dict(color="#39c5cf", width=1.5)))
    fig_ear.add_hline(y=df["ear"].quantile(0.72) * 0.80 if len(df) > 5 else 0.20,
                      line_dash="dot", line_color="#ff7b72", opacity=0.5,
                      annotation_text="Threshold", annotation_position="right")
    fig_ear.update_layout(
        title=dict(text="Eye Aspect Ratio (EAR)", font_size=13, x=0),
        height=230, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#c9d1d9", margin=dict(t=40, b=10, l=10, r=80),
        xaxis=dict(title="Elapsed (s)", gridcolor="#21262d", color="#7d8590"),
        yaxis=dict(title="EAR", gridcolor="#21262d", color="#7d8590"),
        showlegend=False,
    )
    st.plotly_chart(fig_ear, use_container_width=True, config={"displayModeBar": False})

# ── Head pose + gaze ────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Head Pose & Gaze</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)

with c1:
    fig_pitch = go.Figure()
    fig_pitch.add_trace(go.Scatter(
        x=df["elapsed_sec"], y=df["pitch"],
        mode="lines", name="Pitch",
        line=dict(color="#ffa657", width=1.8)))
    fig_pitch.add_hline(y=-15, line_dash="dot", line_color="#ff7b72", opacity=0.5)
    fig_pitch.update_layout(
        title=dict(text="Pitch (nodding)", font_size=12, x=0),
        height=200, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#c9d1d9", margin=dict(t=35, b=10, l=10, r=20),
        xaxis=dict(gridcolor="#21262d", color="#7d8590"),
        yaxis=dict(title="°", gridcolor="#21262d", color="#7d8590"),
        showlegend=False)
    st.plotly_chart(fig_pitch, use_container_width=True, config={"displayModeBar": False})

with c2:
    fig_yaw = go.Figure()
    fig_yaw.add_trace(go.Scatter(
        x=df["elapsed_sec"], y=df["yaw"],
        mode="lines", name="Yaw",
        line=dict(color="#79c0ff", width=1.8)))
    fig_yaw.add_hline(y=20, line_dash="dot", line_color="#ff7b72", opacity=0.4)
    fig_yaw.add_hline(y=-20, line_dash="dot", line_color="#ff7b72", opacity=0.4)
    fig_yaw.update_layout(
        title=dict(text="Yaw (head turn)", font_size=12, x=0),
        height=200, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#c9d1d9", margin=dict(t=35, b=10, l=10, r=20),
        xaxis=dict(gridcolor="#21262d", color="#7d8590"),
        yaxis=dict(title="°", gridcolor="#21262d", color="#7d8590"),
        showlegend=False)
    st.plotly_chart(fig_yaw, use_container_width=True, config={"displayModeBar": False})

with c3:
    fig_gaze = go.Figure()
    fig_gaze.add_trace(go.Scatter(
        x=df["elapsed_sec"], y=df["gaze_dev"],
        mode="lines", name="Gaze dev",
        line=dict(color="#d2a8ff", width=1.8),
        fill="tozeroy", fillcolor="rgba(210,168,255,0.07)"))
    fig_gaze.update_layout(
        title=dict(text="Gaze Deviation", font_size=12, x=0),
        height=200, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#c9d1d9", margin=dict(t=35, b=10, l=10, r=20),
        xaxis=dict(gridcolor="#21262d", color="#7d8590"),
        yaxis=dict(title="0–1", range=[0, 1], gridcolor="#21262d", color="#7d8590"),
        showlegend=False)
    st.plotly_chart(fig_gaze, use_container_width=True, config={"displayModeBar": False})

# ── Level distribution ─────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Drowsiness Level Distribution</div>', unsafe_allow_html=True)

if "level" in df.columns:
    col_pie, col_bar = st.columns([1, 2])

    with col_pie:
        counts = df["level"].value_counts().reindex(LEVEL_ORDER, fill_value=0)
        fig_pie = go.Figure(go.Pie(
            labels=counts.index.tolist(),
            values=counts.values.tolist(),
            marker_colors=[LEVEL_COLORS[l] for l in counts.index],
            hole=0.55,
            textfont_size=12,
            textinfo="label+percent",
        ))
        fig_pie.update_layout(
            height=280, paper_bgcolor="rgba(0,0,0,0)",
            font_color="#c9d1d9", margin=dict(t=10, b=10, l=10, r=10),
            showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

    with col_bar:
        # Stacked level timeline (binned into 60-second intervals)
        bin_size = max(1, int(ride_dur / 60))
        df["bin"] = (df["elapsed_sec"] // bin_size * bin_size).astype(int)
        binned = df.groupby(["bin", "level"]).size().reset_index(name="count")
        fig_stack = go.Figure()
        for lv in LEVEL_ORDER:
            sub = binned[binned["level"] == lv]
            fig_stack.add_trace(go.Bar(
                x=sub["bin"], y=sub["count"],
                name=lv, marker_color=LEVEL_COLORS[lv]
            ))
        fig_stack.update_layout(
            barmode="stack",
            height=280, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#c9d1d9", margin=dict(t=10, b=10, l=10, r=10),
            xaxis=dict(title="Elapsed (s)", gridcolor="#21262d", color="#7d8590"),
            yaxis=dict(title="Samples", gridcolor="#21262d", color="#7d8590"),
            legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h",
                        yanchor="bottom", y=1.02),
            bargap=0.05)
        st.plotly_chart(fig_stack, use_container_width=True, config={"displayModeBar": False})

# ── Blink + Yawn rates ─────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Blink & Yawn Rates</div>', unsafe_allow_html=True)
col_bl, col_yn = st.columns(2)

with col_bl:
    fig_bl = go.Figure()
    fig_bl.add_trace(go.Scatter(
        x=df["elapsed_sec"], y=df["blink_rate_pm"],
        mode="lines", name="Blink rate",
        line=dict(color="#3fb950", width=1.8)))
    fig_bl.add_hrect(y0=15, y1=25, fillcolor="rgba(63,185,80,0.07)",
                     line_width=0, annotation_text="Normal range",
                     annotation_position="top left",
                     annotation_font_color="#7d8590", annotation_font_size=10)
    fig_bl.update_layout(
        title=dict(text="Blinks per minute", font_size=12, x=0),
        height=210, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#c9d1d9", margin=dict(t=35, b=10, l=10, r=20),
        xaxis=dict(gridcolor="#21262d", color="#7d8590"),
        yaxis=dict(title="/min", gridcolor="#21262d", color="#7d8590"),
        showlegend=False)
    st.plotly_chart(fig_bl, use_container_width=True, config={"displayModeBar": False})

with col_yn:
    fig_yn = go.Figure()
    fig_yn.add_trace(go.Scatter(
        x=df["elapsed_sec"], y=df["yawn_rate_pm"],
        mode="lines", name="Yawn rate",
        line=dict(color="#ffa657", width=1.8),
        fill="tozeroy", fillcolor="rgba(255,166,87,0.07)"))
    fig_yn.update_layout(
        title=dict(text="Yawns per minute", font_size=12, x=0),
        height=210, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#c9d1d9", margin=dict(t=35, b=10, l=10, r=20),
        xaxis=dict(gridcolor="#21262d", color="#7d8590"),
        yaxis=dict(title="/min", gridcolor="#21262d", color="#7d8590"),
        showlegend=False)
    st.plotly_chart(fig_yn, use_container_width=True, config={"displayModeBar": False})

# ── Attention heatmap ──────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Attention Heatmap (per-minute segments)</div>', unsafe_allow_html=True)

minute_bins = max(1, int(ride_dur // 60))
if minute_bins >= 2:
    df["minute"] = (df["elapsed_sec"] // 60).astype(int)
    heat_data = df.groupby("minute")["attention_score"].mean().reset_index()
    heat_2d = heat_data["attention_score"].values.reshape(1, -1)
    fig_heat = go.Figure(go.Heatmap(
        z=heat_2d,
        x=[f"{int(m)}m" for m in heat_data["minute"]],
        colorscale=[
            [0.0, "#ff7b72"],
            [0.28, "#a371f7"],
            [0.45, "#58a6ff"],
            [0.62, "#39c5cf"],
            [1.0, "#3fb950"],
        ],
        zmin=0, zmax=100,
        showscale=True,
        colorbar=dict(title="Attention", tickfont=dict(color="#7d8590"), len=0.8),
    ))
    fig_heat.update_layout(
        height=130, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#c9d1d9", margin=dict(t=10, b=30, l=10, r=60),
        xaxis=dict(gridcolor="#21262d", color="#7d8590"),
        yaxis=dict(showticklabels=False),
    )
    st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})
else:
    st.info("Session too short for per-minute heatmap (need > 2 minutes).")

# ── Alert event log ────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Alert Event Log</div>', unsafe_allow_html=True)

if "level" in df.columns:
    non_alert = df[df["level"] != "ALERT"].copy()
    if not non_alert.empty:
        # group consecutive runs
        non_alert["grp"] = (non_alert["level"] != non_alert["level"].shift()).cumsum()
        events = non_alert.groupby("grp").agg(
            start=("elapsed_sec", "min"),
            end=("elapsed_sec", "max"),
            level=("level", "first"),
            min_att=("attention_score", "min")
        ).reset_index(drop=True)
        events["duration"] = events["end"] - events["start"]
        events = events[events["duration"] >= 2].sort_values("start", ascending=False)

        if not events.empty:
            table_html = """
<table style="width:100%;border-collapse:collapse;font-family:'JetBrains Mono',monospace;font-size:0.8rem;">
<thead><tr style="color:#7d8590;border-bottom:1px solid #21262d">
  <th style="text-align:left;padding:6px 8px">Time</th>
  <th style="text-align:left;padding:6px 8px">Level</th>
  <th style="text-align:right;padding:6px 8px">Duration</th>
  <th style="text-align:right;padding:6px 8px">Min Attention</th>
</tr></thead><tbody>
"""
            for _, row in events.head(30).iterrows():
                badge = level_badge(row["level"])
                table_html += f"""<tr style="border-bottom:1px solid #161b22">
  <td style="padding:5px 8px;color:#c9d1d9">{format_duration(row['start'])}</td>
  <td style="padding:5px 8px">{badge}</td>
  <td style="padding:5px 8px;text-align:right;color:#c9d1d9">{row['duration']:.0f}s</td>
  <td style="padding:5px 8px;text-align:right;color:{LEVEL_COLORS.get(row['level'],'#c9d1d9')}">{row['min_att']:.0f}</td>
</tr>"""
            table_html += "</tbody></table>"
            st.markdown(table_html, unsafe_allow_html=True)
        else:
            st.success("✅ No significant drowsiness events recorded.")
    else:
        st.success("✅ Driver stayed alert throughout the session.")

# ── Correlation scatter ────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Factor Correlation</div>', unsafe_allow_html=True)

feat_options = ["perclos", "blink_rate_pm", "yawn_rate_pm", "pitch", "yaw", "gaze_dev"]
feat_labels  = ["PERCLOS", "Blink rate", "Yawn rate", "Pitch", "Yaw", "Gaze dev"]
avail = [f for f in feat_options if f in df.columns]
avail_labels = [feat_labels[feat_options.index(f)] for f in avail]

col_x, col_y = st.columns(2)
with col_x:
    x_feat = st.selectbox("X axis", avail_labels, index=0)
with col_y:
    y_feat = st.selectbox("Y axis", avail_labels, index=min(2, len(avail_labels)-1))

x_col = avail[avail_labels.index(x_feat)]
y_col = avail[avail_labels.index(y_feat)]

fig_sc = px.scatter(
    df, x=x_col, y=y_col, color="level",
    color_discrete_map=LEVEL_COLORS,
    category_orders={"level": LEVEL_ORDER},
    opacity=0.55, size_max=6,
)
fig_sc.update_traces(marker_size=5)
fig_sc.update_layout(
    height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font_color="#c9d1d9", margin=dict(t=20, b=20, l=20, r=20),
    xaxis=dict(title=x_feat, gridcolor="#21262d", color="#7d8590"),
    yaxis=dict(title=y_feat, gridcolor="#21262d", color="#7d8590"),
    legend=dict(title="Level", bgcolor="rgba(0,0,0,0)"),
)
st.plotly_chart(fig_sc, use_container_width=True, config={"displayModeBar": False})

# ── Download ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">Export</div>', unsafe_allow_html=True)
c_dl1, c_dl2, _ = st.columns([1, 1, 3])
with c_dl1:
    st.download_button("⬇ Download telemetry CSV",
                       data=open(sel_path, "rb").read(),
                       file_name=os.path.basename(sel_path),
                       mime="text/csv")
with c_dl2:
    summary = df[["elapsed_sec","attention_score","perclos","level"]].to_csv(index=False)
    st.download_button("⬇ Download summary CSV",
                       data=summary,
                       file_name=f"summary_{sel_label}.csv",
                       mime="text/csv")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;color:#484f58;font-size:0.75rem;margin-top:2rem;padding-top:1rem;border-top:1px solid #21262d">
  WakeVision Enhanced · Driver Drowsiness Detection System
</div>
""", unsafe_allow_html=True)
