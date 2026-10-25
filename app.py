import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# Configure page
st.set_page_config(
    page_title="AEROINTEL AI | Advanced Aero Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Advanced Custom Styling
st.markdown("""
<style>
.stApp {
    background: linear-gradient(145deg, #060e1a 0%, #0a1626 60%, #112239 100%);
    color: #f0f6fc;
}
[data-testid="stSidebar"] {
    background-color: #050d18;
    border-right: 1px solid #1e364f;
}
.metric-container {
    background: linear-gradient(135deg, #0e1e2f, #071322);
    border: 1px solid #1f3d5a;
    border-radius: 12px;
    padding: 15px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}
.hero-banner {
    padding: 25px;
    border-radius: 15px;
    background: linear-gradient(120deg, #112d47, #081626);
    border: 1px solid #204566;
    margin-bottom: 25px;
}
.header-title {
    color: #4da3ff;
    font-family: 'Helvetica Neue', sans-serif;
    font-weight: 800;
    margin: 0;
}
.sub-text { color: #8ba2b9; font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)

from pathlib import Path

BASE = Path(__file__).resolve().parent

CL_PATH = BASE / "AEROINTEL_Cl_Quadratic_RSM.pkl"
CD_PATH = BASE / "AEROINTEL_Cd_Quadratic_RSM.pkl"
DATA_PATH = BASE / "NACA 0012 Validation - Master DoE Matrix.csv"

# Load artifacts
@st.cache_resource
def load_models():
    return joblib.load(CL_PATH), joblib.load(CD_PATH)

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        return None
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        "Angle_of_Attack deg": "AoA_deg",
        "Velocity m/s": "Velocity_m_s"
    })
    return df

# Hero section
st.markdown("""
<div class="hero-banner">
    <span class="sub-text">AEROINTEL FLIGHT RESEARCH LABS · MODEL v2.0</span>
    <h1 class="header-title">AEROINTEL AI</h1>
    <p style="margin: 5px 0 0 0; color: #b0c4de;">Interactive Aerodynamic Parameter Design & Live Quadrature RSM Prediction Dashboard (NACA 0012)</p>
</div>
""", unsafe_allow_html=True)

if not CL_PATH.exists() or not CD_PATH.exists():
    st.error("Models are missing from /content!")
    st.stop()

cl_model, cd_model = load_models()
df = load_data()

# Sidebar
st.sidebar.markdown("### 🛠️ DESIGN CONTROLS")
aoa = st.sidebar.slider("Angle of Attack (deg)", 0.10, 11.98, 5.00, 0.01)
velocity = st.sidebar.slider("Velocity (m/s)", 30.1, 99.7, 60.0, 0.1)

# Perform Live Predictions
inputs = pd.DataFrame({"AoA_deg": [aoa], "Velocity_m_s": [velocity]})
cl_pred = float(cl_model.predict(inputs)[0])
cd_pred = float(cd_model.predict(inputs)[0])
ld_ratio = cl_pred / cd_pred if abs(cd_pred) > 1e-6 else 0.0

# Live Predictions Layout
st.markdown("### ⚡ LIVE DESIGN PERFORMANCE")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="metric-container"><span class="sub-text">Lift Coefficient (Cl)</span><h2>{cl_pred:.5f}</h2></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-container"><span class="sub-text">Drag Coefficient (Cd)</span><h2>{cd_pred:.5f}</h2></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-container"><span class="sub-text">L/D Aerodynamic Efficiency</span><h2>{ld_ratio:.3f}</h2></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="metric-container"><span class="sub-text">Design Airspeed</span><h2>{velocity:.1f} m/s</h2></div>', unsafe_allow_html=True)

st.write("")

tabs = st.tabs(["📈 Interactive Prediction Curves", "🔍 Experimental Run Explorer", "📊 Model Diagnostics", "💾 Export & Raw Data"])

with tabs[0]:
    st.markdown("### Angle of Attack Sweep Analysis")
    aoa_sweep = np.linspace(0.1, 11.98, 200)
    sweep_df = pd.DataFrame({"AoA_deg": aoa_sweep, "Velocity_m_s": np.full_like(aoa_sweep, velocity)})

    cl_sweep = cl_model.predict(sweep_df)
    cd_sweep = cd_model.predict(sweep_df)
    ld_sweep = cl_sweep / np.where(cd_sweep == 0, 1e-6, cd_sweep)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=aoa_sweep, y=cl_sweep, name="Lift Coefficient (Cl)", line=dict(color="#1f77b4", width=3)))
    fig.add_trace(go.Scatter(x=aoa_sweep, y=cd_sweep, name="Drag Coefficient (Cd)", yaxis="y2", line=dict(color="#ff7f0e", width=3)))
    fig.add_vline(x=aoa, line_dash="dash", line_color="white", annotation_text="Your Selected Design Point")

    fig.update_layout(template="plotly_dark")
        height=450,
        title=f"Aerodynamic Response Curves at fixed Speed: {velocity:.1f} m/s",
        xaxis_title="Angle of Attack (deg)",
        yaxis=dict(title="Lift Coefficient Cl", titlefont=dict(color="#1f77b4"), tickfont=dict(color="#1f77b4")),
        yaxis2=dict(title="Drag Coefficient Cd", titlefont=dict(color="#ff7f0e"), tickfont=dict(color="#ff7f0e"), overlaying="y", side="right"),
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    st.markdown("### Find and Compare Experimental Runs")
    if df is not None:
        run_list = sorted(df["Run_ID"].unique())
        selected_run = st.selectbox("Select an experimental run to compare against the AI prediction model:", run_list)

        run_row = df[df["Run_ID"] == selected_run].iloc[0]

        st.markdown(f"#### Comparison for {selected_run} (Speed: {run_row['Velocity_m_s']:.1f} m/s)")

        # Model prediction at identical AoA and Velocity
        comp_input = pd.DataFrame({"AoA_deg": [run_row["AoA_deg"]], "Velocity_m_s": [run_row["Velocity_m_s"]]})
        pred_comp_cl = float(cl_model.predict(comp_input)[0])
        pred_comp_cd = float(cd_model.predict(comp_input)[0])

        c_col1, c_col2 = st.columns(2)
        with c_col1:
            st.write(f"**Lift Coefficient (Cl)**")
            st.write(f"• Observed CFD: `{run_row['Cl']:.5f}`")
            st.write(f"• Model Prediction: `{pred_comp_cl:.5f}`")
            st.write(f"• Residual Error: `{run_row['Cl'] - pred_comp_cl:+.5f}`")
        with c_col2:
            st.write(f"**Drag Coefficient (Cd)**")
            st.write(f"• Observed CFD: `{run_row['Cd']:.5f}`")
            st.write(f"• Model Prediction: `{pred_comp_cd:.5f}`")
            st.write(f"• Residual Error: `{run_row['Cd'] - pred_comp_cd:+.5f}`")
            if run_row["Run_ID"] == "Run_045":
                st.warning("⚠️ Run_045 is identified as a statistical drag outlier! Physical verification is highly recommended.")
    else:
        st.info("No database file available to browse.")

with tabs[2]:
    st.markdown("### Model Performance Statistics")
    st.caption("These stats are fetched live from the cross-validation and holdout validation protocols.")

    # Try to load holdout metrics
    metrics_file = BASE / "AEROINTEL_Final_Holdout_Metrics.csv"
    if metrics_file.exists():
        st.write("**Locked Holdout Performance (8° ≤ AoA < 10°)**")
        st.dataframe(pd.read_csv(metrics_file), use_container_width=True)
    else:
        st.info("Holdout evaluation CSV not found.")

with tabs[3]:
    st.markdown("### Database Tables & Exports")
    if df is not None:
        st.dataframe(df, use_container_width=True, height=250)
        st.download_button("📥 Export Full NACA 0012 Database", df.to_csv(index=False), "naca_database.csv", "text/csv", use_container_width=True)
    else:
        st.info("No database tables loaded.")
