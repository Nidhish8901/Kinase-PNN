import streamlit as st
import os
import sys
import subprocess

# --- Page Configuration ---
st.set_page_config(
    page_title="Bioethanol PINN Dashboard",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS Styling ---
st.markdown("""
<style>
    .stApp { background-color: #0b0f19; color: #e2e8f0; }
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
        color: white; border: none; border-radius: 8px;
        padding: 0.5rem 1rem; font-weight: 600; transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px); box-shadow: 0 8px 20px rgba(139, 92, 246, 0.4);
    }
    [data-testid="stSidebar"] { background-color: #111827; border-right: 1px solid #1f2937; }
    h1, h2, h3 {
        background: -webkit-linear-gradient(45deg, #60a5fa, #c084fc);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .stTextArea textarea {
        background-color: #1e293b !important; color: #38bdf8 !important;
        font-family: 'Fira Code', monospace !important; border: 1px solid #334155 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧪 Bioethanol Fermentation PINN")
st.markdown("*Transfer Learning & Physics-Informed Neural Networks Dashboard*")
st.divider()

# --- 1. Setup Data Directory ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
DATA_PATH = os.path.join(DATA_DIR, "complete_fermentation_dataset.csv")

# --- 2. Require File Upload ---
if not os.path.exists(DATA_PATH):
    st.error("### ⚠️ Dataset Missing")
    st.info("Please upload **'complete_fermentation_dataset.csv'** to initialize the pipeline.")
    
    uploaded_file = st.file_uploader("Drop your CSV here", type=["csv"])
    if uploaded_file is not None:
        with open(DATA_PATH, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("Dataset secured. Initializing dashboard...")
        st.rerun()
    st.stop()

# --- Navigation Sidebar ---
st.sidebar.success("✅ System Ready")
st.sidebar.markdown("### Navigation")
phase = st.sidebar.radio("Select Execution Phase:", [
    "📊 Phase 1: Preprocessing",
    "🧠 Phase 2: PINN Development",
    "🎯 Phase 3: TL Evaluation",
    "⚙️ Phase 3C: Hard-IC PINN",
    "🔬 Cross-Episode Validation"
])

def read_text_file(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    return "Artifact not found. Please run the phase first."

# --- Robust Subprocess Runner (Fixes crashes, freezing, and encoding) ---
def run_script_stream(script_name):
    st.info(f"Executing `{script_name}`. Please wait, this may take several minutes...")
    log_container = st.empty()
    log_text = ""
    
    try:
        # Force Python to output in UTF-8 to prevent Windows cp1252 crashes
        custom_env = os.environ.copy()
        custom_env["PYTHONIOENCODING"] = "utf-8"
        
        # Run using the exact Python executable running Streamlit, enforcing UTF-8
        process = subprocess.Popen(
            [sys.executable, "-X", "utf8", script_name], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True,
            encoding="utf-8",           
            errors="replace",           
            bufsize=1,
            env=custom_env              
        )
        
        # Stream output to UI in real-time
        for line in process.stdout:
            log_text += line
            # Keep only last 4000 characters so the browser doesn't lag
            display_text = log_text if len(log_text) < 4000 else "...\n" + log_text[-4000:]
            log_container.code(display_text, language="text")
            
        process.wait()
        if process.returncode == 0:
            st.success(f"{script_name} completed successfully!")
        else:
            st.error(f"Execution failed with exit code {process.returncode}. Check the logs above.")
            
    except Exception as e:
        st.error(f"Error running {script_name}: {e}")

# ==========================================
# UI Logic
# ==========================================

# --- Phase 1 ---
if phase == "📊 Phase 1: Preprocessing":
    st.header("Phase 1: Dataset Preprocessing")
    st.caption("Cleans raw data, filters contamination, and extracts glucose-consumption phases.")
    
    if st.button("🚀 Execute Phase 1"):
        run_script_stream("phase1_preprocessing.py")
    
    tab1, tab2 = st.tabs(["📈 Visualizations", "📄 Processing Report"])
    with tab1:
        if os.path.exists("outputs/phase1/phase1_visualization.png"):
            st.image("outputs/phase1/phase1_visualization.png", use_container_width=True)
    with tab2:
        report = read_text_file("outputs/phase1/phase1_report.txt")
        st.text_area("Execution Logs", report, height=350)

# --- Phase 2 ---
elif phase == "🧠 Phase 2: PINN Development":
    st.header("Phase 2: Physics-Informed Neural Network")
    st.caption("Trains the base PINN vs Baseline NN on Episode 20.")
    
    if st.button("🚀 Train Models (Phase 2)"):
        run_script_stream("phase2_pinn.py")
        
    tab1, tab2 = st.tabs(["🔮 Model Predictions", "📉 Diagnostics"])
    with tab1:
        if os.path.exists("outputs/phase2/predictions_ep20.png"):
            st.image("outputs/phase2/predictions_ep20.png", use_container_width=True)
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists("outputs/phase2/loss_history.png"):
                st.image("outputs/phase2/loss_history.png", caption="Training Convergence")
        with col2:
            if os.path.exists("outputs/phase2/physics_residuals.png"):
                st.image("outputs/phase2/physics_residuals.png", caption="ODE Physics Residuals")

# --- Phase 3 ---
elif phase == "🎯 Phase 3: TL Evaluation":
    st.header("Phase 3: Transfer Learning Evaluation")
    st.caption("Evaluates transfer learning performance on unseen data (Target: Episode 75).")
    
    if st.button("🚀 Execute TL Evaluation"):
        run_script_stream("phase3_final.py")

    tab1, tab2 = st.tabs(["📊 Performance (RMSE)", "📈 Trajectory Predictions"])
    with tab1:
        if os.path.exists("outputs/phase3_final/rmse_vs_nobs_unconstrained.png"):
            st.image("outputs/phase3_final/rmse_vs_nobs_unconstrained.png", use_container_width=True)
    with tab2:
        if os.path.exists("outputs/phase3_final/predictions_n5_unconstrained.png"):
            st.image("outputs/phase3_final/predictions_n5_unconstrained.png", use_container_width=True)

# --- Phase 3C ---
elif phase == "⚙️ Phase 3C: Hard-IC PINN":
    st.header("Phase 3C: Hard Initial-Condition Enforcement")
    st.caption("Applies exact exponential residual transforms for rigorous IC matching.")
    
    if st.button("🚀 Execute Phase 3C"):
        run_script_stream("phase3c_hard_ic.py")

    tab1, tab2 = st.tabs(["📉 Validation Metrics", "📄 Detailed Report"])
    with tab1:
        if os.path.exists("outputs/phase3c/rmse_vs_nobs.png"):
            st.image("outputs/phase3c/rmse_vs_nobs.png", use_container_width=True)
    with tab2:
        report = read_text_file("outputs/phase3c/phase3c_report.txt")
        st.text_area("Analysis Report", report, height=400)

# --- Cross-Episode Validation ---
elif phase == "🔬 Cross-Episode Validation":
    st.header("Cross-Episode Validation")
    st.caption("Scalability tests across extreme high and low initial glucose environments.")
    
    if st.button("🚀 Run Validation Suite"):
        run_script_stream("run_cross_episode_v2.py")

    tab1, tab2, tab3 = st.tabs(["⚖️ Glucose Comparison", "📈 Generalization", "📑 Executive Summary"])
    with tab1:
        if os.path.exists("outputs/cross_episode_validation/glucose_rmse_comparison.png"):
            st.image("outputs/cross_episode_validation/glucose_rmse_comparison.png", use_container_width=True)
    with tab2:
        if os.path.exists("outputs/cross_episode_validation/glucose_predictions.png"):
            st.image("outputs/cross_episode_validation/glucose_predictions.png", use_container_width=True)
    with tab3:
        report = read_text_file("outputs/cross_episode_validation/comparison_report.txt")
        st.text_area("Performance Metrics", report, height=450)