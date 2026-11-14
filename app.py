"""
AstroClassify: SDSS Stellar Object Classification Web Application
Built with Streamlit for DS605 Course Project (Fundamentals of Machine Learning).
"""

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="AstroClassify | SDSS Object Classification",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 15px;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .result-badge {
        padding: 12px 20px;
        border-radius: 8px;
        font-size: 1.3rem;
        font-weight: 700;
        text-align: center;
        margin: 10px 0;
    }
    .badge-galaxy { background-color: #dbeafe; color: #1e40af; border: 1px solid #93c5fd; }
    .badge-star { background-color: #fef3c7; color: #92400e; border: 1px solid #fcd34d; }
    .badge-qso { background-color: #f3e8ff; color: #6b21a8; border: 1px solid #d8b4fe; }
</style>
""", unsafe_allow_html=True)

# Resolve Paths
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR if (APP_DIR / "models").exists() else APP_DIR / "ML Project" / "astronomy-ml-classification"
if not (PROJECT_ROOT / "models").exists():
    PROJECT_ROOT = APP_DIR.parent

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_DIR = PROJECT_ROOT / "data"

CLASSES = ["GALAXY", "STAR", "QSO"]
PHOTOMETRIC_FEATURES = [
    "u", "g", "r", "i", "z",
    "u_g", "g_r", "r_i", "i_z",
    "u_r", "u_z", "g_z",
    "curvature_ugr", "curvature_gri",
    "mag_mean", "mag_std"
]

@st.cache_resource
def load_all_models():
    models = {}
    model_files = {
        "Random Forest (Champion)": "best_rf_model.joblib",
        "Gradient Boosting": "best_hgb_model.joblib",
        "Voting Ensemble": "best_voting_model.joblib",
        "Decision Tree": "best_dt_model.joblib",
        "k-Nearest Neighbors": "best_knn_model.joblib",
        "Linear SVM": "best_svm_model.joblib",
        "Logistic Regression": "best_lr_model.joblib",
        "RF + Spectroscopic Redshift": "redshift_rf_model.joblib"
    }
    for name, fname in model_files.items():
        fpath = MODELS_DIR / fname
        if fpath.exists():
            models[name] = joblib.load(fpath)
    return models

@st.cache_data
def load_metadata_and_reports():
    meta = {}
    if (MODELS_DIR / "model_metadata.json").exists():
        with open(MODELS_DIR / "model_metadata.json", "r") as f:
            meta = json.load(f)
    comp_df = pd.DataFrame()
    if (REPORTS_DIR / "model_comparison_results.csv").exists():
        comp_df = pd.read_csv(REPORTS_DIR / "model_comparison_results.csv")
    
    sample_scatter = pd.DataFrame()
    scatter_path = PROJECT_ROOT / "dashboard" / "looker_scatter_sample.csv"
    if scatter_path.exists():
        sample_scatter = pd.read_csv(scatter_path)
    return meta, comp_df, sample_scatter

models_dict = load_all_models()
metadata, comparison_df, sample_scatter = load_metadata_and_reports()

def compute_features(u, g, r, i, z):
    u_g = u - g
    g_r = g - r
    r_i = r - i
    i_z = i - z
    u_r = u - r
    u_z = u - z
    g_z = g - z
    curvature_ugr = u_g - g_r
    curvature_gri = g_r - r_i
    raw = [u, g, r, i, z]
    mag_mean = float(np.mean(raw))
    mag_std = float(np.std(raw, ddof=1))
    
    row = {
        "u": u, "g": g, "r": r, "i": i, "z": z,
        "u_g": u_g, "g_r": g_r, "r_i": r_i, "i_z": i_z,
        "u_r": u_r, "u_z": u_z, "g_z": g_z,
        "curvature_ugr": curvature_ugr,
        "curvature_gri": curvature_gri,
        "mag_mean": mag_mean,
        "mag_std": mag_std
    }
    return pd.DataFrame([row])

# App Header
st.markdown('<div class="main-title">🌌 AstroClassify: SDSS Stellar Object Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Automated Classification of Stars, Galaxies, and Quasars | <b>DS605: Fundamentals of Machine Learning</b></div>', unsafe_allow_html=True)

# Top KPI Metrics Ribbon
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Survey Dataset", "SDSS DR17", "99,999 objects")
with c2:
    st.metric("Champion Model", "Random Forest", "88.20% Accuracy")
with c3:
    st.metric("Macro F1-Score", "0.8524", "Balanced Evaluation")
with c4:
    st.metric("ROC-AUC Score", "0.9653", "High Class Separation")

# Sidebar Controls
st.sidebar.image("https://images.unsplash.com/photo-1506703719100-a0f3a48c0f86?w=600&auto=format&fit=crop&q=80", use_container_width=True)
st.sidebar.title("Configuration & Controls")

selected_model_name = st.sidebar.selectbox(
    "Select Classifier Architecture:",
    [
        "Random Forest (Champion)",
        "Gradient Boosting",
        "Voting Ensemble",
        "Decision Tree",
        "k-Nearest Neighbors",
        "Linear SVM",
        "Logistic Regression"
    ]
)

enable_redshift = st.sidebar.toggle("Include Spectroscopic Redshift (Ablation Mode)", value=False)
if enable_redshift:
    st.sidebar.info("🔬 Redshift Mode Activated: Model switches to RF trained with spectroscopic redshift ($z$). Expected accuracy jumps to ~97.6%.")

# Preset Selector
preset = st.sidebar.selectbox(
    "Load Astronomical Preset:",
    [
        "Custom Input",
        "Preset: Main-Sequence Star (Milky Way)",
        "Preset: Luminous Red Galaxy",
        "Preset: High-Redshift Quasar (QSO)"
    ]
)

preset_values = {
    "Custom Input": {"u": 19.5, "g": 18.2, "r": 17.5, "i": 17.1, "z": 16.9, "z_val": 0.05},
    "Preset: Main-Sequence Star (Milky Way)": {"u": 18.4, "g": 17.2, "r": 16.8, "i": 16.6, "z": 16.5, "z_val": 0.0001},
    "Preset: Luminous Red Galaxy": {"u": 20.8, "g": 18.9, "r": 17.4, "i": 16.9, "z": 16.5, "z_val": 0.42},
    "Preset: High-Redshift Quasar (QSO)": {"u": 19.2, "g": 19.0, "r": 18.8, "i": 18.6, "z": 18.3, "z_val": 2.15}
}
default_vals = preset_values[preset]

# Main Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔭 Live Object Classifier",
    "📁 Batch CSV Inference",
    "📊 Model Performance Benchmark",
    "🌌 Exploratory Data & Sky Map",
    "📚 Chalk Talk & Viva Preparation"
])

# -------------------------------------------------------------
# TAB 1: LIVE OBJECT CLASSIFIER
# -------------------------------------------------------------
with tab1:
    st.subheader("Interactive Celestial Object Prediction")
    st.markdown("Adjust the photometric passband magnitudes ($u, g, r, i, z$) captured by the SDSS 2.5-meter optical telescope.")
    
    col_in1, col_in2 = st.columns([1.1, 1.4])
    
    with col_in1:
        st.markdown("#### Photometric Magnitudes (Flux Passbands)")
        u_val = st.slider("u-band (Ultraviolet, ~3543 Å):", 12.0, 28.0, float(default_vals["u"]), 0.1)
        g_val = st.slider("g-band (Green-Optical, ~4770 Å):", 12.0, 28.0, float(default_vals["g"]), 0.1)
        r_val = st.slider("r-band (Red-Optical, ~6231 Å):", 12.0, 28.0, float(default_vals["r"]), 0.1)
        i_val = st.slider("i-band (Near-Infrared 1, ~7625 Å):", 12.0, 28.0, float(default_vals["i"]), 0.1)
        z_val = st.slider("z-band (Near-Infrared 2, ~9134 Å):", 12.0, 28.0, float(default_vals["z"]), 0.1)
        
        redshift_input = 0.0
        if enable_redshift:
            redshift_input = st.slider("Spectroscopic Redshift (z):", -0.01, 7.0, float(default_vals["z_val"]), 0.01)

    # Compute Features
    df_feat = compute_features(u_val, g_val, r_val, i_val, z_val)
    if enable_redshift:
        df_feat["redshift"] = redshift_input

    with col_in2:
        st.markdown("#### Real-Time Engineered Features")
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            st.metric("u - g (UV Slope)", f"{df_feat['u_g'].values[0]:.3f}")
            st.metric("g - r (Optical Slope)", f"{df_feat['g_r'].values[0]:.3f}")
        with fc2:
            st.metric("r - i (Red Slope)", f"{df_feat['r_i'].values[0]:.3f}")
            st.metric("i - z (NIR Slope)", f"{df_feat['i_z'].values[0]:.3f}")
        with fc3:
            st.metric("u - r (Bimodality)", f"{df_feat['u_r'].values[0]:.3f}")
            st.metric("Spectral Dispersion", f"{df_feat['mag_std'].values[0]:.3f}")
        
        # Classification Execution
        chosen_clf_name = "RF + Spectroscopic Redshift" if enable_redshift else selected_model_name
        active_pipeline = models_dict.get(chosen_clf_name)

        if active_pipeline is not None:
            input_cols = PHOTOMETRIC_FEATURES + (["redshift"] if enable_redshift else [])
            pred_class = active_pipeline.predict(df_feat[input_cols])[0]
            
            # Predict Probabilities
            clf_step = active_pipeline.named_steps["clf"]
            if hasattr(clf_step, "predict_proba"):
                probs = active_pipeline.predict_proba(df_feat[input_cols])[0]
            elif hasattr(clf_step, "decision_function"):
                df_scores = active_pipeline.decision_function(df_feat[input_cols])[0]
                exp_s = np.exp(df_scores - np.max(df_scores))
                probs = exp_s / np.sum(exp_s)
            else:
                probs = [0.33, 0.33, 0.34]
            
            prob_dict = {cls: float(p) for cls, p in zip(CLASSES, probs)}
            confidence = prob_dict[pred_class]

            badge_class = {
                "GALAXY": ("badge-galaxy", "🌌 GALAXY (Extended Extragalactic System)"),
                "STAR": ("badge-star", "⭐ STAR (Milky Way Galactic Stellar Object)"),
                "QSO": ("badge-qso", "⚡ QUASAR (QSO — Quasi-Stellar Object / Active Galactic Nucleus)")
            }[pred_class]

            st.markdown(f'<div class="result-badge {badge_class[0]}">{badge_class[1]}<br><span style="font-size:0.95rem; font-weight:normal;">Confidence: <b>{confidence*100:.1f}%</b> via {chosen_clf_name}</span></div>', unsafe_allow_html=True)
            
            # Probabilities Bar Chart
            prob_df = pd.DataFrame({
                "Class": CLASSES,
                "Probability": [prob_dict[c] for c in CLASSES]
            })
            fig_prob = px.bar(
                prob_df, x="Probability", y="Class", orientation="h",
                color="Class",
                color_discrete_map={"GALAXY": "#2563eb", "STAR": "#d97706", "QSO": "#9333ea"},
                range_x=[0, 1.0],
                text_auto=".1%",
                height=180
            )
            fig_prob.update_layout(margin=dict(l=10, r=10, t=10, b=10), showlegend=False, yaxis_title="")
            st.plotly_chart(fig_prob, use_container_width=True)
        else:
            st.warning("Selected model pipeline could not be loaded from disk.")

    st.divider()
    # Color-Color Diagram with Real-Time Marker Overlay
    st.markdown("#### Celestial Diagnostic Position (Color-Color Diagram: $(g-r)$ vs $(u-g)$)")
    st.caption("The colored background illustrates 3,000 real SDSS objects. Your query object is indicated by the large red diamond marker.")
    
    if not sample_scatter.empty:
        fig_scatter = px.scatter(
            sample_scatter.sample(min(2500, len(sample_scatter)), random_state=42),
            x="g_r", y="u_g", color="class",
            color_discrete_map={"GALAXY": "#3b82f6", "STAR": "#f59e0b", "QSO": "#a855f7"},
            opacity=0.35,
            labels={"g_r": "g - r (Optical Color Slope)", "u_g": "u - g (Ultraviolet Excess)"},
            title="SDSS Empirical Survey Footprint vs Current Query Object"
        )
        # Add query object as prominent glowing marker
        fig_scatter.add_trace(go.Scatter(
            x=[df_feat["g_r"].values[0]],
            y=[df_feat["u_g"].values[0]],
            mode="markers",
            marker=dict(size=18, color="#ef4444", symbol="diamond", line=dict(width=3, color="black")),
            name=f"Current Query ({pred_class})"
        ))
        fig_scatter.update_layout(height=480, margin=dict(l=20, r=20, t=40, b=20), xaxis_range=[-1.0, 3.0], yaxis_range=[-1.5, 4.0])
        st.plotly_chart(fig_scatter, use_container_width=True)

# -------------------------------------------------------------
# TAB 2: BATCH CSV INFERENCE
# -------------------------------------------------------------
with tab2:
    st.subheader("Batch CSV Classification Pipeline")
    st.markdown("Upload a CSV file containing SDSS photometric measurements (`u, g, r, i, z`) to classify thousands of celestial targets simultaneously.")

    # Sample CSV Generator
    sample_rows = [
        {"obj_ID": "123766105", "u": 18.35, "g": 17.18, "r": 16.79, "i": 16.61, "z": 16.48, "redshift": 0.0001, "True_Class": "STAR"},
        {"obj_ID": "123766106", "u": 20.81, "g": 18.92, "r": 17.41, "i": 16.92, "z": 16.51, "redshift": 0.4350, "True_Class": "GALAXY"},
        {"obj_ID": "123766107", "u": 19.22, "g": 19.01, "r": 18.82, "i": 18.63, "z": 18.35, "redshift": 2.1450, "True_Class": "QSO"},
        {"obj_ID": "123766108", "u": 17.65, "g": 16.52, "r": 16.12, "i": 15.98, "z": 15.89, "redshift": 0.0002, "True_Class": "STAR"},
        {"obj_ID": "123766109", "u": 21.45, "g": 19.34, "r": 17.82, "i": 17.21, "z": 16.85, "redshift": 0.3850, "True_Class": "GALAXY"},
        {"obj_ID": "123766110", "u": 19.82, "g": 19.55, "r": 19.21, "i": 18.99, "z": 18.65, "redshift": 1.8700, "True_Class": "QSO"}
    ]
    sample_csv_data = pd.DataFrame(sample_rows).to_csv(index=False)
    
    st.download_button(
        "📥 Download Sample Test CSV (with Ground Truth)",
        data=sample_csv_data,
        file_name="sample_sdss_test_data.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("Upload Target Catalog CSV:", type=["csv"])
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write(f"Uploaded **{len(batch_df)}** astronomical targets.")
        
        req_bands = ["u", "g", "r", "i", "z"]
        if not all(b in batch_df.columns for b in req_bands):
            st.error("Uploaded CSV must contain columns: `u`, `g`, `r`, `i`, `z`.")
        else:
            # Batch Feature Engineering
            b_feat = batch_df.copy()
            b_feat["u_g"] = b_feat["u"] - b_feat["g"]
            b_feat["g_r"] = b_feat["g"] - b_feat["r"]
            b_feat["r_i"] = b_feat["r"] - b_feat["i"]
            b_feat["i_z"] = b_feat["i"] - b_feat["z"]
            b_feat["u_r"] = b_feat["u"] - b_feat["r"]
            b_feat["u_z"] = b_feat["u"] - b_feat["z"]
            b_feat["g_z"] = b_feat["g"] - b_feat["z"]
            b_feat["curvature_ugr"] = b_feat["u_g"] - b_feat["g_r"]
            b_feat["curvature_gri"] = b_feat["g_r"] - b_feat["r_i"]
            b_feat["mag_mean"] = b_feat[req_bands].mean(axis=1)
            b_feat["mag_std"] = b_feat[req_bands].std(axis=1)

            use_red = enable_redshift and ("redshift" in b_feat.columns)
            clf_name = "RF + Spectroscopic Redshift" if use_red else selected_model_name
            active_clf = models_dict.get(clf_name)

            if active_clf:
                cols = PHOTOMETRIC_FEATURES + (["redshift"] if use_red else [])
                batch_preds = active_clf.predict(b_feat[cols])
                batch_df["Predicted_Class"] = batch_preds
                
                if hasattr(active_clf.named_steps["clf"], "predict_proba"):
                    b_probs = active_clf.predict_proba(b_feat[cols])
                    batch_df["Confidence"] = np.max(b_probs, axis=1).round(3)
                
                # Visual Distribution Summary
                col_b1, col_b2 = st.columns([1, 1.4])
                with col_b1:
                    class_summary = batch_df["Predicted_Class"].value_counts().reset_index()
                    class_summary.columns = ["Class", "Count"]
                    fig_pie = px.pie(
                        class_summary, names="Class", values="Count",
                        color="Class",
                        color_discrete_map={"GALAXY": "#2563eb", "STAR": "#d97706", "QSO": "#9333ea"},
                        hole=0.45,
                        title="Predicted Class Distribution"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col_b2:
                    st.markdown("#### Classified Records Table")
                    disp_cols = [c for c in ["obj_ID", "u", "g", "r", "i", "z", "Predicted_Class", "Confidence", "True_Class"] if c in batch_df.columns]
                    st.dataframe(batch_df[disp_cols].head(50), use_container_width=True)
                
                out_csv = batch_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Complete Predicted Catalog (CSV)",
                    data=out_csv,
                    file_name="classified_sdss_predictions.csv",
                    mime="text/csv"
                )

# -------------------------------------------------------------
# TAB 3: MODEL PERFORMANCE BENCHMARK
# -------------------------------------------------------------
with tab3:
    st.subheader("Model Performance Benchmark & Evaluation Metrics")
    st.markdown("Rigorous evaluation on **20,000 held-out test objects** with Stratified 5-Fold Cross Validation.")

    if not comparison_df.empty:
        # Comparison Table
        st.markdown("#### 1. Evaluation Benchmark Table")
        st.dataframe(
            comparison_df.style.format({
                "CV_Macro_F1": "{:.4f}",
                "Test_Accuracy": "{:.2%}",
                "Macro_Precision": "{:.4f}",
                "Macro_Recall": "{:.4f}",
                "Macro_F1": "{:.4f}",
                "Weighted_F1": "{:.4f}",
                "F1_Galaxy": "{:.4f}",
                "F1_Star": "{:.4f}",
                "F1_QSO": "{:.4f}",
                "ROC_AUC_Macro": "{:.4f}",
                "Train_Time_sec": "{:.2f}s"
            }),
            use_container_width=True
        )

        st.markdown("#### 2. Performance Comparison Chart")
        fig_bench = px.bar(
            comparison_df, x="Model", y=["Test_Accuracy", "Macro_F1", "Weighted_F1"],
            barmode="group",
            labels={"value": "Score", "variable": "Evaluation Metric"},
            color_discrete_sequence=["#1e3a8a", "#d97706", "#059669"],
            title="Comparison of Supervised Learning Algorithms (Photometry-Only)"
        )
        fig_bench.update_layout(yaxis_range=[0.5, 1.0], height=420)
        st.plotly_chart(fig_bench, use_container_width=True)

        col_cm, col_roc = st.columns(2)
        with col_cm:
            st.markdown("#### 3. Normalized Confusion Matrices")
            cm_img_path = FIGURES_DIR / "02_confusion_matrices.png"
            if cm_img_path.exists():
                st.image(str(cm_img_path), use_container_width=True, caption="Normalized Confusion Matrices across Top 6 Models")
        
        with col_roc:
            st.markdown("#### 4. Multiclass ROC Curves (One-vs-Rest)")
            roc_img_path = FIGURES_DIR / "03_multiclass_roc_curves.png"
            if roc_img_path.exists():
                st.image(str(roc_img_path), use_container_width=True, caption="One-vs-Rest ROC Curves for Champion Random Forest")

        st.markdown("#### 5. Feature Importance & Scientific Redshift Ablation")
        c_fi, c_abl = st.columns(2)
        with c_fi:
            fi_path = FIGURES_DIR / "04_feature_importance_rf.png"
            if fi_path.exists():
                st.image(str(fi_path), use_container_width=True, caption="Gini Feature Importance (Random Forest)")
        with c_abl:
            abl_path = FIGURES_DIR / "05_photometry_vs_redshift_ablation.png"
            if abl_path.exists():
                st.image(str(abl_path), use_container_width=True, caption="Ablation Study: Value of Spectroscopic Redshift")

# -------------------------------------------------------------
# TAB 4: EXPLORATORY DATA & SKY MAP
# -------------------------------------------------------------
with tab4:
    st.subheader("Exploratory Data Analysis & Spatial Sky Footprint")
    st.markdown("Observations collected by the Sloan Digital Sky Survey across Right Ascension ($\\alpha$) and Declination ($\\delta$).")

    eda_dir = PROJECT_ROOT / "figures" / "eda"
    fe_dir = PROJECT_ROOT / "figures" / "feature_engineering"

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.markdown("#### Survey Sky Footprint (RA vs Dec)")
        sky_path = eda_dir / "12_sky_coverage_sample.png"
        if sky_path.exists():
            st.image(str(sky_path), use_container_width=True, caption="Spatial Distribution of SDSS Observations")
        
        st.markdown("#### Passband Correlations")
        corr_path = fe_dir / "01_feature_correlation_matrix.png"
        if corr_path.exists():
            st.image(str(corr_path), use_container_width=True, caption="Correlation Matrix of Engineered Photometric Features")

    with col_e2:
        st.markdown("#### Magnitude Distributions by Class")
        mag_path = eda_dir / "02_magnitude_histograms_by_class.png"
        if mag_path.exists():
            st.image(str(mag_path), use_container_width=True, caption="Apparent Magnitude Distributions Across SDSS Filters")

        st.markdown("#### PCA Scree Plot & Dimensionality Reduction")
        scree_path = fe_dir / "02_pca_scree_plot.png"
        if scree_path.exists():
            st.image(str(scree_path), use_container_width=True, caption="Principal Component Analysis Scree Plot")

# -------------------------------------------------------------
# TAB 5: CHALK TALK & VIVA PREPARATION GUIDE
# -------------------------------------------------------------
with tab5:
    st.subheader("DS605 Viva & Chalk Talk Interactive Preparation Guide")
    st.markdown("Comprehensive theoretical revision covering syllabus topics, mathematical formulations, and anticipated questions.")

    with st.expander("📌 1. Project Motivation & Problem Formulation", expanded=True):
        st.markdown("""
        - **Core Problem:** Classify celestial objects into **Stars, Galaxies, and Quasars** based on multi-band photometric survey measurements.
        - **Why Machine Learning?** Modern sky surveys like SDSS, Rubin Observatory LSST, and Euclid image hundreds of millions of objects. Manual classification or individual spectroscopic targeting is physically and financially impossible.
        - **Why Photometry vs Spectroscopy?** Photometric imaging takes wide-field snapshots in seconds through colored glass filters ($u, g, r, i, z$). Spectroscopy requires dispersing light with a diffraction grating, taking 100 to 1,000 times more telescope time. Hence, building high-accuracy *photometry-only* models is the paramount challenge in astronomical data science.
        """)

    with st.expander("📐 2. Mathematical Formulations of Supervised Learning Algorithms"):
        st.markdown(r"""
        #### A. Multinomial Logistic Regression (Softmax Loss)
        For multiclass target $k \in \{1, \dots, K\}$:
        $$P(Y = k | x) = \frac{\exp(w_k^T x + b_k)}{\sum_{j=1}^K \exp(w_j^T x + b_j)}$$
        The objective minimizes Cross-Entropy Loss with $L_2$ regularization:
        $$\mathcal{L}(W) = -\frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K y_{ik} \log P(Y = k | x_i) + \frac{\lambda}{2} \sum_{k=1}^K \|w_k\|^2$$

        #### B. Support Vector Machines (Linear SVM & Hinge Loss)
        Syllabus Unit: *Classification using Hinge Loss*. The soft-margin primal objective is:
        $$\min_{w, b, \xi} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^N \xi_i \quad \text{s.t.} \quad y_i(w^T x_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0$$
        The Hinge Loss penalty is: $\ell(y, f(x)) = \max(0, 1 - y f(x))$. Support vectors are the data points lying on the margin boundaries ($y_i f(x_i) = 1$) or violating them ($\xi_i > 0$).

        #### C. Decision Tree (CART & Gini Impurity)
        Gini Impurity for node $t$:
        $$I_G(t) = 1 - \sum_{k=1}^K p(k|t)^2$$
        Optimal split maximizes Impurity Decrease (Information Gain):
        $$\Delta I(t) = I_G(t) - \frac{N_{t_L}}{N_t} I_G(t_L) - \frac{N_{t_R}}{N_t} I_G(t_R)$$

        #### D. Random Forest (Bagging & Variance Reduction)
        Averaging $B$ individual de-correlated trees reduces ensemble variance:
        $$\text{Var}(\bar{T}) = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2$$
        Feature subsampling (selecting $m = \sqrt{p}$ features at each split) reduces the inter-tree correlation $\rho$, driving total variance down without increasing individual tree bias.
        """)

    with st.expander("⚖️ 3. Handling Class Imbalance (STAR: 21.6%, GALAXY: 59.4%, QSO: 19.0%)"):
        st.markdown(r"""
        - **Why Accuracy is Deceptive:** A naive majority classifier predicting every object as `GALAXY` achieves **59.45% Accuracy**, yet has a **Macro F1 of only 0.248** and misses 100% of Stars and Quasars!
        - **Cost-Sensitive Learning (`class_weight='balanced'`):** Penalizes errors inversely proportional to class frequencies:
          $$w_k = \frac{N}{K \cdot N_k}$$
        - **Macro F1-Score:** The arithmetic mean of per-class F1-scores: $\text{Macro F1} = \frac{1}{3}(F1_{\text{Galaxy}} + F1_{\text{Star}} + F1_{\text{QSO}})$, ensuring minority classes are weighted equally during optimization.
        """)

    with st.expander("❓ 4. Anticipated Chalk Talk Questions & Model Answers"):
        st.markdown(r"""
        **Q1: Why did you drop the row with -9999 in Phase 1?**  
        *Answer:* `-9999` is SDSS's numerical sentinel flag indicating a failed camera readout or corrupted CCD pixel. Because it occurred in only **1 out of 100,000 observations (0.001%)**, dropping that single row preserves complete data integrity without requiring synthetic imputation.

        **Q2: What is the physical meaning of colour indices ($u-g, g-r$)?**  
        *Answer:* Magnitudes are logarithmic flux measurements: $m_1 - m_2 = -2.5 \log_{10}(F_1 / F_2)$. The difference between two passbands is independent of the object's absolute distance and measures the slope of the object's Spectral Energy Distribution (SED). It reflects physical temperature, surface gravity, and absorption features.

        **Q3: Why does adding spectroscopic redshift yield 97.6% accuracy, and why didn't you use it in your primary model?**  
        *Answer:* Quasars have massive cosmological redshifts ($z > 1.0$), galaxies have moderate redshifts ($z \approx 0.1 - 0.7$), and Milky Way stars have nearly zero redshift ($z \approx 0$). Including spectroscopic redshift makes the problem trivial. However, spectroscopic redshift requires taking spectra, which costs 100x more telescope time than imaging. A true photometric classification model must operate on imaging bands alone. We included redshift strictly as an ablation study to demonstrate its diagnostic power.
        """)

# App Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #6b7280; font-size: 0.85rem;">
    <b>Astronomical Object Classification Project</b> | M.Sc. Data Science Autumn 2026<br>
    Sloan Digital Sky Survey (SDSS DR17) | Scikit-Learn • Streamlit • Plotly
</div>
""", unsafe_allow_html=True)
