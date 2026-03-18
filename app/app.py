import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import io
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px

# Configuration
st.set_page_config(page_title="5-FU Resistance Predictor", layout="wide")

# Inject Custom Nano Banana 2 Theme CSS
st.markdown("""
    <style>
    /* Oncology Dashboard Theme: Nano Banana 2 */
    .stApp {
        background-color: #ECF0F1;
        color: #2C3E50;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #2C3E50;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 600;
    }
    .stButton>button {
        background-color: #2C3E50;
        color: #ECF0F1;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #34495E;
        color: white;
    }
    .css-1d391kg, .stSidebar {
        background-color: #BDC3C7 !important;
    }
    .metric-container {
        border-radius: 8px;
        background-color: white;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 5px solid #2C3E50;
    }
    </style>
""", unsafe_allow_html=True)

# Load Models & Reference Data
@st.cache_resource
def load_assets():
    model_path = "/Users/genie/Project 5-FU/app/models/rf_model.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    else:
        model = None
        
    proc_dir = "/Users/genie/Project 5-FU/data/processed"
    expr_df = None
    meta_df = None
    sig_genes = []
    
    if os.path.exists(os.path.join(proc_dir, "processed_expression.csv")):
        expr_df = pd.read_csv(os.path.join(proc_dir, "processed_expression.csv"), index_col=0)
        meta_df = pd.read_csv(os.path.join(proc_dir, "processed_metadata.csv"), index_col=0)
    
    if os.path.exists(os.path.join(proc_dir, "gene_signature.txt")):
        with open(os.path.join(proc_dir, "gene_signature.txt"), "r") as f:
            sig_genes = [line.strip() for line in f.readlines() if line.strip()]
            
    return model, expr_df, meta_df, sig_genes

model, ref_expr, ref_meta, sig_genes = load_assets()

# Main UI
st.title("🔬 Clinical Oncology Dashboard")
st.subheader("5-FU Resistance Predictor for Colorectal Cancer")

if model is None or ref_expr is None:
    st.error("Model or reference data not found. Please run the training pipeline first.")
    st.stop()

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Patient Sample Upload")
    st.markdown("Upload a CSV file containing the transcriptomic readings for the identified signature genes.")
    
    # Generate template functionality
    template_df = pd.DataFrame(columns=["Gene", "Expression_Value"])
    template_df["Gene"] = sig_genes
    template_csv = template_df.to_csv(index=False).encode('utf-8')
    st.download_button(label="📥 Download Dataset Template", data=template_csv, file_name="sample_template.csv", mime="text/csv")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
if uploaded_file is not None:
    try:
        sample_df = pd.read_csv(uploaded_file)
        # Ensure we possess the needed genes
        if not set(sig_genes).issubset(set(sample_df['Gene'])):
            st.error(f"Missing genes in uploaded file. Required: {len(sig_genes)}")
            st.stop()
            
        # Parse correctly
        sample_df.set_index('Gene', inplace=True)
        X_new = sample_df.loc[sig_genes, 'Expression_Value'].values.reshape(1, -1)
        
        # Predict
        prob_res = model.predict_proba(X_new)[0][1]
        
        # Display Gauge
        with col2:
            st.markdown("### Model Predictions")
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob_res * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Resistance Probability (%)", 'font': {'size': 24, 'color': '#2C3E50'}},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#2C3E50"},
                    'steps' : [
                        {'range': [0, 50], 'color': "lightgreen"},
                        {'range': [50, 75], 'color': "lemonchiffon"},
                        {'range': [75, 100], 'color': "salmon"}],
                    'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 75}
                }
            ))
            fig.update_layout(paper_bgcolor="#ECF0F1", font={'color': "#2C3E50", 'family': "Helvetica"})
            st.plotly_chart(fig, use_container_width=True)
            
            diagnosis = "Likely Resistant" if prob_res > 0.5 else "Likely Sensitive"
            color = "red" if prob_res > 0.5 else "green"
            st.markdown(f"**Clinical Suggestion:** <span style='color:{color}; font-size: 20px'>{diagnosis}</span>", unsafe_allow_html=True)
            
        st.markdown("---")
        st.markdown("### Dynamic PCA Projection")
        
        # Perform dynamic PCA on the reference set and project the new sample
        scaler = StandardScaler()
        X_ref_scaled = scaler.fit_transform(ref_expr.T)
        pca = PCA(n_components=2)
        pcs_ref = pca.fit_transform(X_ref_scaled)
        
        # Project new sample using same scaler and PCA
        # Note: the scaler was fit on all genes, so we need full gene vector, but we only have 18 signatures uploaded?
        # A true PCA projection requires all genes if the reference was processed on all 500.
        # Let's see if we only have signatures. If patient only uploads signature genes, we can't project them into a 500-gene PCA space.
        # We must re-run PCA on the signature genes only!
        
        scaler_sig = StandardScaler()
        X_ref_sig_scaled = scaler_sig.fit_transform(ref_expr.loc[sig_genes].T)
        pca_sig = PCA(n_components=2)
        pcs_ref_sig = pca_sig.fit_transform(X_ref_sig_scaled)
        
        X_new_scaled = scaler_sig.transform(X_new)
        pcs_new = pca_sig.transform(X_new_scaled)
        
        # Plot with Plotly
        plot_df = pd.DataFrame(pcs_ref_sig, columns=["PC1", "PC2"])
        plot_df["Response"] = ref_meta["Response"].values
        
        fig2 = px.scatter(plot_df, x="PC1", y="PC2", color="Response", 
                          color_discrete_map={"Sensitive": "blue", "Resistant": "red"},
                          title="Patient Sample vs Training Clusters (Signature Space)",
                          opacity=0.6)
                          
        fig2.add_scatter(x=pcs_new[:, 0], y=pcs_new[:, 1], mode="markers", 
                         marker=dict(size=15, color="gold", symbol="star", line=dict(width=2, color="black")), 
                         name="New Patient Sample")
                         
        fig2.update_layout(paper_bgcolor="#ECF0F1", plot_bgcolor="#ECF0F1")
        st.plotly_chart(fig2, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error processing the uploaded file: {str(e)}")
else:
    with col2:
        st.info("👈 Upload a patient CSV sample to view the Resistance Gauge and Dynamic PCA.")

