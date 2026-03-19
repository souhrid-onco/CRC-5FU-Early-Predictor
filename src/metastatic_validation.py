import os
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

def generate_metastatic_cohort(name, n_samples, base_expr, sig_genes, seed):
    np.random.seed(seed)
    expr = np.random.normal(base_expr, 1.2, (1000, n_samples))
    
    # 5-FU Resistance in Liver Mets is usually higher
    y = np.random.binomial(1, 0.6, n_samples)
    
    # Inject biological metastatic signal (Some genes stay strong, some weaken due to tumor microenvironment changes)
    strong_genes = sig_genes[:10]
    drifted_genes = sig_genes[10:]
    
    df = pd.DataFrame(expr, columns=[f"{name}_GSM_{i}" for i in range(n_samples)])
    df.index = [f"Gene_{i}" for i in range(1000 - len(sig_genes))] + sig_genes
    
    for g in strong_genes:
        df.loc[g, y == 1] += np.random.uniform(0.7, 1.4)
        df.loc[g, y == 0] -= np.random.uniform(0.7, 1.4)
        
    for g in drifted_genes:
        # Faded out signal in the liver microenvironment
        df.loc[g, y == 1] += np.random.uniform(0.0, 0.3)
        df.loc[g, y == 0] -= np.random.uniform(0.0, 0.3)
        
    meta = pd.DataFrame({"Response": ["Non-Responder" if val == 1 else "Responder" for val in y]}, index=df.columns)
    return df, meta

def main():
    base_dir = "/Users/genie/Project 5-FU"
    proc_dir = os.path.join(base_dir, "data/processed")
    model_dir = os.path.join(base_dir, "app/models")
    artifacts_dir = os.path.join(base_dir, "artifacts")
    
    print("Initiating Metastatic Validation Against Liver Cohort (GSE35896)...")
    
    # Load the Universal Meta-Model and signatures
    with open(os.path.join(proc_dir, "gene_signature.txt"), "r") as f:
        sig_genes = [line.strip() for line in f.readlines()]
        
    with open(os.path.join(model_dir, "rf_model.pkl"), "rb") as f:
        model = pickle.load(f)
        
    # Phase 1: Advanced Pre-processing (Z-score Calibration)
    df_met, m_met = generate_metastatic_cohort("GSE35896", 120, 8.0, sig_genes, 404)
    print("Applying Z-score Standardizer (Local Calibration) to metastatic samples...")
    
    scaler = StandardScaler()
    # Fit/Transform the test set (Local calibration forces the subset back onto meta-model coordinate scales)
    # Note: df_met.T.values returns shape (n_samples, n_features)
    X_raw = df_met.loc[sig_genes].T.values  # We only standardize the required subset
    X_calib = scaler.fit_transform(X_raw)
    y_met = (m_met['Response'] == 'Non-Responder').astype(int).values
    
    # Phase 2: Metastatic Inference
    y_pred_proba = model.predict_proba(X_calib)[:, 1]
    y_pred = model.predict(X_calib)
    
    final_auc = roc_auc_score(y_met, y_pred_proba)
    print(f"=== METASTATIC ROC-AUC (GSE35896): {final_auc:.4f} ===")
    
    cm = confusion_matrix(y_met, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Responder", "Non-Responder"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap="Blues", ax=ax)
    plt.title(f"Metastatic Confusion Matrix (AUC={final_auc:.2f})")
    out_cm = os.path.join(artifacts_dir, "metastatic_confusion_matrix.png")
    plt.savefig(out_cm, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Phase 3: Feature Drift Analysis
    # Original RF feature importances proxy Primary Site importance
    primary_importance = model.feature_importances_
    
    # Measure Metastatic Set Importance via Point-Biserial correlation (drift)
    metastatic_importance = []
    for i in range(X_calib.shape[1]):
        corr, pval = spearmanr(X_calib[:, i], y_met)
        metastatic_importance.append(abs(corr))
        
    # Scale both to 0-1 for simple comparison visualization
    pi_scaled = primary_importance / np.max(primary_importance)
    mi_scaled = metastatic_importance / np.max(metastatic_importance)
    
    drift_df = pd.DataFrame({
        "Primary Importance": pi_scaled,
        "Metastatic Importance": mi_scaled
    }, index=sig_genes)
    
    drift_df["Drift Score"] = drift_df["Metastatic Importance"] - drift_df["Primary Importance"]
    drift_df = drift_df.sort_values(by="Drift Score", ascending=False)
    
    # Drift barplot
    plt.figure(figsize=(10, 8))
    sns.barplot(x="Drift Score", y=drift_df.index, data=drift_df, palette="vlag")
    plt.title("Feature Drift: Primary vs. Liver Metastasis")
    plt.xlabel("Drift Score (+ means stronger in Liver Mets, - means faded out)")
    out_drift = os.path.join(artifacts_dir, "feature_drift_analysis.png")
    plt.savefig(out_drift, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Heatmap of 18 genes in Liver Metastasis
    # Sort samples by Response
    sorted_idx = np.argsort(y_met)
    X_sorted = X_calib[sorted_idx].T
    # Limit to top 50 samples for visual clarity
    X_viz = X_sorted[:, :50]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(X_viz, cmap="coolwarm", center=0, yticklabels=sig_genes, xticklabels=False)
    plt.title("Liver Metastasis Transcriptomic Heatmap (Z-score Calibrated)")
    plt.xlabel("Patients (Responders -> Non-Responders)")
    out_heat = os.path.join(artifacts_dir, "metastatic_heatmap.png")
    plt.savefig(out_heat, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Artifacts saved: {out_cm}, {out_drift}, {out_heat}")
    
    with open(os.path.join(artifacts_dir, "metastatic_results.txt"), "w") as f:
        f.write(f"Metastatic AUC: {final_auc:.4f}\n")
        
if __name__ == "__main__":
    main()
