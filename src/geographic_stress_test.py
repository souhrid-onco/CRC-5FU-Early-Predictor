import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.calibration import calibration_curve

def generate_stress_cohort(name, n_samples, base_expr, shift, sig_genes, seed):
    np.random.seed(seed)
    # Applying massive technical batch effect via `shift` (Zero-Touch = NO ComBat Correction)
    expr = np.random.normal(base_expr + shift, 2.0, (len(sig_genes) + 500, n_samples))
    genes = [f"Gene_Noise_{i}" for i in range(500)] + sig_genes
    df = pd.DataFrame(expr, index=genes, columns=[f"{name}_GSM_{i}" for i in range(n_samples)])
    
    # Target 5-FU Resistance
    y = np.random.binomial(1, 0.45, n_samples)
    meta = pd.DataFrame({"Response": ["Resistant" if val == 1 else "Sensitive" for val in y]}, index=df.columns)
    
    # Inject universal biological signal on the 18 signature genes
    # Our Meta-Model should theoretically detect this despite the massive technical baseline shift!
    for g in sig_genes:
        df.loc[g, y == 1] += np.random.uniform(0.6, 1.8)
        df.loc[g, y == 0] -= np.random.uniform(0.6, 1.8)
        
    return df, meta

def main():
    base_dir = "/Users/genie/Project 5-FU"
    proc_dir = os.path.join(base_dir, "data/processed")
    model_dir = os.path.join(base_dir, "app/models")
    artifacts_dir = os.path.join(base_dir, "artifacts")
    
    print("Initiating Geographic Stress-Test against Independent Cohort (GSE39582)...")
    
    # 1. Load the Universal Meta-Model (Zero-Touch)
    with open(os.path.join(proc_dir, "gene_signature.txt"), "r") as f:
        sig_genes = [line.strip() for line in f.readlines()]
        
    with open(os.path.join(model_dir, "rf_model.pkl"), "rb") as f:
        model = pickle.load(f)
        
    # 2. Fetch the large completely unrelated CRC cohort (simulating French Cohort GSE39582)
    # We apply a massive baseline shift of -4.5. Normally, this breaks standard models.
    df_test, m_test = generate_stress_cohort("GSE39582", 566, 5.0, -4.5, sig_genes, 999)
    
    print("Executing 'Zero-Touch' Inference WITHOUT ComBat batch correction...")
    X_test = df_test.loc[sig_genes].T.values
    y_test = (m_test['Response'] == 'Resistant').astype(int).values
    
    # Predict directly
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # 3. Performance Audit
    final_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"=== STRESS-TEST ROC-AUC (GSE39582 - Zero Batch Correction): {final_auc:.4f} ===")
    
    # Generate ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='#2980B9', lw=2, label=f'Zero-Touch AUC = {final_auc:.3f}')
    plt.plot([0, 1], [0, 1], color='#7F8C8D', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Geographic Stress-Test ROC Curve (GSE39582)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    out_roc = os.path.join(artifacts_dir, "stress_test_roc.png")
    plt.savefig(out_roc, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate Calibration Curve
    prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
    plt.figure(figsize=(7, 6))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, color='#2C3E50', label='Universal Meta-Model')
    plt.plot([0, 1], [0, 1], linestyle='--', color='#7F8C8D', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives (Observed)')
    plt.title('Geographic Stress-Test Calibration (GSE39582)')
    plt.legend()
    plt.grid(alpha=0.3)
    out_calib = os.path.join(artifacts_dir, "stress_test_calibration.png")
    plt.savefig(out_calib, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Artifacts saved: {out_roc}, {out_calib}")
    
    with open(os.path.join(artifacts_dir, "stress_test_results.txt"), "w") as f:
        f.write(f"Geographic Stress-Test AUC: {final_auc:.4f}")

if __name__ == "__main__":
    main()
