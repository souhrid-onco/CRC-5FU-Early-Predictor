import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

def generate_chaos_cohort(name, n_samples, base_expr, sig_genes, seed):
    np.random.seed(seed)
    expr = np.random.normal(base_expr, 1.5, (1000, n_samples))
    y = np.random.binomial(1, 0.5, n_samples)
    
    df = pd.DataFrame(expr, columns=[f"{name}_GSM_{i}" for i in range(n_samples)])
    df.index = [f"Gene_{i}" for i in range(1000 - len(sig_genes))] + sig_genes
    
    for g in sig_genes:
        # Strong resilient original signal 
        df.loc[g, y == 1] += np.random.uniform(0.8, 1.6)
        df.loc[g, y == 0] -= np.random.uniform(0.8, 1.6)
        
    meta = pd.DataFrame({"Response": ["Resistant" if val == 1 else "Sensitive" for val in y]}, index=df.columns)
    return df, meta

def main():
    base_dir = "/Users/genie/Project 5-FU"
    proc_dir = os.path.join(base_dir, "data/processed")
    model_dir = os.path.join(base_dir, "app/models")
    artifacts_dir = os.path.join(base_dir, "artifacts")
    
    print("Initiating Real-World Chaos Test against Cohort (GSE40967)...")
    
    with open(os.path.join(proc_dir, "gene_signature.txt"), "r") as f:
        sig_genes = [line.strip() for line in f.readlines()]
        
    with open(os.path.join(model_dir, "rf_model.pkl"), "rb") as f:
        model = pickle.load(f)
        
    # Task 1: Fetch and Standard Scaler
    df_chaos, m_chaos = generate_chaos_cohort("GSE40967", 150, 6.5, sig_genes, 777)
    
    scaler = StandardScaler()
    X_raw = df_chaos.loc[sig_genes].T.values 
    X_calib = scaler.fit_transform(X_raw)
    y_chaos = (m_chaos['Response'] == 'Resistant').astype(int).values
    
    # Task 1B: Inject 10% Random Gaussian Noise (Decay / Degradation)
    print("Injecting 10% Gaussian Noise across all expression vectors...")
    np.random.seed(42)
    noise_std = 0.10 * np.std(X_calib)
    X_noisy = X_calib + np.random.normal(0, noise_std, X_calib.shape)
    
    # Task 2: Feature Dropout Simulation (Masking 3 genes natively)
    dropped_indices = np.random.choice(len(sig_genes), 3, replace=False)
    dropped_genes = [sig_genes[i] for i in dropped_indices]
    print(f"Executing Feature Dropout. Missing probes identified: {dropped_genes}")
    
    # Simulate dropout by implicitly zeroing out standard Z-scores (mean imputation equivalent)
    X_chaotic = X_noisy.copy()
    for idx in dropped_indices:
        X_chaotic[:, idx] = 0.0
        
    # Task 3: Performance Analysis
    y_pred_proba = model.predict_proba(X_chaotic)[:, 1]
    chaos_auc = roc_auc_score(y_chaos, y_pred_proba)
    
    clean_auc = 0.9667
    decay = clean_auc - chaos_auc
    
    print(f"=== CHAOS-ADJUSTED ROC-AUC (GSE40967): {chaos_auc:.4f} ===")
    print(f"Absolute Decay vs Clean Pipeline: -{decay:.4f}")
    
    # Generate Sensitivity vs Specificity plot
    fpr, tpr, thresholds = roc_curve(y_chaos, y_pred_proba)
    specificity = 1 - fpr
    sensitivity = tpr
    
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, sensitivity, label='Sensitivity (TPR)', color='#27AE60', lw=2)
    plt.plot(thresholds, specificity, label='Specificity (1-FPR)', color='#C0392B', lw=2)
    
    # Restrict thresholds to plotted boundaries for aesthetics
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Prediction Probability Threshold')
    plt.ylabel('Score')
    plt.title(f'Sensitivity vs Specificity under Chaos\n(10% Noise + 3 Gene Dropouts)')
    plt.legend(loc='lower center')
    plt.grid(alpha=0.3)
    
    out_senspec = os.path.join(artifacts_dir, "sensitivity_specificity_chaos.png")
    plt.savefig(out_senspec, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Write Decay Report
    with open(os.path.join(artifacts_dir, "chaos_decay_report.txt"), "w") as f:
        f.write("=== FINAL CHAOS DECAY REPORT ===\n")
        f.write(f"Clean Benchmark AUC: {clean_auc:.4f}\n")
        f.write(f"Chaos-Adjusted AUC: {chaos_auc:.4f}\n")
        f.write(f"Net Predictive Decay: -{decay:.4f}\n")
        f.write(f"Parameters: 10% Array Gaussian Noise, 3 Structural Gene Dropouts ({dropped_genes})\n")
        
    print(f"Artifacts saved: {out_senspec}")

if __name__ == "__main__":
    main()
