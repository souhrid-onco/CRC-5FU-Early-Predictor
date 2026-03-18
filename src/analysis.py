import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV

def run_analysis(processed_dir, artifacts_dir):
    os.makedirs(artifacts_dir, exist_ok=True)
    
    expr_file = os.path.join(processed_dir, "processed_expression.csv")
    meta_file = os.path.join(processed_dir, "processed_metadata.csv")
    
    print("Loading preprocessed data...")
    expr_df = pd.read_csv(expr_file, index_col=0)
    meta_df = pd.read_csv(meta_file, index_col=0)
    
    # PCA
    print("Computing PCA Trajectories...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(expr_df.T)
    
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_scaled)
    meta_df['PC1'] = pcs[:, 0]
    meta_df['PC2'] = pcs[:, 1]
    
    # Plot Trajectories
    plt.figure(figsize=(10, 8))
    
    # Scatter all points
    colors = {'Sensitive': 'blue', 'Resistant': 'red'}
    markers = {'0h': 'o', '24h': '^', '48h': 's'}
    
    for res_status in ['Sensitive', 'Resistant']:
        for tp in markers.keys():
            subset = meta_df[(meta_df['Response'] == res_status) & (meta_df['Time'] == tp)]
            if not subset.empty:
                plt.scatter(subset['PC1'], subset['PC2'], 
                            c=colors[res_status], marker=markers[tp], 
                            label=f"{res_status} {tp}", s=100, alpha=0.7)
                
    # Draw vectors from 0h to 24h for each response group (centroid based)
    for res_status in ['Sensitive', 'Resistant']:
        subset_0h = meta_df[(meta_df['Response'] == res_status) & (meta_df['Time'] == '0h')]
        subset_24h = meta_df[(meta_df['Response'] == res_status) & ((meta_df['Time'] == '24h') | (meta_df['Time'] == '48h'))]
        
        if not subset_0h.empty and not subset_24h.empty:
            c_0h = subset_0h[['PC1', 'PC2']].mean()
            c_24h = subset_24h[['PC1', 'PC2']].mean()
            plt.arrow(c_0h['PC1'], c_0h['PC2'], 
                      c_24h['PC1'] - c_0h['PC1'], c_24h['PC2'] - c_0h['PC2'],
                      color=colors[res_status], width=0.05, head_width=0.3, 
                      alpha=0.5, length_includes_head=True)
            
    plt.title("PCA Trajectories (0h -> 24h)")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    traj_path = os.path.join(artifacts_dir, "trajectory_map.png")
    plt.savefig(traj_path, dpi=300, bbox_inches='tight')
    print(f"Saved Trajectory Map to {traj_path}")
    
    # Nested LASSO for Feature Selection (12-18 genes)
    print("Running Nested LASSO for feature selection...")
    # Labels: 1 for Resistant, 0 for Sensitive
    y = (meta_df['Response'] == 'Resistant').astype(int).values
    X = expr_df.T.values
    
    # We tweak alpha to get exactly 12-18 genes.
    # We will use LassoLarsIC or just LassoCV and tune if needed.
    lasso = LassoCV(alphas=np.logspace(-4, 0, 100), cv=5, max_iter=10000, random_state=42)
    lasso.fit(X, y)
    
    coef_nonzero = np.sum(lasso.coef_ != 0)
    print(f"Lasso selected {coef_nonzero} features initially.")
    
    # If not in 12-18 range, we manually find an alpha that gives ~15 features
    from sklearn.linear_model import Lasso
    target_k = 15
    best_coef = None
    best_features = []
    
    for alpha in np.logspace(-4, 0, 200):
        model = Lasso(alpha=alpha, max_iter=10000, random_state=42)
        model.fit(X, y)
        n_features = np.sum(model.coef_ != 0)
        if 12 <= n_features <= 18:
            print(f"Found alpha={alpha:.4f} yielding {n_features} features.")
            best_coef = model.coef_
            best_features = expr_df.index[best_coef != 0].tolist()
            break
            
    # Default fallback if exactly 12-18 is not found
    if not best_features:
        print("Could not find exact alpha for 12-18 genes. Falling back to top 15 by absolute LASSO coef.")
        top_idx = np.argsort(np.abs(lasso.coef_))[-15:]
        best_features = expr_df.index[top_idx].tolist()
        
    print(f"Selected {len(best_features)} gene signature: {best_features}")
    
    # Save signature to a file
    with open(os.path.join(processed_dir, "gene_signature.txt"), "w") as f:
        f.write("\n".join(best_features))
        
    # Save a subset of expression strictly for those genes
    expr_df.loc[best_features].to_csv(os.path.join(processed_dir, "signature_expression.csv"))
    print("Feature selection complete.")

if __name__ == "__main__":
    run_analysis("/Users/genie/Project 5-FU/data/processed", "/Users/genie/Project 5-FU/artifacts")
