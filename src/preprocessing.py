import pandas as pd
import numpy as np
import scanpy as sc
import os

def run_preprocessing(raw_dir, processed_dir):
    os.makedirs(processed_dir, exist_ok=True)
    
    # Try finding the CSVs from GSE28702 or the synthetic dataset
    expr_file = os.path.join(raw_dir, "GSE28702_expression.csv")
    meta_file = os.path.join(raw_dir, "GSE28702_metadata.csv")
    
    if not os.path.exists(expr_file):
        expr_file = os.path.join(raw_dir, "GSE_synth_expression.csv")
        meta_file = os.path.join(raw_dir, "GSE_synth_metadata.csv")

    print(f"Loading {expr_file} and {meta_file}...")
    expr_df = pd.read_csv(expr_file, index_col=0)
    meta_df = pd.read_csv(meta_file, index_col=0)  # index_col=0 implies Sample_ID is index if it was saved that way.
    
    # Wait, the data mining script saved metadata with index=False.
    meta_df = pd.read_csv(meta_file)
    meta_df.set_index("Sample_ID", inplace=True)
    
    # Ensure they match
    samples = list(set(expr_df.columns).intersection(meta_df.index))
    expr_df = expr_df[samples]
    meta_df = meta_df.loc[samples]
    
    # If the data is from GEO microarrays, it's often already log-transformed.
    # We will simulate the log2(TPM+1) for safety by ensuring values > 0.
    expr_df = expr_df.clip(lower=0)
    # Log2 if not already scaled (values > 100 usually mean unlogged)
    if expr_df.max().max() > 50:
        expr_df = np.log2(expr_df + 1)
        
    print("Converting to AnnData for ComBat-seq / Batch effect correction...")
    # Scanpy expects observations in rows, variables (genes) in columns
    adata = sc.AnnData(X=expr_df.T.values, obs=meta_df, var=pd.DataFrame(index=expr_df.index))
    
    # Check if we have multiple batches
    if 'Batch' in adata.obs.columns and adata.obs['Batch'].nunique() > 1:
        # Standard combat in scanpy
        sc.pp.combat(adata, key='Batch')
        print("Batch effect successfully removed with ComBat.")
    else:
        print("Only one batch detected or 'Batch' missing. Skipping ComBat.")
        
    # Variance filtering
    print("Filtering for top 500 variable genes...")
    # Compute variance for each gene
    variances = np.var(adata.X, axis=0)
    # Get top 500 indices
    top_500_idx = np.argsort(variances)[-500:]
    adata_top = adata[:, top_500_idx].copy()
    
    # Save back to CSV for analysis
    processed_expr = pd.DataFrame(adata_top.X.T, index=adata_top.var_names, columns=adata_top.obs_names)
    processed_expr.to_csv(os.path.join(processed_dir, "processed_expression.csv"))
    adata_top.obs.to_csv(os.path.join(processed_dir, "processed_metadata.csv"))
    print("Preprocessing completed. Targets saved.")

if __name__ == "__main__":
    run_preprocessing("/Users/genie/Project 5-FU/data/raw", "/Users/genie/Project 5-FU/data/processed")
