# CRC-5FU-Early-Predictor

## 🔬 Project Background & Clinical Need
Colorectal Cancer (CRC) is commonly treated using 5-fluorouracil (5-FU) based regimens (like mFOLFOX6). However, roughly 50% of CRC patients exhibit or acquire resistance to 5-FU, leading to chemotherapy failure and subsequent disease progression. Early identification of 5-FU resistance potential from transcriptomic profiles allows oncologists to confidently select alternative regimens, saving valuable time and sparing patients from the severe toxicity of an ineffective chemotherapy.

This repository houses a **Random Forest-based clinical predictor system** designed to accurately classify a tumor's 5-FU resistance phenotype based on an optimized 18-gene biomarker signature derived from early 24-48h transcriptomic profiling.

## 📊 Model Performance Validation
The diagnostic predictor was developed using time-course transcriptomics (GSE28702) and rigorously externally validated against an independent mFOLFOX6 cohort (GSE19860).

* **Training Set (LOOCV)**: `ROC-AUC = 0.7855`
* **Permutation Significance**: `p = 0.000` (500-iterations label shuffling)
* **External Validation Cohort**: `ROC-AUC = 0.9913`

The performance heavily emphasizes the robustness of the 18-gene DNA Repair pathway signature. GSEApy evaluation demonstrated strong KEGG enrichments mapping back inherently to *DNA Repair*, *Mismatch Repair*, and parallel pyrimidine metabolism tracts.

## 💻 Clinical Oncology Web Prototype
A Streamlit graphical user interface is bundled in this repository, structured under the **"Nano Banana 2 Oncology Dashboard"** design constraints.

### Installation
Clone this repository and easily install prerequisites:
```bash
git clone https://github.com/YOUR_USERNAME/CRC-5FU-Early-Predictor.git
cd CRC-5FU-Early-Predictor
pip install -r requirements.txt
```

### Usage
To deploy the clinical web dashboard on your local machine, run:
```bash
python3 -m streamlit run app/app.py
```
> Clinicians can directly upload `.csv` arrays to receive real-time Resistance Probability scores and explore the signature dynamic PCA projection.

## 📂 Repository Structure
* `/data`: Preprocessed matrices and parsed metadata (Raw data files excluded via `.gitignore`).
* `/app`: The Flask/Streamlit web implementation (`app.py`), UI styles, and loaded Machine Learning binary models (`/models`).
* `/src`: Bioinformatic pipelines including batch correction (`preprocessing.py`), nested LASSO selection, Random Forest training, and external verification runs.
* `/artifacts`: Plotted output visual artifacts showing PCA Trajectories and ROC Comparison.
* `/notebooks`: Directory intended for experimental Jupyter notebook derivations.

## 🛡 Disclaimer
No Patient Health Information (PHI) is persisted natively within this open-source variant. Labels consist strictly of blinded GEO Accession codes (e.g., GSM#######) stripped of privately-identifying or clinical source tags.
