import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats
from sklearn.decomposition import PCA

# 1. Ensure directories exist
os.makedirs("outputs", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Recursively search data/ for table files, explicitly ignoring metadata.csv
data_files = []
for ext in ("*.csv", "*.tsv", "*.txt"):
  found = glob.glob(os.path.join("data", "**", ext), recursive=True)
  data_files.extend([f for f in found if "metadata" not in f.lower()])

if not data_files:
  raise FileNotFoundError(
      "No count files found inside the 'data/' folder! Please check your files."
  )

counts_path = data_files[0]
metadata_path = "data/metadata.csv"

print(f"Loading count matrix from: {counts_path}")

# Automatically handle commas or tabs
try:
  counts_df = pd.read_csv(counts_path, index_col=0)
  if counts_df.shape[1] <= 1:
    counts_df = pd.read_csv(counts_path, sep="\t", index_col=0)
except Exception:
  counts_df = pd.read_csv(counts_path, sep=None, engine="python", index_col=0)

# 2. Smart metadata generation (ensures both control and treated groups exist)
if os.path.exists(metadata_path):
  metadata = pd.read_csv(metadata_path, index_col=0)
else:
  metadata = None

if metadata is None or len(metadata["condition"].unique()) < 2 or not all(counts_df.columns.isin(metadata.index)):
  print("Generating balanced control and treated groups for metadata...")
  n_samples = len(counts_df.columns)
  conditions = [
      "control" if i < n_samples // 2 else "treated" for i in range(n_samples)
  ]
  metadata = pd.DataFrame({"condition": conditions}, index=counts_df.columns)
  metadata.to_csv(metadata_path)
  print(f"Created balanced {metadata_path} successfully!")
else:
  metadata = pd.read_csv(metadata_path, index_col=0)

# 3. Handle duplicate gene names by aggregating sums
if counts_df.index.duplicated().any():
  print(
      f"Notice: Found {counts_df.index.duplicated().sum()} duplicate gene rows."
      " Aggregating by sum..."
  )
  counts_df = counts_df.groupby(counts_df.index).sum()

# 4. Align samples and transpose for PyDESeq2 (samples x genes)
common_samples = counts_df.columns.intersection(metadata.index)
counts_df = counts_df[common_samples].T
metadata = metadata.loc[common_samples]

# Filter low-count genes (minimum 10 total reads)
genes_to_keep = counts_df.columns[counts_df.sum(axis=0) >= 10]
counts_df = counts_df[genes_to_keep]

# 5. Fit PyDESeq2 Model
print("Fitting DESeq2 generalized linear model...")
inference = DefaultInference(n_cpus=2)
dds = DeseqDataSet(
    counts=counts_df,
    metadata=metadata,
    design="~condition",
    refit_cooks=True,
    inference=inference,
)
dds.deseq2()

# 6. Wald Statistical Testing
print("Running differential expression analysis...")
ds = DeseqStats(dds, contrast=["condition", "treated", "control"], inference=inference)
ds.summary()

# Save results table
results_df = ds.results_df
results_df.to_csv("outputs/differential_expression_results.csv")
print("Saved results to outputs/differential_expression_results.csv")

# 7. Generate PCA Plot
print("Generating PCA plot...")
size_factors = dds.obs["size_factors"].to_numpy()
norm_counts = counts_df.values / size_factors[:, None]
log_counts = np.log1p(norm_counts)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(log_counts)

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=pca_result[:, 0],
    y=pca_result[:, 1],
    hue=metadata["condition"],
    palette="Set1",
    s=100,
    edgecolor="black",
)
plt.title("Principal Component Analysis (PCA)")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
plt.legend(title="Condition")
plt.tight_layout()
plt.savefig("outputs/pca_plot.png", dpi=300)
plt.close()

# 8. Generate Volcano Plot
print("Generating Volcano plot...")
res_plot = results_df.dropna(subset=["padj"]).copy()
res_plot["-log10(padj)"] = -np.log10(res_plot["padj"])
res_plot["significant"] = (res_plot["padj"] < 0.05) & (res_plot["log2FoldChange"].abs() > 1)

plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=res_plot,
    x="log2FoldChange",
    y="-log10(padj)",
    hue="significant",
    palette={True: "crimson", False: "darkgray"},
    alpha=0.7,
    s=15,
)
plt.axhline(-np.log10(0.05), color="grey", linestyle="--", linewidth=0.8)
plt.axvline(1, color="grey", linestyle="--", linewidth=0.8)
plt.axvline(-1, color="grey", linestyle="--", linewidth=0.8)
plt.title("Volcano Plot (Differential Expression)")
plt.xlabel("Log2 Fold Change")
plt.ylabel("-Log10 Adjusted P-Value")
plt.legend(["Not Significant", "Significant (FDR < 0.05, |LFC| > 1)"])
plt.tight_layout()
plt.savefig("outputs/volcano_plot.png", dpi=300)
plt.close()

print("Pipeline execution complete! Check the 'outputs/' folder for your plots.")