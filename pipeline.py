import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

# Ensure output directory exists
os.makedirs("outputs", exist_ok=True)

print("1. Loading raw counts matrix...")
# Load the raw counts file from the data folder
counts_path = "data/GSE52778_raw_counts_GRCh38.p13_NCBI.tsv.gz"
counts = pd.read_csv(counts_path, sep="\t", index_col=0)

# DESeq2 requires samples as rows and genes as columns, so we transpose (.T)
counts = counts.T

# Create sample metadata table mapping sample rows to conditions
# GSE52778 has 8 samples split evenly: 4 Control, 4 Treated
metadata = pd.DataFrame(index=counts.index)
half_len = len(counts) // 2
metadata['condition'] = ['Control'] * half_len + ['Treated'] * (len(counts) - half_len)

print(f"Loaded {counts.shape[1]} genes across {counts.shape[0]} samples.")

print("2. Filtering low-count genes...")
# Remove genes with fewer than 10 total counts across all samples
keep_genes = counts.sum(axis=0) >= 10
counts_filtered = counts.loc[:, keep_genes]
print(f"Retained {counts_filtered.shape[1]} genes after low-count filtering.")

print("3. Running DESeq2 differential expression analysis...")
# Initialize and run PyDESeq2 model
dds = DeseqDataSet(
    counts=counts_filtered,
    metadata=metadata,
    design_factors="condition"
)
dds.deseq2()

# Extract contrast results (Treated vs Control)
stat_res = DeseqStats(dds, contrast=["condition", "Treated", "Control"])
stat_res.summary()
results_df = stat_res.results_df

# Save differential expression results table
results_df.to_csv("outputs/differential_expression_results.csv")
print("Saved differential expression results to outputs/")


print("4. Generating publication-grade Volcano plot...")
sns.set_theme(style="whitegrid")
volcano_df = results_df.dropna(subset=['padj', 'log2FoldChange']).copy()
volcano_df['neg_log10_padj'] = -np.log10(volcano_df['padj'])

# Define significance thresholds (FDR < 0.05 and absolute Log2 Fold Change > 1)
volcano_df['significance'] = 'Not Significant'
volcano_df.loc[(volcano_df['padj'] < 0.05) & (volcano_df['log2FoldChange'] > 1), 'significance'] = 'Upregulated'
volcano_df.loc[(volcano_df['padj'] < 0.05) & (volcano_df['log2FoldChange'] < -1), 'significance'] = 'Downregulated'

# Plot Volcano
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=volcano_df,
    x='log2FoldChange',
    y='neg_log10_padj',
    hue='significance',
    palette={'Not Significant': 'darkgrey', 'Upregulated': '#d95f02', 'Downregulated': '#7570b3'},
    alpha=0.8,
    s=25
)
plt.axhline(-np.log10(0.05), linestyle='--', color='black', alpha=0.4, linewidth=1)
plt.axvline(1, linestyle='--', color='black', alpha=0.4, linewidth=1)
plt.axvline(-1, linestyle='--', color='black', alpha=0.4, linewidth=1)
plt.title('Differential Expression: Treated vs Control', fontsize=14, fontweight='bold')
plt.xlabel('Log2 Fold Change', fontsize=12)
plt.ylabel('-Log10 Adjusted P-Value', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
plt.tight_layout()
plt.savefig('outputs/volcano_plot.png', dpi=300)
plt.close()


print("5. Generating PCA plot...")
# Extract normalized counts from PyDESeq2
norm_counts = pd.DataFrame(
    dds.layers['normalized_counts'],
    index=counts_filtered.index,
    columns=counts_filtered.columns
)

# Apply log2 transformation for variance stabilization
log_norm_counts = np.log2(norm_counts + 1)

# Run PCA
pca = PCA(n_components=2)
pca_coords = pca.fit_transform(log_norm_counts.T)

pca_df = pd.DataFrame(pca_coords, index=log_norm_counts.columns, columns=['PC1', 'PC2'])
pca_df['condition'] = metadata['condition'].values
var_explained = pca.explained_variance_ratio_ * 100

# Plot PCA
plt.figure(figsize=(7, 6))
sns.scatterplot(
    data=pca_df,
    x='PC1',
    y='PC2',
    hue='condition',
    s=120,
    palette={'Control': '#1b9e77', 'Treated': '#e7298a'},
    edgecolor='black',
    alpha=0.9
)
plt.title('Principal Component Analysis (PCA)', fontsize=14, fontweight='bold')
plt.xlabel(f'PC1 ({var_explained[0]:.1f}% Variance)', fontsize=12)
plt.ylabel(f'PC2 ({var_explained[1]:.1f}% Variance)', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
plt.tight_layout()
plt.savefig('outputs/pca_plot.png', dpi=300)
plt.close()

print("Pipeline execution complete! All results and plots are saved inside the outputs/ folder.")