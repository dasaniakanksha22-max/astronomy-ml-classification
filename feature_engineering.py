"""
Phase 4: Feature Engineering and Dimensionality Reduction for SDSS Classification.
Generates engineered features, statistical diagnostics, PCA, and visualization artifacts.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.preprocessing import StandardScaler

# Set aesthetic styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11

PROJECT_ROOT = Path(r"c:\Users\Akanksha\OneDrive\Desktop\ML_PROJECT2\ML Project\astronomy-ml-classification")
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures" / "feature_engineering"
REPORTS_DIR = PROJECT_ROOT / "reports"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load Cleaned Dataset
clean_data_path = DATA_DIR / "processed" / "sdss_cleaned.csv"
print(f"Loading cleaned data from {clean_data_path}...")
df = pd.read_csv(clean_data_path, dtype={"obj_ID": "string", "spec_obj_ID": "string"})
print(f"Loaded {len(df)} rows and {len(df.columns)} columns.")

# 2. Derive Photometric Features
# A. Adjacent color indices (spectral slopes)
df["u_g"] = df["u"] - df["g"]
df["g_r"] = df["g"] - df["r"]
df["r_i"] = df["r"] - df["i"]
df["i_z"] = df["i"] - df["z"]

# B. Non-adjacent color indices (wide spectral baselines)
df["u_r"] = df["u"] - df["r"]
df["u_z"] = df["u"] - df["z"]
df["g_z"] = df["g"] - df["z"]

# C. Spectral Curvatures (second differences)
df["curvature_ugr"] = df["u_g"] - df["g_r"]
df["curvature_gri"] = df["g_r"] - df["r_i"]

# D. Aggregate magnitude indicators
raw_bands = ["u", "g", "r", "i", "z"]
df["mag_mean"] = df[raw_bands].mean(axis=1)
df["mag_std"] = df[raw_bands].std(axis=1)

feature_cols = [
    "u", "g", "r", "i", "z",
    "u_g", "g_r", "r_i", "i_z",
    "u_r", "u_z", "g_z",
    "curvature_ugr", "curvature_gri",
    "mag_mean", "mag_std"
]

print(f"Engineered {len(feature_cols)} photometric features.")

# 3. Principal Component Analysis (PCA) - Course Syllabus: Dimensionality Reduction using PCA and SVD
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[feature_cols])

pca = PCA(n_components=5, random_state=42)
X_pca = pca.fit_transform(X_scaled)
for i in range(3):
    df[f"pca_{i+1}"] = X_pca[:, i]

explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)
print(f"PCA Explained Variance: {explained_variance}")
print(f"Cumulative Explained Variance: {cumulative_variance}")

# 4. Feature Importance & Statistical Tests (ANOVA F-statistic and Mutual Information)
from sklearn.model_selection import train_test_split
sample_df, _ = train_test_split(df, train_size=15000, stratify=df["class"], random_state=42)
y_sample = sample_df["class"]
X_sample = sample_df[feature_cols]

f_scores, f_pvalues = f_classif(X_sample, y_sample)
mi_scores = mutual_info_classif(X_sample, y_sample, random_state=42)

feature_stats = pd.DataFrame({
    "Feature": feature_cols,
    "ANOVA_F_Score": f_scores,
    "ANOVA_p_value": f_pvalues,
    "Mutual_Information": mi_scores
}).sort_values(by="Mutual_Information", ascending=False).reset_index(drop=True)

print("\nTop 10 Features by Mutual Information:")
print(feature_stats.head(10))

# 5. Visualizations
palette = {"GALAXY": "#2b5c8f", "STAR": "#d97706", "QSO": "#7c3aed"}

# Figure 1: Feature Correlation Matrix
plt.figure(figsize=(12, 10))
corr_matrix = df[feature_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmin=-1, vmax=1, annot=True, fmt=".2f",
            square=True, linewidths=.5, cbar_kws={"shrink": .8})
plt.title("Photometric Feature Correlation Matrix (Phase 4)", fontsize=14, pad=15)
plt.tight_layout()
fig1_path = FIGURES_DIR / "01_feature_correlation_matrix.png"
plt.savefig(fig1_path, dpi=300)
plt.close()
print(f"Saved {fig1_path}")

# Figure 2: Scree Plot & PCA Variance Explained
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
components = np.arange(1, len(explained_variance) + 1)
ax1.bar(components, explained_variance * 100, color="#2b5c8f", alpha=0.8, edgecolor="black")
ax1.plot(components, explained_variance * 100, "o-", color="#d97706", lw=2)
ax1.set_xlabel("Principal Component", fontsize=11)
ax1.set_ylabel("Variance Explained (%)", fontsize=11)
ax1.set_title("PCA Scree Plot (Individual Variance)", fontsize=12)
ax1.set_xticks(components)

ax2.plot(components, cumulative_variance * 100, "s-", color="#059669", lw=2, markersize=8)
ax2.axhline(80, color="gray", linestyle="--", alpha=0.7, label="80% Threshold")
ax2.axhline(90, color="red", linestyle="--", alpha=0.7, label="90% Threshold")
ax2.set_xlabel("Number of Principal Components", fontsize=11)
ax2.set_ylabel("Cumulative Variance Explained (%)", fontsize=11)
ax2.set_title("Cumulative Explained Variance", fontsize=12)
ax2.set_xticks(components)
ax2.legend()
plt.tight_layout()
fig2_path = FIGURES_DIR / "02_pca_scree_plot.png"
plt.savefig(fig2_path, dpi=300)
plt.close()
print(f"Saved {fig2_path}")

# Figure 3: PCA 2D Projection by Celestial Class
plt.figure(figsize=(9, 7))
sns.scatterplot(
    data=df.sample(6000, random_state=42),
    x="pca_1", y="pca_2", hue="class",
    palette=palette, alpha=0.45, s=20, edgecolor="none"
)
plt.title("SDSS Object Classes in 2D PCA Space (PC1 vs PC2)", fontsize=13)
plt.xlabel(f"Principal Component 1 ({explained_variance[0]*100:.1f}% Variance)")
plt.ylabel(f"Principal Component 2 ({explained_variance[1]*100:.1f}% Variance)")
plt.xlim(-8, 8)
plt.ylim(-6, 6)
plt.legend(title="Astronomical Class", frameon=True)
plt.tight_layout()
fig3_path = FIGURES_DIR / "03_pca_2d_projection.png"
plt.savefig(fig3_path, dpi=300)
plt.close()
print(f"Saved {fig3_path}")

# Figure 4: Top Features Distribution by Class
top_features = ["u_g", "g_r", "r_i", "u_r"]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
for idx, feat in enumerate(top_features):
    ax = axes[idx]
    for cls, color in palette.items():
        subset = df[df["class"] == cls][feat]
        # Filter extreme outlier view range for clear distribution comparison
        q01, q99 = subset.quantile(0.01), subset.quantile(0.99)
        sns.kdeplot(subset[(subset >= q01) & (subset <= q99)], ax=ax, label=cls, color=color, lw=2, fill=True, alpha=0.15)
    ax.set_title(f"Class-Wise Distribution of Color Index '{feat}'", fontsize=12)
    ax.set_xlabel(feat)
    ax.set_ylabel("Kernel Density")
    ax.legend(title="Class")
plt.tight_layout()
fig4_path = FIGURES_DIR / "04_engineered_colors_density.png"
plt.savefig(fig4_path, dpi=300)
plt.close()
print(f"Saved {fig4_path}")

# Figure 5: Feature Importance Ranking (Mutual Information)
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_stats, x="Mutual_Information", y="Feature", hue="Feature", palette="Blues_r", legend=False)
plt.title("Feature Importance by Mutual Information (Photometric Attributes)", fontsize=13)
plt.xlabel("Mutual Information with Class Label")
plt.ylabel("Engineered Feature")
plt.tight_layout()
fig5_path = FIGURES_DIR / "05_mutual_information_ranking.png"
plt.savefig(fig5_path, dpi=300)
plt.close()
print(f"Saved {fig5_path}")

# 6. Save Processed Feature Dataset & Reports
export_features_path = DATA_DIR / "processed" / "sdss_features.csv"
print(f"Saving engineered feature dataset to {export_features_path}...")
export_cols = ["obj_ID", "alpha", "delta"] + feature_cols + ["pca_1", "pca_2", "pca_3", "redshift", "class"]
df[export_cols].to_csv(export_features_path, index=False)
print("Saved successfully.")

feature_stats.to_csv(REPORTS_DIR / "phase4_feature_summary.csv", index=False)

phase4_report = {
    "num_observations": len(df),
    "num_features_engineered": len(feature_cols),
    "feature_list": feature_cols,
    "pca_components": 5,
    "pca_explained_variance_ratio": [float(round(v, 4)) for v in explained_variance],
    "pca_cumulative_variance_ratio": [float(round(v, 4)) for v in cumulative_variance],
    "top_3_features_by_mi": list(feature_stats["Feature"].head(3)),
    "output_file": str(export_features_path)
}
with open(REPORTS_DIR / "phase4_feature_engineering_report.json", "w", encoding="utf-8") as f:
    json.dump(phase4_report, f, indent=4)

print("Phase 4 Feature Engineering complete!")
