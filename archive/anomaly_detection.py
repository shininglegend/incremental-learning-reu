# From colab. Most code generated by AI under supervision of Titus
# -*- coding: utf-8 -*-
"""Anomaly Detection.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/109OF1JF_L_oizkcqvqi1QktiHyqddbDh
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.cluster import HDBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE


# Define columns that should be strings
string_columns = [
    'DESYNPUF_ID', 'PRVDR_NUM', 'ADMTNG_ICD9_DGNS_CD',
    'ICD9_DGNS_CD_1', 'ICD9_DGNS_CD_2', 'ICD9_DGNS_CD_3',
    'ICD9_DGNS_CD_4', 'ICD9_DGNS_CD_5', 'ICD9_DGNS_CD_6',
    'ICD9_DGNS_CD_7', 'ICD9_DGNS_CD_8', 'ICD9_DGNS_CD_9',
    'ICD9_DGNS_CD_10', 'ICD9_PRCDR_CD_1', 'ICD9_PRCDR_CD_2',
    'ICD9_PRCDR_CD_3', 'ICD9_PRCDR_CD_4', 'ICD9_PRCDR_CD_5',
    'ICD9_PRCDR_CD_6'
]

dtype_dict = {col: 'string' for col in string_columns}

df = pd.read_csv('in.csv', dtype=dtype_dict, nrows=10000)

print(df.shape) # Show what the data shape (x, y) is
# print(df.head()) # Show the first 5 rows
# print(df.isnull().sum()) # Show which cols have null values
print(df.info()) # Summary
print(df['ICD9_DGNS_CD_1'].sample(10))

# Define all ICD diagnostic and procedure code columns
icd_columns = [
   'ADMTNG_ICD9_DGNS_CD',
   'ICD9_DGNS_CD_1', 'ICD9_DGNS_CD_2', 'ICD9_DGNS_CD_3',
   'ICD9_DGNS_CD_4', 'ICD9_DGNS_CD_5', 'ICD9_DGNS_CD_6',
   'ICD9_DGNS_CD_7', 'ICD9_DGNS_CD_8', 'ICD9_DGNS_CD_9',
   'ICD9_DGNS_CD_10', 'ICD9_PRCDR_CD_1', 'ICD9_PRCDR_CD_2',
   'ICD9_PRCDR_CD_3', 'ICD9_PRCDR_CD_4', 'ICD9_PRCDR_CD_5',
   'ICD9_PRCDR_CD_6'
]

# Fill NaN values with 'missing' before encoding
df[icd_columns] = df[icd_columns].fillna('missing')

# One-hot encode all columns at once - keep sparse format for memory efficiency
encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore', dtype=np.float32)
encoded_sparse = encoder.fit_transform(df[icd_columns])

# Create sparse DataFrame with proper column names
feature_names = encoder.get_feature_names_out(icd_columns)
encoded_df = pd.DataFrame.sparse.from_spmatrix(encoded_sparse, columns=feature_names, index=df.index)

# Drop original ICD columns and concatenate encoded columns
df_processed = df.drop(columns=icd_columns)
df_final = pd.concat([df_processed, encoded_df], axis=1)

# Show
print(df_final.shape)

# Filter features by frequency - use sparse-aware operations
min_frequency = len(df) * 0.01
column_sums = np.array(encoded_sparse.sum(axis=0)).flatten()  # More memory efficient
frequent_mask = column_sums >= min_frequency
frequent_features = feature_names[frequent_mask]
df_filtered = encoded_df[frequent_features]
print(df_filtered.shape)

# 1. Check data sparsity - use sparse-aware calculation
sparse_matrix = df_filtered.sparse.to_coo()
total_elements = df_filtered.shape[0] * df_filtered.shape[1]
non_zero_elements = sparse_matrix.nnz
sparsity = (total_elements - non_zero_elements) / total_elements
print(f"Data sparsity: {sparsity:.3f}")

# 2. Convert to dense only what's needed for scaling - use sparse-compatible scaler
sparse_data = sparse.csr_matrix(df_filtered.sparse.to_coo())
scaler = StandardScaler(with_mean=False)
df_scaled = scaler.fit_transform(sparse_data).astype(np.float32)

# 3. HDBSCAN clustering with appropriate parameters for binary/sparse data
hdb = HDBSCAN(
    min_cluster_size=3,        # Minimum 8 patients per cluster
    min_samples=2,              # Core point threshold (5)
    metric='cosine',           # Better for sparse high-dimensional data than hamming
    cluster_selection_epsilon=0.1,  # For more stable clusters
    cluster_selection_method='leaf',
    n_jobs=-1
)

# 4. Fit clustering
cluster_labels = hdb.fit_predict(df_scaled)

# 5. Evaluate results
n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
n_noise = list(cluster_labels).count(-1)

print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points (potential anomalies): {n_noise}")
print(f"Percentage of data as noise: {n_noise/len(cluster_labels)*100:.1f}%")

# 6. Silhouette score (if clusters found)
if n_clusters > 1:
    # Exclude noise points for silhouette calculation
    mask = cluster_labels != -1
    if mask.sum() > 0:
        sil_score = silhouette_score(df_scaled[mask], cluster_labels[mask])
        print(f"Silhouette score: {sil_score:.3f}")

# # 7. Analyze cluster strengths
# cluster_strengths = clusterer.cluster_persistence_
# if cluster_strengths is not None:
#     print("Cluster persistence scores:", cluster_strengths)

# 8. Identify anomalies (noise points)
anomaly_indices = np.where(cluster_labels == -1)[0]
print(f"Anomaly patient indices: {anomaly_indices[:20]}...")  # Show first 20

# 1. PCA for 2D visualization - handle sparse input

# Use TruncatedSVD which works better with sparse data
pca = TruncatedSVD(n_components=2, random_state=42)
pca_result = pca.fit_transform(df_scaled).astype(np.float32)

# 2. Create visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: PCA scatter
scatter1 = axes[0,0].scatter(pca_result[:, 0], pca_result[:, 1],
                            c=cluster_labels, cmap='tab10', alpha=0.7)
axes[0,0].set_title('PCA Projection')
axes[0,0].set_xlabel(f'SVD1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
axes[0,0].set_ylabel(f'SVD2 ({pca.explained_variance_ratio_[1]:.2%} variance)')

# Plot 2: t-SNE (if data not too large) - convert sparse to dense only for small datasets
if df_scaled.shape[0] <= 1000:
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_result = tsne.fit_transform(df_scaled.toarray().astype(np.float32))  # Convert to dense for t-SNE
    scatter2 = axes[0,1].scatter(tsne_result[:, 0], tsne_result[:, 1],
                                c=cluster_labels, cmap='tab10', alpha=0.7)
    axes[0,1].set_title('t-SNE Projection')
else:
    axes[0,1].text(0.5, 0.5, 'Dataset too large for t-SNE',
                   ha='center', va='center', transform=axes[0,1].transAxes)

# Plot 3: Cluster size distribution
cluster_counts = np.bincount(cluster_labels[cluster_labels >= 0])
axes[1,0].bar(range(len(cluster_counts)), cluster_counts)
axes[1,0].set_title('Cluster Sizes')
axes[1,0].set_xlabel('Cluster ID')
axes[1,0].set_ylabel('Number of Patients')

# Plot 4: Anomaly highlighting
anomaly_mask = cluster_labels == -1
axes[1,1].scatter(pca_result[~anomaly_mask, 0], pca_result[~anomaly_mask, 1],
                 c='lightblue', alpha=0.5, label='Normal')
axes[1,1].scatter(pca_result[anomaly_mask, 0], pca_result[anomaly_mask, 1],
                 c='red', s=100, alpha=0.8, label='Anomalies')
axes[1,1].set_title('Anomaly Detection')
axes[1,1].legend()

plt.tight_layout()
plt.show()

# Summary stats
print(f"Clusters found: {len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)}")
print(f"Anomalies: {sum(cluster_labels == -1)} ({sum(cluster_labels == -1)/len(cluster_labels)*100:.1f}%)")
print(f"Total variance explained by SVD1+SVD2: {sum(pca.explained_variance_ratio_):.2%}")
