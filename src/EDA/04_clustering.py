import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (12, 8)
sns.set_style("whitegrid")

df = pd.read_csv('merged_data.csv')

df['description_length'] = df['Описание вакансии'].fillna('').str.len()
df['requirements_length'] = df['Требования'].fillna('').str.len()
df['description_words'] = df['Описание вакансии'].fillna('').str.split().str.len()
df['requirements_words'] = df['Требования'].fillna('').str.split().str.len()

df['has_requirements'] = (df['Требования'].notna() & (df['Требования'] != '')).astype(int)
df['has_skills'] = (df['Навыки'].notna() & (df['Навыки'] != '') & (df['Навыки'] != 'Не указаны')).astype(int)

features_for_clustering = ['description_length', 'requirements_length', 
                          'description_words', 'requirements_words',
                          'has_requirements', 'has_skills']

df_cluster = df[features_for_clustering].copy()
df_cluster = df_cluster.fillna(0)

sample_size = min(50000, len(df_cluster))
if sample_size < len(df_cluster):
    df_cluster_sample = df_cluster.sample(n=sample_size, random_state=42)
    df_indices = df_cluster_sample.index
else:
    df_cluster_sample = df_cluster
    df_indices = df_cluster.index

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cluster_sample)

print(f"Размерность данных для кластеризации: {X_scaled.shape}")

print("\n=== KMeans кластеризация ===")
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
clusters_kmeans = kmeans.fit_predict(X_scaled)
df.loc[df_indices, 'cluster_kmeans'] = clusters_kmeans
df['cluster_kmeans'] = df['cluster_kmeans'].fillna(-1)

print(f"Кластеры KMeans: {np.unique(clusters_kmeans)}")
for i in range(n_clusters):
    cluster_data = df[df['cluster_kmeans'] == i]
    print(f"\nКластер {i}: {len(cluster_data)} вакансий")
    print(f"  Средняя длина описания: {cluster_data['description_length'].mean():.0f}")
    print(f"  Средняя длина требований: {cluster_data['requirements_length'].mean():.0f}")
    if cluster_data['salary_target'].notna().sum() > 0:
        print(f"  Средняя зарплата: {cluster_data['salary_target'].mean():.0f}")

print("\n=== DBSCAN кластеризация ===")
dbscan = DBSCAN(eps=0.5, min_samples=10)
clusters_dbscan = dbscan.fit_predict(X_scaled)
df.loc[df_indices, 'cluster_dbscan'] = clusters_dbscan
df['cluster_dbscan'] = df['cluster_dbscan'].fillna(-1)

unique_clusters = np.unique(clusters_dbscan)
print(f"Кластеры DBSCAN: {len(unique_clusters)} (включая шум: -1)")
for i in unique_clusters:
    cluster_data = df[df['cluster_dbscan'] == i]
    print(f"\nКластер {i}: {len(cluster_data)} вакансий")
    if len(cluster_data) > 0:
        print(f"  Средняя длина описания: {cluster_data['description_length'].mean():.0f}")
        print(f"  Средняя длина требований: {cluster_data['requirements_length'].mean():.0f}")
        if cluster_data['salary_target'].notna().sum() > 0:
            print(f"  Средняя зарплата: {cluster_data['salary_target'].mean():.0f}")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, axes = plt.subplots(2, 2, figsize=(16, 16))

scatter1 = axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters_kmeans, 
                              cmap='viridis', alpha=0.6, s=10)
axes[0, 0].set_xlabel('PC1')
axes[0, 0].set_ylabel('PC2')
axes[0, 0].set_title('KMeans кластеризация')
plt.colorbar(scatter1, ax=axes[0, 0])

scatter2 = axes[0, 1].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters_dbscan, 
                              cmap='Set3', alpha=0.6, s=10)
axes[0, 1].set_xlabel('PC1')
axes[0, 1].set_ylabel('PC2')
axes[0, 1].set_title('DBSCAN кластеризация')
plt.colorbar(scatter2, ax=axes[0, 1])

if df['salary_target'].notna().sum() > 0:
    df_with_salary_mask = df.loc[df_indices, 'salary_target'].notna()
    if df_with_salary_mask.sum() > 0:
        scatter3 = axes[1, 0].scatter(X_pca[df_with_salary_mask, 0], 
                                     X_pca[df_with_salary_mask, 1], 
                                     c=df.loc[df_indices[df_with_salary_mask], 'salary_target'], 
                                     cmap='plasma', alpha=0.6, s=10)
    axes[1, 0].set_xlabel('PC1')
    axes[1, 0].set_ylabel('PC2')
    axes[1, 0].set_title('Распределение зарплат')
    plt.colorbar(scatter3, ax=axes[1, 0], label='Зарплата')
    
    cluster_salary = df.groupby('cluster_kmeans')['salary_target'].mean()
    cluster_salary.plot(kind='bar', ax=axes[1, 1])
    axes[1, 1].set_xlabel('Кластер')
    axes[1, 1].set_ylabel('Средняя зарплата')
    axes[1, 1].set_title('Средняя зарплата по кластерам')
    axes[1, 1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig('plots/09_clustering.png', dpi=300, bbox_inches='tight')
plt.close()

df.to_csv('merged_data.csv', index=False, encoding='utf-8')

