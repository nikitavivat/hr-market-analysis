import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
import time
import warnings
warnings.filterwarnings('ignore')
from config import gpt_api_key

plt.rcParams['figure.figsize'] = (12, 8)
sns.set_style("whitegrid")

df = pd.read_csv('merged_data.csv')

client = OpenAI(api_key=gpt_api_key)

def get_embedding(text, model="text-embedding-3-small"):
    if not text or len(text.strip()) == 0:
        return None
    try:
        text = text.replace("\n", " ")[:8000]
        response = client.embeddings.create(input=[text], model=model)
        return response.data[0].embedding
    except Exception as e:
        print(f"Error: {e}")
        return None

print(f"Всего записей: {df.shape[0]}")

df['description_text'] = df['Описание вакансии'].fillna('') + ' ' + df['Требования'].fillna('')
df['description_length'] = df['description_text'].str.len()
df['description_words'] = df['description_text'].str.split().str.len()

print(f"Средняя длина описания: {df['description_length'].mean():.0f} символов")
print(f"Среднее количество слов: {df['description_words'].mean():.0f}")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

df['description_length'].hist(bins=50, ax=axes[0, 0])
axes[0, 0].set_xlabel('Длина описания (символы)')
axes[0, 0].set_ylabel('Частота')
axes[0, 0].set_title('Распределение длины описаний')

df['description_words'].hist(bins=50, ax=axes[0, 1])
axes[0, 1].set_xlabel('Количество слов')
axes[0, 1].set_ylabel('Частота')
axes[0, 1].set_title('Распределение количества слов')

if df['salary_target'].notna().sum() > 0:
    df_with_salary = df[df['salary_target'].notna()]
    axes[1, 0].scatter(df_with_salary['description_length'], df_with_salary['salary_target'], alpha=0.1)
    axes[1, 0].set_xlabel('Длина описания')
    axes[1, 0].set_ylabel('Зарплата')
    axes[1, 0].set_title('Зависимость зарплаты от длины описания')
    
    axes[1, 1].scatter(df_with_salary['description_words'], df_with_salary['salary_target'], alpha=0.1)
    axes[1, 1].set_xlabel('Количество слов')
    axes[1, 1].set_ylabel('Зарплата')
    axes[1, 1].set_title('Зависимость зарплаты от количества слов')

plt.tight_layout()
plt.savefig('plots/07_nlp_basic_stats.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n========= Векторизация ==========")
sample_size = min(1000, len(df))
df_sample = df.sample(n=sample_size, random_state=42).copy()

print(f"Векторизация {sample_size} записей")
embeddings = []
for idx, row in df_sample.iterrows():
    embedding = get_embedding(row['description_text'])
    embeddings.append(embedding)
    if (len(embeddings) % 100) == 0:
        print(f"Обработано: {len(embeddings)}/{sample_size}")
    time.sleep(0.1)

df_sample = df_sample[df_sample.index.isin([i for i, e in enumerate(embeddings) if e is not None])]
embeddings = [e for e in embeddings if e is not None]

if len(embeddings) > 0:
    embeddings_array = np.array(embeddings)
    print(f"Размерность эмбеддингов: {embeddings_array.shape}")
    
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    print("\nПрименение PCA")
    pca = PCA(n_components=2)
    embeddings_2d_pca = pca.fit_transform(embeddings_array)
    print(f"Объясненная дисперсия: {pca.explained_variance_ratio_.sum():.4f}")
    
    print("\nПрименение t-SNE")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d_tsne = tsne.fit_transform(embeddings_array[:500])
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    scatter1 = axes[0, 0].scatter(embeddings_2d_pca[:, 0], embeddings_2d_pca[:, 1], 
                                  c=df_sample.iloc[:len(embeddings_2d_pca)]['salary_target'] if df_sample['salary_target'].notna().sum() > 0 else None,
                                  cmap='viridis', alpha=0.6, s=10)
    axes[0, 0].set_xlabel('PC1')
    axes[0, 0].set_ylabel('PC2')
    axes[0, 0].set_title('PCA визуализация эмбеддингов')
    if df_sample['salary_target'].notna().sum() > 0:
        plt.colorbar(scatter1, ax=axes[0, 0], label='Зарплата')
    
    scatter2 = axes[0, 1].scatter(embeddings_2d_tsne[:, 0], embeddings_2d_tsne[:, 1],
                                  c=df_sample.iloc[:len(embeddings_2d_tsne)]['salary_target'] if df_sample['salary_target'].notna().sum() > 0 else None,
                                  cmap='viridis', alpha=0.6, s=10)
    axes[0, 1].set_xlabel('t-SNE 1')
    axes[0, 1].set_ylabel('t-SNE 2')
    axes[0, 1].set_title('t-SNE визуализация эмбеддингов')
    if df_sample['salary_target'].notna().sum() > 0:
        plt.colorbar(scatter2, ax=axes[0, 1], label='Зарплата')
    
    if df_sample['salary_target'].notna().sum() > 0:
        df_with_salary_sample = df_sample[df_sample['salary_target'].notna()]
        axes[1, 0].hist(df_with_salary_sample['salary_target'], bins=30)
        axes[1, 0].set_xlabel('Зарплата')
        axes[1, 0].set_ylabel('Частота')
        axes[1, 0].set_title('Распределение зарплат в выборке')
        
        salary_by_source = df_sample.groupby('Источник')['salary_target'].mean()
        salary_by_source.plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_ylabel('Средняя зарплата')
        axes[1, 1].set_title('Средняя зарплата по источникам')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('plots/08_nlp_embeddings.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    np.save('embeddings_sample.npy', embeddings_array)
    df_sample.to_csv('df_sample_with_embeddings.csv', index=False, encoding='utf-8')

df.to_csv('merged_data.csv', index=False, encoding='utf-8')

