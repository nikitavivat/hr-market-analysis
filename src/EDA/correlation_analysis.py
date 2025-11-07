import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

print("=" * 70)
print("АНАЛИЗ КОРРЕЛЯЦИЙ")
print("=" * 70)

df = pd.read_csv('merged_data.csv')
print(f"\nЗагружено записей: {len(df)}")

print("\n=== Подготовка признаков для корреляционного анализа ===")

df['description_length'] = df['Описание вакансии'].fillna('').str.len()
df['requirements_length'] = df['Требования'].fillna('').str.len()
df['title_length'] = df['Название'].fillna('').str.len()
df['description_words'] = df['Описание вакансии'].fillna('').str.split().str.len()
df['requirements_words'] = df['Требования'].fillna('').str.split().str.len()

df['has_requirements'] = (df['Требования'].notna() & (df['Требования'] != '')).astype(int)
df['has_skills'] = (df['Навыки'].notna() & (df['Навыки'] != '') & (df['Навыки'] != 'Не указаны')).astype(int)

df['work_type_remote'] = df['Тип работы'].fillna('').str.contains('Удаленная|Remote|remote', na=False).astype(int)
df['work_type_full'] = df['Тип работы'].fillna('').str.contains('Полная|Full', na=False).astype(int)
df['work_type_partial'] = df['Тип работы'].fillna('').str.contains('Частичная|Part', na=False).astype(int)

df['city_moscow'] = df['Город'].fillna('').str.contains('Москва|Moscow', na=False).astype(int)
df['city_spb'] = df['Город'].fillna('').str.contains('Санкт-Петербург|Петербург|SPb|St. Petersburg', na=False).astype(int)

if 'transparency_score' not in df.columns:
    df['transparency_score'] = 0
    df.loc[df['salary_target'].notna(), 'transparency_score'] += 1
    df.loc[df['Требования'].notna() & (df['Требования'] != ''), 'transparency_score'] += 1
    df.loc[df['Навыки'].notna() & (df['Навыки'] != '') & (df['Навыки'] != 'Не указаны'), 'transparency_score'] += 1
    df.loc[df['Количество откликов'].notna(), 'transparency_score'] += 1

numeric_features = [
    'salary_target',
    'Зарплата_от',
    'Зарплата_до',
    'Количество откликов',
    'description_length',
    'requirements_length',
    'title_length',
    'description_words',
    'requirements_words',
    'has_requirements',
    'has_skills',
    'work_type_remote',
    'work_type_full',
    'work_type_partial',
    'city_moscow',
    'city_spb',
    'transparency_score'
]

if 'cluster_kmeans' in df.columns:
    numeric_features.append('cluster_kmeans')
if 'cluster_dbscan' in df.columns:
    numeric_features.append('cluster_dbscan')

df_numeric = df[numeric_features].copy()

print(f"\nЧисленных признаков: {len(numeric_features)}")
print(f"Признаки: {numeric_features}")

print("\n=== Расчет корреляционной матрицы ===")
correlation_matrix = df_numeric.corr(method='pearson')

print("\n=== Корреляция с целевой переменной (salary_target) ===")
salary_correlations = correlation_matrix['salary_target'].sort_values(ascending=False)
print("\nТоп-15 признаков, наиболее коррелированных с зарплатой:")
print(salary_correlations.head(15))
print("\nТоп-15 признаков, наименее коррелированных с зарплатой:")
print(salary_correlations.tail(15))

print("\n=== Сильные корреляции (|r| > 0.3) ===")
strong_correlations = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr_value = correlation_matrix.iloc[i, j]
        if abs(corr_value) > 0.3 and not np.isnan(corr_value):
            strong_correlations.append({
                'Признак 1': correlation_matrix.columns[i],
                'Признак 2': correlation_matrix.columns[j],
                'Корреляция': corr_value
            })

if strong_correlations:
    strong_corr_df = pd.DataFrame(strong_correlations).sort_values('Корреляция', key=abs, ascending=False)
    print(strong_corr_df.to_string(index=False))
else:
    print("Сильных корреляций не найдено")

print("\n=== Визуализация корреляционной матрицы ===")

fig, axes = plt.subplots(2, 2, figsize=(20, 18))

mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=axes[0, 0])
axes[0, 0].set_title('Корреляционная матрица (верхний треугольник)', fontsize=14, fontweight='bold')
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].tick_params(axis='y', rotation=0)

sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=axes[0, 1])
axes[0, 1].set_title('Полная корреляционная матрица', fontsize=14, fontweight='bold')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].tick_params(axis='y', rotation=0)

salary_correlations_sorted = salary_correlations.drop('salary_target').sort_values(ascending=True)
salary_correlations_sorted.plot(kind='barh', ax=axes[1, 0], color='steelblue')
axes[1, 0].set_xlabel('Корреляция с зарплатой', fontsize=12)
axes[1, 0].set_title('Корреляция всех признаков с зарплатой', fontsize=14, fontweight='bold')
axes[1, 0].axvline(x=0, color='black', linestyle='--', linewidth=0.8)
axes[1, 0].grid(axis='x', alpha=0.3)

top_correlations = salary_correlations.drop('salary_target').abs().sort_values(ascending=False).head(10)
top_correlations.plot(kind='barh', ax=axes[1, 1], color='coral')
axes[1, 1].set_xlabel('Абсолютная корреляция с зарплатой', fontsize=12)
axes[1, 1].set_title('Топ-10 признаков по силе корреляции с зарплатой', fontsize=14, fontweight='bold')
axes[1, 1].invert_yaxis()

plt.tight_layout()
plt.savefig('plots/07_correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("График сохранен: plots/07_correlation_matrix.png")

print("\n=== Детальная визуализация сильных корреляций ===")

if strong_correlations:
    top_pairs = pd.DataFrame(strong_correlations).sort_values('Корреляция', key=abs, ascending=False).head(6)
    
    n_pairs = len(top_pairs)
    n_cols = 3
    n_rows = (n_pairs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
    if n_pairs == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if isinstance(axes, np.ndarray) else [axes]
    else:
        axes = axes.flatten()
    
    for idx, (_, row) in enumerate(top_pairs.iterrows()):
        feat1 = row['Признак 1']
        feat2 = row['Признак 2']
        corr_val = row['Корреляция']
        
        valid_mask = df_numeric[feat1].notna() & df_numeric[feat2].notna()
        if valid_mask.sum() > 100:
            sample_size = min(5000, valid_mask.sum())
            sample_indices = df_numeric[valid_mask].sample(n=sample_size, random_state=42).index
            
            axes[idx].scatter(df_numeric.loc[sample_indices, feat1], 
                            df_numeric.loc[sample_indices, feat2], 
                            alpha=0.3, s=10)
            axes[idx].set_xlabel(feat1, fontsize=10)
            axes[idx].set_ylabel(feat2, fontsize=10)
            axes[idx].set_title(f'Корреляция: {corr_val:.3f}', fontsize=12, fontweight='bold')
            axes[idx].grid(alpha=0.3)
    
    for idx in range(len(top_pairs), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('plots/08_correlation_scatter_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("График сохранен: plots/08_correlation_scatter_plots.png")

print("\n=== Корреляция по источникам данных ===")
if 'Источник' in df.columns:
    sources = df['Источник'].unique()
    print(f"\nИсточники данных: {sources}")
    
    fig, axes = plt.subplots(1, len(sources), figsize=(6*len(sources), 5))
    if len(sources) == 1:
        axes = [axes]
    
    for idx, source in enumerate(sources):
        df_source = df[df['Источник'] == source]
        df_source_numeric = df_source[numeric_features].copy()
        corr_source = df_source_numeric.corr(method='pearson')
        
        sns.heatmap(corr_source, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=axes[idx])
        axes[idx].set_title(f'Корреляция: {source}', fontsize=12, fontweight='bold')
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].tick_params(axis='y', rotation=0)
    
    plt.tight_layout()
    plt.savefig('plots/09_correlation_by_source.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("График сохранен: plots/09_correlation_by_source.png")
    
    print("\nКорреляция с зарплатой по источникам:")
    for source in sources:
        df_source = df[df['Источник'] == source]
        df_source_numeric = df_source[numeric_features].copy()
        if 'salary_target' in df_source_numeric.columns:
            corr_source = df_source_numeric.corr(method='pearson')
            salary_corr_source = corr_source['salary_target'].sort_values(ascending=False)
            print(f"\n{source}:")
            print(salary_corr_source.head(10))

print("\n=== Сохранение результатов ===")
correlation_matrix.to_csv('plots/correlation_matrix.csv', encoding='utf-8')
print("Корреляционная матрица сохранена: plots/correlation_matrix.csv")

salary_correlations_df = pd.DataFrame({
    'Признак': salary_correlations.index,
    'Корреляция_с_зарплатой': salary_correlations.values
}).sort_values('Корреляция_с_зарплатой', ascending=False)
salary_correlations_df.to_csv('plots/salary_correlations.csv', index=False, encoding='utf-8')
print("Корреляции с зарплатой сохранены: plots/salary_correlations.csv")

if strong_correlations:
    strong_corr_df.to_csv('plots/strong_correlations.csv', index=False, encoding='utf-8')
    print("Сильные корреляции сохранены: plots/strong_correlations.csv")

print("\n" + "=" * 70)
print("АНАЛИЗ КОРРЕЛЯЦИЙ ЗАВЕРШЕН")
print("=" * 70)
print("\nСозданные файлы:")
print("  - plots/07_correlation_matrix.png - корреляционная матрица")
print("  - plots/08_correlation_scatter_plots.png - графики сильных корреляций")
print("  - plots/09_correlation_by_source.png - корреляции по источникам")
print("  - plots/correlation_matrix.csv - полная матрица корреляций")
print("  - plots/salary_correlations.csv - корреляции с зарплатой")
print("  - plots/strong_correlations.csv - сильные корреляции")

