import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (16, 10)
sns.set_style("whitegrid")

print("=== Загрузка данных ===")
embeddings = np.load('embeddings_final.npy')
indices = np.load('embeddings_sample_indices.npy')

df = pd.read_csv('merged_data.csv')
df_sample = df.iloc[indices].copy()

print(f"Загружено: {len(df_sample)} записей")
print(f"Эмбеддинги: {embeddings.shape}")

df_with_salary = df_sample[df_sample['salary_target'].notna()].copy()
embeddings_with_salary = embeddings[df_sample['salary_target'].notna()]

print(f"С зарплатой: {len(df_with_salary)} записей")

print("\n=== Определение сложности на основе эмбеддингов ===")

pca = PCA(n_components=1)
complexity_score = pca.fit_transform(embeddings_with_salary).flatten()

scaler_complexity = StandardScaler()
complexity_score = scaler_complexity.fit_transform(complexity_score.reshape(-1, 1)).flatten()

df_with_salary['complexity_score'] = complexity_score

print(f"Сложность определена. Диапазон: {complexity_score.min():.2f} до {complexity_score.max():.2f}")

print("\n=== Создание графиков ===")

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

axes[0, 0].scatter(df_with_salary['complexity_score'], df_with_salary['salary_target'], 
                   alpha=0.3, s=10, edgecolors='none')
axes[0, 0].set_xlabel('Сложность требований (на основе эмбеддингов)')
axes[0, 0].set_ylabel('Зарплата (руб.)')
axes[0, 0].set_title('Зависимость зарплаты от сложности требований')
axes[0, 0].grid(True, alpha=0.3)

z = np.polyfit(df_with_salary['complexity_score'], df_with_salary['salary_target'], 1)
p = np.poly1d(z)
axes[0, 0].plot(df_with_salary['complexity_score'].sort_values(), 
                p(df_with_salary['complexity_score'].sort_values()), 
                "r--", alpha=0.8, linewidth=2, label='Тренд')
axes[0, 0].legend()

axes[0, 1].hexbin(df_with_salary['complexity_score'], df_with_salary['salary_target'], 
                  gridsize=30, cmap='YlOrRd')
axes[0, 1].set_xlabel('Сложность требований')
axes[0, 1].set_ylabel('Зарплата (руб.)')
axes[0, 1].set_title('Плотность распределения (Hexbin)')
axes[0, 1].grid(True, alpha=0.3)

complexity_bins = pd.qcut(df_with_salary['complexity_score'], q=10, duplicates='drop')
salary_by_complexity = df_with_salary.groupby(complexity_bins)['salary_target'].agg(['mean', 'median', 'count'])

axes[0, 2].plot(range(len(salary_by_complexity)), salary_by_complexity['mean'], 
                marker='o', linewidth=2, markersize=8, label='Средняя')
axes[0, 2].plot(range(len(salary_by_complexity)), salary_by_complexity['median'], 
                marker='s', linewidth=2, markersize=8, label='Медианная')
axes[0, 2].set_xlabel('Децили сложности (от низкой к высокой)')
axes[0, 2].set_ylabel('Зарплата (руб.)')
axes[0, 2].set_title('Зарплата по децилям сложности')
axes[0, 2].set_xticks(range(len(salary_by_complexity)))
axes[0, 2].set_xticklabels([f'{i+1}' for i in range(len(salary_by_complexity))])
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

axes[1, 0].hist2d(df_with_salary['complexity_score'], df_with_salary['salary_target'], 
                  bins=50, cmap='Blues')
axes[1, 0].set_xlabel('Сложность требований')
axes[1, 0].set_ylabel('Зарплата (руб.)')
axes[1, 0].set_title('2D гистограмма')
axes[1, 0].grid(True, alpha=0.3)

low_complexity = df_with_salary[df_with_salary['complexity_score'] < df_with_salary['complexity_score'].quantile(0.33)]
mid_complexity = df_with_salary[(df_with_salary['complexity_score'] >= df_with_salary['complexity_score'].quantile(0.33)) & 
                                 (df_with_salary['complexity_score'] < df_with_salary['complexity_score'].quantile(0.67))]
high_complexity = df_with_salary[df_with_salary['complexity_score'] >= df_with_salary['complexity_score'].quantile(0.67)]

data_to_plot = [low_complexity['salary_target'].values, 
                mid_complexity['salary_target'].values,
                high_complexity['salary_target'].values]

bp = axes[1, 1].boxplot(data_to_plot, labels=['Низкая', 'Средняя', 'Высокая'], patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
axes[1, 1].set_ylabel('Зарплата (руб.)')
axes[1, 1].set_xlabel('Сложность требований')
axes[1, 1].set_title('Распределение зарплат по уровням сложности')
axes[1, 1].grid(True, alpha=0.3, axis='y')

from scipy import stats
correlation = stats.pearsonr(df_with_salary['complexity_score'], df_with_salary['salary_target'])[0]

axes[1, 2].text(0.5, 0.7, f'Корреляция:\n{correlation:.4f}', 
                transform=axes[1, 2].transAxes, fontsize=16, 
                ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

stats_text = f"""
Статистика по сложности:

Низкая сложность (33%):
  Средняя ЗП: {low_complexity['salary_target'].mean():.0f} руб.
  Медианная ЗП: {low_complexity['salary_target'].median():.0f} руб.
  Записей: {len(low_complexity)}

Средняя сложность (34%):
  Средняя ЗП: {mid_complexity['salary_target'].mean():.0f} руб.
  Медианная ЗП: {mid_complexity['salary_target'].median():.0f} руб.
  Записей: {len(mid_complexity)}

Высокая сложность (33%):
  Средняя ЗП: {high_complexity['salary_target'].mean():.0f} руб.
  Медианная ЗП: {high_complexity['salary_target'].median():.0f} руб.
  Записей: {len(high_complexity)}
"""

axes[1, 2].text(0.5, 0.3, stats_text, transform=axes[1, 2].transAxes,
                fontsize=10, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
axes[1, 2].axis('off')

plt.suptitle('Анализ зависимости зарплаты от сложности требований\n(на основе эмбеддингов)', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/21_complexity_salary.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ plots/21_complexity_salary.png")

print("\n=== Дополнительный анализ ===")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

pca_full = PCA(n_components=2)
complexity_2d = pca_full.fit_transform(embeddings_with_salary)

scatter = axes[0].scatter(complexity_2d[:, 0], complexity_2d[:, 1], 
                         c=df_with_salary['salary_target'], 
                         cmap='viridis', alpha=0.5, s=10)
axes[0].set_xlabel('PC1 (основная компонента сложности)')
axes[0].set_ylabel('PC2')
axes[0].set_title('2D визуализация сложности (цвет = зарплата)')
plt.colorbar(scatter, ax=axes[0], label='Зарплата (руб.)')
axes[0].grid(True, alpha=0.3)

salary_by_complexity_bins = df_with_salary.groupby(complexity_bins)['salary_target'].mean()
axes[1].bar(range(len(salary_by_complexity_bins)), salary_by_complexity_bins.values, 
           color='steelblue', edgecolor='black')
axes[1].set_xlabel('Децили сложности (от низкой к высокой)')
axes[1].set_ylabel('Средняя зарплата (руб.)')
axes[1].set_title('Средняя зарплата по децилям сложности')
axes[1].set_xticks(range(len(salary_by_complexity_bins)))
axes[1].set_xticklabels([f'{i+1}' for i in range(len(salary_by_complexity_bins))], rotation=45)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('plots/22_complexity_salary_additional.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ plots/22_complexity_salary_additional.png")

print("\n" + "="*60)
print("Анализ завершен!")
print("="*60)
print(f"\nКорреляция сложности и зарплаты: {correlation:.4f}")
print(f"\nСозданные графики:")
print("  ✓ plots/21_complexity_salary.png - основной анализ")
print("  ✓ plots/22_complexity_salary_additional.png - дополнительный анализ")
print("="*60)

