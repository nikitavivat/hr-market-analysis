import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

df = pd.read_csv('merged_data.csv')

print(f"Размер датасета: {df.shape}")
print(f"Колонки: {df.columns.tolist()}")

missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({'Пропущено': missing, 'Процент': missing_pct})
missing_df = missing_df[missing_df['Пропущено'] > 0].sort_values('Пропущено', ascending=False)
print(missing_df)

fig, ax = plt.subplots(figsize=(10, 6))
missing_df['Процент'].plot(kind='barh', ax=ax)
ax.set_xlabel('Процент пропусков')
ax.set_title('Распределение пропусков в данных')
plt.tight_layout()
plt.savefig('plots/01_missing_values.png', dpi=300, bbox_inches='tight')
plt.close()

if df['salary_target'].notna().sum() > 0:
    salary_stats = df['salary_target'].describe()
    print(salary_stats)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    df[df['salary_target'].notna()]['salary_target'].hist(bins=50, ax=axes[0, 0])
    axes[0, 0].set_xlabel('Зарплата')
    axes[0, 0].set_ylabel('Частота')
    axes[0, 0].set_title('Распределение зарплат')
    
    df[df['salary_target'].notna()]['salary_target'].plot(kind='box', ax=axes[0, 1])
    axes[0, 1].set_ylabel('Зарплата')
    axes[0, 1].set_title('Boxplot зарплат')
    
    log_salary = np.log1p(df[df['salary_target'].notna()]['salary_target'])
    log_salary.hist(bins=50, ax=axes[1, 0])
    axes[1, 0].set_xlabel('log(Зарплата)')
    axes[1, 0].set_ylabel('Частота')
    axes[1, 0].set_title('Распределение логарифма зарплат')
    
    df[df['salary_target'].notna()].groupby('Источник')['salary_target'].mean().plot(kind='bar', ax=axes[1, 1])
    axes[1, 1].set_ylabel('Средняя зарплата')
    axes[1, 1].set_title('Средняя зарплата по источникам')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('plots/02_salary_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

city_stats = df.groupby('Город').agg({
    'salary_target': ['count', 'mean', 'median'],
    'Название': 'count'
}).round(0)
city_stats.columns = ['Количество', 'Средняя ЗП', 'Медианная ЗП', 'Всего вакансий']
city_stats = city_stats.sort_values('Количество', ascending=False).head(20)
print(city_stats)

fig, axes = plt.subplots(2, 1, figsize=(14, 12))
city_stats['Количество'].plot(kind='barh', ax=axes[0])
axes[0].set_xlabel('Количество вакансий')
axes[0].set_title('Топ-20 городов по количеству вакансий')
axes[0].invert_yaxis()

city_stats_with_salary = city_stats[city_stats['Средняя ЗП'].notna()]
if len(city_stats_with_salary) > 0:
    city_stats_with_salary['Средняя ЗП'].sort_values(ascending=True).tail(15).plot(kind='barh', ax=axes[1])
    axes[1].set_xlabel('Средняя зарплата')
    axes[1].set_title('Топ-15 городов по средней зарплате')
    axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('plots/03_city_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n=== Анализ по компаниям ===")
company_stats = df.groupby('Компания').agg({
    'salary_target': ['count', 'mean'],
    'Название': 'count'
}).round(0)
company_stats.columns = ['Вакансий с ЗП', 'Средняя ЗП', 'Всего вакансий']
company_stats = company_stats.sort_values('Всего вакансий', ascending=False).head(20)
print(company_stats)

fig, ax = plt.subplots(figsize=(12, 8))
company_stats['Всего вакансий'].plot(kind='barh', ax=ax)
ax.set_xlabel('Количество вакансий')
ax.set_title('Топ-20 компаний по количеству вакансий')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('plots/04_company_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n=== Анализ по типам работы ===")
work_type_stats = df.groupby('Тип работы').agg({
    'salary_target': ['count', 'mean'],
    'Название': 'count'
}).round(0)
work_type_stats.columns = ['Вакансий с ЗП', 'Средняя ЗП', 'Всего вакансий']
work_type_stats = work_type_stats.sort_values('Всего вакансий', ascending=False)
print(work_type_stats)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
work_type_stats['Всего вакансий'].plot(kind='bar', ax=axes[0])
axes[0].set_ylabel('Количество вакансий')
axes[0].set_title('Распределение по типам работы')
axes[0].tick_params(axis='x', rotation=45)

work_type_with_salary = work_type_stats[work_type_stats['Средняя ЗП'].notna()]
if len(work_type_with_salary) > 0:
    work_type_with_salary['Средняя ЗП'].sort_values(ascending=True).plot(kind='barh', ax=axes[1])
    axes[1].set_xlabel('Средняя зарплата')
    axes[1].set_title('Средняя зарплата по типам работы')
    axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('plots/05_work_type_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n=== Индекс прозрачности ===")
df['transparency_score'] = 0
df.loc[df['salary_target'].notna(), 'transparency_score'] += 1
df.loc[df['Требования'].notna() & (df['Требования'] != ''), 'transparency_score'] += 1
df.loc[df['Навыки'].notna() & (df['Навыки'] != '') & (df['Навыки'] != 'Не указаны'), 'transparency_score'] += 1
df.loc[df['Количество откликов'].notna(), 'transparency_score'] += 1

transparency_stats = df.groupby('Источник')['transparency_score'].agg(['mean', 'std', 'count'])
print(transparency_stats)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
df.groupby('Источник')['transparency_score'].mean().plot(kind='bar', ax=axes[0])
axes[0].set_ylabel('Средний индекс прозрачности')
axes[0].set_title('Индекс прозрачности по источникам')
axes[0].tick_params(axis='x', rotation=45)

df['transparency_score'].hist(bins=5, ax=axes[1])
axes[1].set_xlabel('Индекс прозрачности')
axes[1].set_ylabel('Частота')
axes[1].set_title('Распределение индекса прозрачности')

plt.tight_layout()
plt.savefig('plots/06_transparency_index.png', dpi=300, bbox_inches='tight')
plt.close()

df.to_csv('merged_data.csv', index=False, encoding='utf-8')

