import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

df_linkedin = pd.read_csv('linkedin_vacancies.csv', on_bad_lines='skip', encoding='utf-8')
df_hh = pd.read_csv('hh_vacancies.csv', on_bad_lines='skip', encoding='utf-8')
df_hh_old = pd.read_csv('hh_vacancies_old_20251106_014345.csv', on_bad_lines='skip', encoding='utf-8')

print(f"LinkedIn: {df_linkedin.shape}")
print(f"HH: {df_hh.shape}")
print(f"HH Old: {df_hh_old.shape}")

df_combined = pd.concat([df_linkedin, df_hh, df_hh_old], ignore_index=True)
print(f"Объединенный датасет: {df_combined.shape}")
print(f"Дубликаты: {df_combined.duplicated().sum()}")

df_combined = df_combined.drop_duplicates()
print(f"После удаления дубликатов: {df_combined.shape}")

df_combined['Зарплата_от'] = pd.to_numeric(df_combined['Зарплата_от'], errors='coerce')
df_combined['Зарплата_до'] = pd.to_numeric(df_combined['Зарплата_до'], errors='coerce')
df_combined['Количество откликов'] = pd.to_numeric(df_combined['Количество откликов'], errors='coerce')

def create_salary_target(row):
    if pd.notna(row['Зарплата_от']) and pd.notna(row['Зарплата_до']):
        return (row['Зарплата_от'] + row['Зарплата_до']) / 2
    elif pd.notna(row['Зарплата_от']):
        return row['Зарплата_от']
    elif pd.notna(row['Зарплата_до']):
        return row['Зарплата_до']
    else:
        return np.nan

df_combined['salary_target'] = df_combined.apply(create_salary_target, axis=1)

df_combined['Описание вакансии'] = df_combined['Описание вакансии'].fillna('')
df_combined['Требования'] = df_combined['Требования'].fillna('')
df_combined['Название'] = df_combined['Название'].fillna('')
df_combined['Компания'] = df_combined['Компания'].fillna('')
df_combined['Город'] = df_combined['Город'].fillna('')
df_combined['Навыки'] = df_combined['Навыки'].fillna('')
df_combined['Тип работы'] = df_combined['Тип работы'].fillna('')

missing_analysis = df_combined.isnull().sum()
missing_analysis = missing_analysis[missing_analysis > 0]
print(missing_analysis)

if df_combined['salary_target'].notna().sum() > 0:
    Q1 = df_combined['salary_target'].quantile(0.25)
    Q3 = df_combined['salary_target'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df_combined[(df_combined['salary_target'] < lower_bound) | (df_combined['salary_target'] > upper_bound)]
    print(f"Выбросов (IQR метод): {outliers.shape[0]}")
    print(f"Нижняя граница: {lower_bound:.0f}, Верхняя граница: {upper_bound:.0f}")
    
    df_combined = df_combined[(df_combined['salary_target'].isna()) | 
                               ((df_combined['salary_target'] >= lower_bound) & 
                                (df_combined['salary_target'] <= upper_bound))]

print(f"\nФинальный размер: {df_combined.shape}")

df_combined.to_csv('merged_data.csv', index=False, encoding='utf-8')

