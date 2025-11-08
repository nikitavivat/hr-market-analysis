import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (16, 12)
sns.set_style("whitegrid")

print("=== Загрузка моделей и данных ===")

with open('all_models.pkl', 'rb') as f:
    models_data = pickle.load(f)

with open('model_results.json', 'r', encoding='utf-8') as f:
    results_data = json.load(f)

embeddings = np.load('embeddings_final.npy')
indices = np.load('embeddings_sample_indices.npy')

df = pd.read_csv('merged_data.csv')
df_sample = df.iloc[indices].copy()

print(f"Загружено моделей: {len(models_data['models'])}")
print(f"Размер выборки: {len(df_sample)}")

print("\n=== Подготовка признаков ===")
df_sample['description_length'] = df_sample['Описание вакансии'].fillna('').str.len()
df_sample['requirements_length'] = df_sample['Требования'].fillna('').str.len()
df_sample['title_length'] = df_sample['Название'].fillna('').str.len()
df_sample['description_words'] = df_sample['Описание вакансии'].fillna('').str.split().str.len()
df_sample['requirements_words'] = df_sample['Требования'].fillna('').str.split().str.len()
df_sample['has_requirements'] = df_sample['Требования'].notna().astype(int)
df_sample['has_skills'] = (df_sample['Навыки'].fillna('') != 'Не указаны').astype(int)
df_sample['work_type_remote'] = df_sample['Тип работы'].fillna('').str.contains('Удаленная', na=False).astype(int)
df_sample['work_type_full'] = df_sample['Тип работы'].fillna('').str.contains('Полная', na=False).astype(int)
df_sample['work_type_partial'] = df_sample['Тип работы'].fillna('').str.contains('Частичная', na=False).astype(int)
df_sample['work_type_shift'] = df_sample['Тип работы'].fillna('').str.contains('Сменный', na=False).astype(int)
df_sample['city_moscow'] = df_sample['Город'].fillna('').str.contains('Москва', na=False).astype(int)
df_sample['city_spb'] = df_sample['Город'].fillna('').str.contains('Санкт-Петербург|Петербург', na=False).astype(int)

from sklearn.preprocessing import LabelEncoder
le_company = LabelEncoder()
le_company.classes_ = models_data['le_company'].classes_
df_sample['company_encoded'] = le_company.transform(df_sample['Компания'].fillna('Unknown'))

le_city = LabelEncoder()
le_city.classes_ = models_data['le_city'].classes_
df_sample['city_encoded'] = le_city.transform(df_sample['Город'].fillna('Unknown'))

le_work_type = LabelEncoder()
le_work_type.classes_ = models_data['le_work_type'].classes_
df_sample['work_type_encoded'] = le_work_type.transform(df_sample['Тип работы'].fillna('Unknown'))

df_sample['source_encoded'] = (df_sample['Источник'] == 'linkedin').astype(int)

feature_cols = models_data['feature_cols']
X_features = df_sample[feature_cols].fillna(0).values
X_combined = np.hstack([embeddings, X_features])
y = df_sample['salary_target'].values

print(f"Размер данных: {X_combined.shape}")
print(f"Целевая переменная: {y.shape}")

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    X_combined, y, test_size=0.2, random_state=42
)

scaler = models_data['scaler']
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)

print("\n=== Предсказания моделей ===")
models = models_data['models']
predictions = {}

for name, model in models.items():
    try:
        if name in ['Ridge', 'Lasso', 'ElasticNet']:
            y_pred = model.predict(X_val_scaled)
        else:
            y_pred = model.predict(X_val)
        predictions[name] = y_pred
        print(f"{name}: предсказания готовы")
    except Exception as e:
        print(f"{name}: ошибка - {e}")

print("\n=== Создание графиков регрессий ===")

fig = plt.figure(figsize=(24, 20))
gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.4)

model_names = list(predictions.keys())
best_model = results_data['best_model']

for idx, name in enumerate(model_names):
    row = idx // 3
    col = idx % 3
    
    if row >= 4:
        break
    
    y_pred = predictions[name]
    
    ax1 = fig.add_subplot(gs[row*2, col])
    ax1.scatter(y_val, y_pred, alpha=0.4, s=15, edgecolors='none')
    ax1.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2, label='Идеальная линия')
    ax1.set_xlabel('Истинная зарплата', fontsize=10)
    ax1.set_ylabel('Предсказанная зарплата', fontsize=10)
    ax1.set_title(f'{name}\nR²={results_data["models_metrics"][name]["R2"]:.4f}', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    from sklearn.metrics import r2_score
    r2 = r2_score(y_val, y_pred)
    ax1.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax1.transAxes, 
             fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax2 = fig.add_subplot(gs[row*2+1, col])
    residuals = y_val - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.4, s=15, edgecolors='none')
    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('Предсказанная зарплата', fontsize=10)
    ax2.set_ylabel('Остатки (Истинная - Предсказанная)', fontsize=10)
    ax2.set_title(f'Остатки {name}', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(y_val, y_pred)
    ax2.text(0.05, 0.95, f'MAE = {mae:.0f}', transform=ax2.transAxes,
             fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.suptitle('Графики регрессий всех моделей', fontsize=16, fontweight='bold', y=0.995)
plt.savefig('plots/17_all_regression_plots.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ plots/17_all_regression_plots.png")

print("\n=== Детальные графики для лучшей модели ===")
best_pred = predictions[best_model]

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

axes[0, 0].scatter(y_val, best_pred, alpha=0.5, s=20, edgecolors='none')
axes[0, 0].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Истинная зарплата')
axes[0, 0].set_ylabel('Предсказанная зарплата')
axes[0, 0].set_title(f'{best_model} - Предсказания vs Реальность')
axes[0, 0].grid(True, alpha=0.3)

residuals = y_val - best_pred
axes[0, 1].scatter(best_pred, residuals, alpha=0.5, s=20, edgecolors='none')
axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0, 1].set_xlabel('Предсказанная зарплата')
axes[0, 1].set_ylabel('Остатки')
axes[0, 1].set_title('Остатки модели')
axes[0, 1].grid(True, alpha=0.3)

axes[0, 2].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
axes[0, 2].axvline(x=0, color='r', linestyle='--', lw=2)
axes[0, 2].set_xlabel('Остатки')
axes[0, 2].set_ylabel('Частота')
axes[0, 2].set_title('Распределение остатков')
axes[0, 2].grid(True, alpha=0.3)

error_percent = np.abs(residuals / y_val) * 100
axes[1, 0].hist(error_percent, bins=50, edgecolor='black', alpha=0.7)
axes[1, 0].set_xlabel('Процент ошибки (%)')
axes[1, 0].set_ylabel('Частота')
axes[1, 0].set_title('Распределение процентной ошибки')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].scatter(y_val, error_percent, alpha=0.3, s=10, edgecolors='none')
axes[1, 1].set_xlabel('Истинная зарплата')
axes[1, 1].set_ylabel('Процент ошибки (%)')
axes[1, 1].set_title('Ошибка в зависимости от зарплаты')
axes[1, 1].grid(True, alpha=0.3)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(y_val, best_pred)
rmse = np.sqrt(mean_squared_error(y_val, best_pred))
r2 = r2_score(y_val, best_pred)

metrics_text = f"""
Метрики {best_model}:

MAE: {mae:.0f} руб.
RMSE: {rmse:.0f} руб.
R²: {r2:.4f}

Средняя ошибка: {np.mean(error_percent):.2f}%
Медианная ошибка: {np.median(error_percent):.2f}%
"""

axes[1, 2].text(0.1, 0.5, metrics_text, transform=axes[1, 2].transAxes,
                fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
axes[1, 2].axis('off')

plt.suptitle(f'Детальный анализ модели {best_model}', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/18_best_model_detailed.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ plots/18_best_model_detailed.png")

print("\n=== Сравнение всех моделей ===")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

results_df = pd.DataFrame(results_data['models_metrics']).T

results_df[['MAE', 'RMSE']].plot(kind='bar', ax=axes[0, 0], color=['orange', 'red'])
axes[0, 0].set_ylabel('Ошибка (руб.)')
axes[0, 0].set_title('MAE и RMSE по моделям')
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

results_df['R2'].plot(kind='bar', ax=axes[0, 1], color='green')
axes[0, 1].set_ylabel('R²')
axes[0, 1].set_title('R² по моделям')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)

mae_values = [results_data['models_metrics'][m]['MAE'] for m in model_names]
axes[1, 0].barh(model_names, mae_values, color='steelblue')
axes[1, 0].set_xlabel('MAE (руб.)')
axes[1, 0].set_title('MAE по моделям (горизонтально)')
axes[1, 0].grid(True, alpha=0.3, axis='x')

r2_values = [results_data['models_metrics'][m]['R2'] for m in model_names]
colors = ['gold' if m == best_model else 'lightblue' for m in model_names]
axes[1, 1].barh(model_names, r2_values, color=colors)
axes[1, 1].set_xlabel('R²')
axes[1, 1].set_title('R² по моделям (лучшая выделена)')
axes[1, 1].grid(True, alpha=0.3, axis='x')

plt.suptitle('Сравнение всех моделей', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/19_models_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ plots/19_models_comparison.png")

print("\n=== Графики распределения ошибок ===")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

all_residuals = []
all_names = []

for name, pred in predictions.items():
    residuals = y_val - pred
    all_residuals.append(residuals)
    all_names.append(name)

axes[0, 0].boxplot(all_residuals, labels=all_names)
axes[0, 0].set_ylabel('Остатки')
axes[0, 0].set_title('Распределение остатков по моделям (Boxplot)')
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)

for idx, (name, residuals) in enumerate(zip(all_names, all_residuals)):
    axes[0, 1].hist(residuals, bins=50, alpha=0.5, label=name, edgecolor='black')
axes[0, 1].set_xlabel('Остатки')
axes[0, 1].set_ylabel('Частота')
axes[0, 1].set_title('Распределение остатков (все модели)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axvline(x=0, color='r', linestyle='--', alpha=0.5)

mae_list = [results_data['models_metrics'][m]['MAE'] for m in all_names]
rmse_list = [results_data['models_metrics'][m]['RMSE'] for m in all_names]

axes[1, 0].scatter(mae_list, rmse_list, s=200, alpha=0.6, edgecolors='black')
for i, name in enumerate(all_names):
    axes[1, 0].annotate(name, (mae_list[i], rmse_list[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
axes[1, 0].set_xlabel('MAE (руб.)')
axes[1, 0].set_ylabel('RMSE (руб.)')
axes[1, 0].set_title('MAE vs RMSE')
axes[1, 0].grid(True, alpha=0.3)

r2_list = [results_data['models_metrics'][m]['R2'] for m in all_names]
axes[1, 1].bar(range(len(all_names)), r2_list, color=colors, edgecolor='black')
axes[1, 1].set_xticks(range(len(all_names)))
axes[1, 1].set_xticklabels(all_names, rotation=45, ha='right')
axes[1, 1].set_ylabel('R²')
axes[1, 1].set_title('R² всех моделей')
axes[1, 1].grid(True, alpha=0.3, axis='y')
axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)

plt.suptitle('Анализ ошибок моделей', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/20_error_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ plots/20_error_analysis.png")

print("\n" + "="*60)
print("Все графики регрессий созданы!")
print("="*60)
print("\nСозданные графики:")
print("  ✓ plots/17_all_regression_plots.png - все модели (scatter + residuals)")
print("  ✓ plots/18_best_model_detailed.png - детальный анализ лучшей модели")
print("  ✓ plots/19_models_comparison.png - сравнение всех моделей")
print("  ✓ plots/20_error_analysis.png - анализ ошибок")
print("="*60)

