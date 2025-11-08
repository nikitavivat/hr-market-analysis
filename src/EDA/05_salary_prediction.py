import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from openai import OpenAI
import time
import warnings
warnings.filterwarnings('ignore')
from config import gpt_api_key
import pickle
import json


plt.rcParams['figure.figsize'] = (12, 8)
sns.set_style("whitegrid")

df = pd.read_csv('merged_data.csv')


df_train = df[df['Источник'] == 'linkedin'].copy()

df_train = df_train[df_train['salary_target'].notna()].copy()
print(f"LinkedIn с зарплатой: {df_train.shape[0]}")

if df_train.shape[0] < 10:
    df_train = df[df['salary_target'].notna()].copy()

client = OpenAI(api_key=gpt_api_key)

def get_embedding(text, model="text-embedding-3-small"):
    if not text or len(text.strip()) == 0:
        return None
    try:
        text = text.replace("\n", " ")[:8000]
        response = client.embeddings.create(input=[text], model=model)
        return response.data[0].embedding
    except Exception as e:
        return None

def get_embeddings_batch(texts, model="text-embedding-3-small", batch_size=100):
    embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_processed = []
        batch_mapping = []
        
        for idx, text in enumerate(batch):
            if text and len(str(text).strip()) > 0:
                processed = str(text).replace("\n", " ")[:8000]
                batch_processed.append(processed)
                batch_mapping.append(idx)
        
        if not batch_processed:
            embeddings.extend([None] * len(batch))
        else:
            try:
                response = client.embeddings.create(input=batch_processed, model=model)
                batch_embeddings = [item.embedding for item in response.data]
                
                result = [None] * len(batch)
                for emb_idx, orig_idx in enumerate(batch_mapping):
                    result[orig_idx] = batch_embeddings[emb_idx]
                
                embeddings.extend(result)
            except Exception as e:
                print(f"Ошибка в батче {i//batch_size + 1}/{total_batches}: {e}")
                embeddings.extend([None] * len(batch))
        
        if (i + batch_size) % 1000 == 0 or (i + batch_size) >= len(texts):
            print(f"Обработано: {min(i + batch_size, len(texts))}/{len(texts)} записей ({i//batch_size + 1}/{total_batches} батчей)")
        time.sleep(0.1)
    
    return embeddings

df_train['description_text'] = df_train['Описание вакансии'].fillna('') + ' ' + df_train['Требования'].fillna('')

df_train['description_length'] = df_train['Описание вакансии'].fillna('').str.len()
df_train['requirements_length'] = df_train['Требования'].fillna('').str.len()
df_train['title_length'] = df_train['Название'].fillna('').str.len()
df_train['description_words'] = df_train['Описание вакансии'].fillna('').str.split().str.len()
df_train['requirements_words'] = df_train['Требования'].fillna('').str.split().str.len()
df_train['has_requirements'] = df_train['Требования'].notna().astype(int)
df_train['has_skills'] = (df_train['Навыки'].fillna('') != 'Не указаны').astype(int)
df_train['work_type_remote'] = df_train['Тип работы'].fillna('').str.contains('Удаленная', na=False).astype(int)
df_train['work_type_full'] = df_train['Тип работы'].fillna('').str.contains('Полная', na=False).astype(int)
df_train['work_type_partial'] = df_train['Тип работы'].fillna('').str.contains('Частичная', na=False).astype(int)
df_train['work_type_shift'] = df_train['Тип работы'].fillna('').str.contains('Сменный', na=False).astype(int)
df_train['city_moscow'] = df_train['Город'].fillna('').str.contains('Москва', na=False).astype(int)
df_train['city_spb'] = df_train['Город'].fillna('').str.contains('Санкт-Петербург|Петербург', na=False).astype(int)

le_company = LabelEncoder()
df_train['company_encoded'] = le_company.fit_transform(df_train['Компания'].fillna('Unknown'))

le_city = LabelEncoder()
df_train['city_encoded'] = le_city.fit_transform(df_train['Город'].fillna('Unknown'))

le_work_type = LabelEncoder()
df_train['work_type_encoded'] = le_work_type.fit_transform(df_train['Тип работы'].fillna('Unknown'))

df_train['source_encoded'] = (df_train['Источник'] == 'linkedin').astype(int)

max_samples = 30000
sample_size = min(max_samples, len(df_train))
df_train_sample = df_train.sample(n=sample_size, random_state=42).copy()

description_texts = df_train_sample['description_text'].tolist()

batch_size = 100
print(f"Размер батча: {batch_size}")
print(f"Всего батчей: {(len(description_texts) + batch_size - 1) // batch_size}")

embeddings = get_embeddings_batch(description_texts, batch_size=batch_size)

valid_mask = [e is not None for e in embeddings]
valid_indices = [i for i, valid in enumerate(valid_mask) if valid]
df_train_sample = df_train_sample.iloc[valid_indices].copy()
embeddings = [e for e in embeddings if e is not None]

print(f"Successfully vectorized: {len(embeddings)}/{sample_size}")

if len(embeddings) > 0:
    embeddings_array = np.array(embeddings)
    print(f"Размерность эмбеддингов: {embeddings_array.shape}")
    
    pca_emb = PCA(n_components=2)
    embeddings_2d = pca_emb.fit_transform(embeddings_array)
    print(f"Объясненная дисперсия PCA: {pca_emb.explained_variance_ratio_.sum():.4f}")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    scatter1 = axes[0, 0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                  c=df_train_sample['salary_target'].values,
                                  cmap='viridis', alpha=0.6, s=20)
    axes[0, 0].set_xlabel('PC1')
    axes[0, 0].set_ylabel('PC2')
    axes[0, 0].set_title('PCA эмбеддингов (цвет = зарплата)')
    plt.colorbar(scatter1, ax=axes[0, 0], label='Зарплата')
    
    sample_for_tsne = min(5000, len(embeddings_array))
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings_array)-1))
    embeddings_tsne = tsne.fit_transform(embeddings_array[:sample_for_tsne])
    
    scatter2 = axes[0, 1].scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1],
                                  c=df_train_sample.iloc[:sample_for_tsne]['salary_target'].values,
                                  cmap='plasma', alpha=0.6, s=20)
    axes[0, 1].set_xlabel('t-SNE 1')
    axes[0, 1].set_ylabel('t-SNE 2')
    axes[0, 1].set_title('t-SNE эмбеддингов (цвет = зарплата)')
    plt.colorbar(scatter2, ax=axes[0, 1], label='Зарплата')
    
    axes[1, 0].hist(df_train_sample['salary_target'].values, bins=50, edgecolor='black')
    axes[1, 0].set_xlabel('Зарплата')
    axes[1, 0].set_ylabel('Частота')
    axes[1, 0].set_title('Распределение зарплат в выборке')
    
    pca_var = pca_emb.explained_variance_ratio_
    axes[1, 1].bar(range(1, min(11, len(pca_var)+1)), pca_var[:10])
    axes[1, 1].set_xlabel('Компонента')
    axes[1, 1].set_ylabel('Объясненная дисперсия')
    axes[1, 1].set_title('Первые 10 компонент PCA')
    
    plt.tight_layout()
    plt.savefig('plots/14_embeddings_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    feature_cols = ['description_length', 'requirements_length', 'title_length',
                   'description_words', 'requirements_words', 'has_requirements', 
                   'has_skills', 'work_type_remote', 'work_type_full', 
                   'work_type_partial', 'work_type_shift', 'city_moscow', 
                   'city_spb', 'company_encoded', 'city_encoded', 
                   'work_type_encoded', 'source_encoded']
    
    X_features = df_train_sample[feature_cols].fillna(0).values
    X_combined = np.hstack([embeddings_array, X_features])
    y = df_train_sample['salary_target'].values
    
    print(f"\nОбщее количество признаков: {X_combined.shape[1]}")
    print(f"  - Эмбеддинги: {embeddings_array.shape[1]}")
    print(f"  - Другие признаки: {X_features.shape[1]}")
    print(f"Размер обучающей выборки: {X_combined.shape[0]}")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_combined, y, test_size=0.2, random_state=42
    )
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    models = {
        'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=200, max_depth=10, random_state=42),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5)
    }
    
    results = {}
    predictions = {}
    
    for name, model in models.items():
        try:
            print(f"\nОбучение {name}...")
            if name in ['Ridge', 'Lasso', 'ElasticNet']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_val_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
            
            mae = mean_absolute_error(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            r2 = r2_score(y_val, y_pred)
            
            results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
            predictions[name] = y_pred
            print(f"{name} - MAE: {mae:.0f}, RMSE: {rmse:.0f}, R2: {r2:.4f}")
        except Exception as e:
            print(f"Ошибка при обучении {name}: {e}")
            continue
    
    for name, metrics in results.items():
        print(f"{name}: MAE={metrics['MAE']:.0f}, RMSE={metrics['RMSE']:.0f}, R2={metrics['R2']:.4f}")
    
    best_model_name = max(results, key=lambda x: results[x]['R2'])
    print(f"\nЛучшая модель: {best_model_name}")
    
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    results_df = pd.DataFrame(results).T
    
    ax1 = fig.add_subplot(gs[0, 0])
    results_df[['MAE', 'RMSE']].plot(kind='bar', ax=ax1)
    ax1.set_ylabel('Ошибка')
    ax1.set_title('Сравнение моделей (MAE, RMSE)')
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend()
    
    ax2 = fig.add_subplot(gs[0, 1])
    results_df['R2'].plot(kind='bar', ax=ax2, color='green')
    ax2.set_ylabel('R²')
    ax2.set_title('R² по моделям')
    ax2.tick_params(axis='x', rotation=45)
    
    ax3 = fig.add_subplot(gs[0, 2])
    results_df['MAE'].plot(kind='barh', ax=ax3, color='orange')
    ax3.set_xlabel('MAE')
    ax3.set_title('MAE по моделям')
    
    best_pred = predictions[best_model_name]
    
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.scatter(y_val, best_pred, alpha=0.5, s=10)
    ax4.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
    ax4.set_xlabel('Истинная зарплата')
    ax4.set_ylabel('Предсказанная зарплата')
    ax4.set_title(f'Предсказания vs Реальность ({best_model_name})')
    ax4.grid(True, alpha=0.3)
    
    residuals = y_val - best_pred
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.scatter(best_pred, residuals, alpha=0.5, s=10)
    ax5.axhline(y=0, color='r', linestyle='--')
    ax5.set_xlabel('Предсказанная зарплата')
    ax5.set_ylabel('Остатки')
    ax5.set_title('Остатки модели')
    ax5.grid(True, alpha=0.3)
    
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    ax6.set_xlabel('Остатки')
    ax6.set_ylabel('Частота')
    ax6.set_title('Распределение остатков')
    ax6.axvline(x=0, color='r', linestyle='--')
    
    for idx, (name, pred) in enumerate(predictions.items()):
        row = 2
        col = idx % 3
        ax = fig.add_subplot(gs[row, col])
        ax.scatter(y_val, pred, alpha=0.3, s=5)
        ax.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=1)
        ax.set_xlabel('Истинная')
        ax.set_ylabel('Предсказанная')
        ax.set_title(f'{name} (R²={results[name]["R2"]:.3f})')
        ax.grid(True, alpha=0.3)
    
    plt.savefig('plots/15_model_results_detailed.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n=== Важность признаков (RandomForest) ===")
    if 'RandomForest' in models:
        rf_model = models['RandomForest']
        feature_importance = pd.DataFrame({
            'feature': [f'embedding_{i}' for i in range(embeddings_array.shape[1])] + feature_cols,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)

        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        top_features = feature_importance.head(20)
        axes[0].barh(range(len(top_features)), top_features['importance'].values)
        axes[0].set_yticks(range(len(top_features)))
        axes[0].set_yticklabels(top_features['feature'].values)
        axes[0].set_xlabel('Важность')
        axes[0].set_title('Топ-20 важных признаков')
        axes[0].invert_yaxis()
        
        non_embedding_features = feature_importance[~feature_importance['feature'].str.startswith('embedding')]
        top_non_embedding = non_embedding_features.head(15)
        axes[1].barh(range(len(top_non_embedding)), top_non_embedding['importance'].values)
        axes[1].set_yticks(range(len(top_non_embedding)))
        axes[1].set_yticklabels(top_non_embedding['feature'].values)
        axes[1].set_xlabel('Важность')
        axes[1].set_title('Топ-15 важных признаков без эмбеддингов')
        axes[1].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('plots/16_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    

    
    results_summary = {
        'best_model': best_model_name,
        'models_metrics': {k: {m: float(v) for m, v in metrics.items()} for k, metrics in results.items()},
        'n_features': int(X_combined.shape[1]),
        'n_samples': int(X_combined.shape[0]),
        'n_embeddings': int(embeddings_array.shape[1]),
        'n_other_features': int(X_features.shape[1])
    }
    
    with open('model_results.json', 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    
    best_model_data = {
        'model': models[best_model_name],
        'scaler': scaler if best_model_name in ['Ridge', 'Lasso', 'ElasticNet'] else None,
        'model_type': best_model_name,
        'feature_cols': feature_cols,
        'le_company': le_company,
        'le_city': le_city,
        'le_work_type': le_work_type,
        'tfidf_model': None
    }
    
    with open('best_model.pkl', 'wb') as f:
        pickle.dump(best_model_data, f)
    
    all_models_data = {
        'models': models,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'le_company': le_company,
        'le_city': le_city,
        'le_work_type': le_work_type,
        'results': results
    }
    
    with open('all_models.pkl', 'wb') as f:
        pickle.dump(all_models_data, f)
    
    np.save('embeddings_final.npy', embeddings_array)
    np.save('embeddings_sample_indices.npy', df_train_sample.index.values)
    
    if 'RandomForest' in models:
        feature_importance.to_csv('feature_importance.csv', index=False, encoding='utf-8')
    
    
    import os
    print("\nПроверка созданных файлов:")
    files_to_check = [
        'model_results.json',
        'best_model.pkl',
        'all_models.pkl',
        'embeddings_final.npy',
        'embeddings_sample_indices.npy'
    ]
    if 'RandomForest' in models:
        files_to_check.append('feature_importance.csv')
    
    for file in files_to_check:
        if os.path.exists(file):
            size = os.path.getsize(file) / (1024*1024)
            print(f"  ✓ {file} ({size:.2f} MB)")
        else:
            print(f"  ✗ {file} - НЕ СОЗДАН!")
    
    print("\nПроверка созданных графиков:")
    plots_to_check = [
        'plots/14_embeddings_visualization.png',
        'plots/15_model_results_detailed.png',
        'plots/16_feature_importance.png'
    ]
    
    for plot in plots_to_check:
        if os.path.exists(plot):
            size = os.path.getsize(plot) / 1024
            print(f"  ✓ {plot} ({size:.1f} KB)")
        else:
            print(f"  ✗ {plot} - НЕ СОЗДАН!")
    
    print(f"\nЛучшая модель: {best_model_name} (R²={results[best_model_name]['R2']:.4f})")
    print(f"Размер обучающей выборки: {X_combined.shape[0]}")
    print(f"Количество признаков: {X_combined.shape[1]}")
else:
    print("Error: Failed to get embeddings")

