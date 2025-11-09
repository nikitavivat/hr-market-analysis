import pandas as pd
import numpy as np
import pickle
from openai import OpenAI
from sklearn.preprocessing import LabelEncoder, StandardScaler
from config import gpt_api_key
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("R² (R-squared) - коэффициент детерминации")
print("="*60)
print("""
R² показывает, насколько хорошо модель объясняет вариацию данных:

• R² = 1.0  → Идеальная модель (100% точность)
• R² = 0.8  → Модель объясняет 80% вариации (хорошо)
• R² = 0.5  → Модель объясняет 50% вариации (средне)
• R² = 0.0  → Модель не лучше среднего значения
• R² < 0.0  → Модель хуже среднего значения

Формула: R² = 1 - (SS_res / SS_tot)
где:
  SS_res = сумма квадратов остатков (ошибок модели)
  SS_tot = общая сумма квадратов (отклонений от среднего)
""")
print("="*60)

print("\n=== Загрузка модели ===")
with open('best_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
scaler = model_data['scaler']
feature_cols = model_data['feature_cols']
le_company = model_data['le_company']
le_city = model_data['le_city']
le_work_type = model_data['le_work_type']

print(f"Модель: {model_data['model_type']}")
print(f"Признаков: {len(feature_cols)}")

print("\n=== Загрузка данных LinkedIn ===")
df = pd.read_csv('merged_data.csv')
df_linkedin = df[df['Источник'] == 'linkedin'].copy()

if len(df_linkedin) == 0:
    print("Нет данных LinkedIn!")
    exit()

first_vacancy = df_linkedin.iloc[0].copy()
print(f"\nПервая вакансия LinkedIn:")
print(f"Название: {first_vacancy['Название']}")
print(f"Компания: {first_vacancy['Компания']}")
print(f"Город: {first_vacancy['Город']}")
print(f"Тип работы: {first_vacancy['Тип работы']}")
if pd.notna(first_vacancy['salary_target']):
    print(f"Реальная зарплата: {first_vacancy['salary_target']:.0f} руб.")
else:
    print("Реальная зарплата: не указана")

print("\n=== Подготовка признаков ===")
vacancy = first_vacancy.copy()

vacancy['description_text'] = str(vacancy['Описание вакансии']) + ' ' + str(vacancy['Требования'])

vacancy['description_length'] = len(str(vacancy['Описание вакансии']))
vacancy['requirements_length'] = len(str(vacancy['Требования']))
vacancy['title_length'] = len(str(vacancy['Название']))
vacancy['description_words'] = len(str(vacancy['Описание вакансии']).split())
vacancy['requirements_words'] = len(str(vacancy['Требования']).split())
vacancy['has_requirements'] = 1 if pd.notna(vacancy['Требования']) and str(vacancy['Требования']).strip() else 0
vacancy['has_skills'] = 1 if (pd.notna(vacancy['Навыки']) and str(vacancy['Навыки']) != 'Не указаны') else 0
vacancy['work_type_remote'] = 1 if 'Удаленная' in str(vacancy['Тип работы']) else 0
vacancy['work_type_full'] = 1 if 'Полная' in str(vacancy['Тип работы']) else 0
vacancy['work_type_partial'] = 1 if 'Частичная' in str(vacancy['Тип работы']) else 0
vacancy['work_type_shift'] = 1 if 'Сменный' in str(vacancy['Тип работы']) else 0
vacancy['city_moscow'] = 1 if 'Москва' in str(vacancy['Город']) else 0
vacancy['city_spb'] = 1 if 'Санкт-Петербург' in str(vacancy['Город']) or 'Петербург' in str(vacancy['Город']) else 0

try:
    vacancy['company_encoded'] = le_company.transform([str(vacancy['Компания'])])[0]
except:
    vacancy['company_encoded'] = 0

try:
    vacancy['city_encoded'] = le_city.transform([str(vacancy['Город'])])[0]
except:
    vacancy['city_encoded'] = 0

try:
    vacancy['work_type_encoded'] = le_work_type.transform([str(vacancy['Тип работы'])])[0]
except:
    vacancy['work_type_encoded'] = 0

vacancy['source_encoded'] = 1

print("Признаки подготовлены")

print("\n=== Векторизация описания через GPT ===")
client = OpenAI(api_key=gpt_api_key)

def get_embedding(text, model="text-embedding-3-small"):
    if not text or len(str(text).strip()) == 0:
        return None
    try:
        text = str(text).replace("\n", " ")[:8000]
        response = client.embeddings.create(input=[text], model=model)
        return response.data[0].embedding
    except Exception as e:
        print(f"Ошибка: {e}")
        return None

embedding = get_embedding(vacancy['description_text'])
if embedding is None:
    print("Ошибка при получении эмбеддинга!")
    exit()

embeddings_array = np.array([embedding])
print(f"Эмбеддинг получен: {embeddings_array.shape}")

print("\n=== Формирование финальных признаков ===")
X_features = np.array([[vacancy[col] for col in feature_cols]])
X_features = np.nan_to_num(X_features, nan=0.0)

X_combined = np.hstack([embeddings_array, X_features])
print(f"Размерность признаков: {X_combined.shape}")

print("\n=== Предсказание ===")
if model_data['model_type'] in ['Ridge', 'Lasso', 'ElasticNet']:
    X_scaled = scaler.transform(X_combined)
    predicted_salary = model.predict(X_scaled)[0]
else:
    predicted_salary = model.predict(X_combined)[0]

print("\n" + "="*60)
print("РЕЗУЛЬТАТЫ ПРЕДСКАЗАНИЯ")
print("="*60)
print(f"\nВакансия: {first_vacancy['Название']}")
print(f"Компания: {first_vacancy['Компания']}")
print(f"Город: {first_vacancy['Город']}")
print(f"Тип работы: {first_vacancy['Тип работы']}")

if pd.notna(first_vacancy['salary_target']):
    real_salary = first_vacancy['salary_target']
    print(f"\nРеальная зарплата: {real_salary:.0f} руб.")
    print(f"Предсказанная зарплата: {predicted_salary:.0f} руб.")
    
    error = abs(predicted_salary - real_salary)
    error_percent = (error / real_salary) * 100
    
    print(f"\nОшибка: {error:.0f} руб. ({error_percent:.1f}%)")
    
    if error_percent < 10:
        print("✓ Отличное предсказание!")
    elif error_percent < 20:
        print("✓ Хорошее предсказание")
    elif error_percent < 30:
        print("~ Приемлемое предсказание")
    else:
        print("✗ Большая ошибка")
else:
    print(f"\nПредсказанная зарплата: {predicted_salary:.0f} руб.")
    print("(Реальная зарплата не указана в данных)")

print("\n" + "="*60)
print(f"Использованная модель: {model_data['model_type']}")
print("="*60)

