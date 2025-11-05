import pandas as pd

print("=== Загрузка данных ===")
df = pd.read_csv('merged_data.csv')

print(f"Исходный размер: {df.shape}")
print(f"Дубликаты: {df.duplicated().sum()}")

print("\n=== Удаление полных дубликатов ===")
df_clean = df.drop_duplicates()

print(f"После удаления полных дубликатов: {df_clean.shape}")
print(f"Удалено: {len(df) - len(df_clean)} записей")

print("\n=== Удаление дубликатов по ключевым полям ===")
key_columns = ['Название', 'Компания', 'Город', 'Описание вакансии']
df_clean = df_clean.drop_duplicates(subset=key_columns, keep='first')

print(f"После удаления дубликатов по ключевым полям: {df_clean.shape}")
print(f"Всего удалено: {len(df) - len(df_clean)} записей")

print("\n=== Сохранение ===")
df_clean.to_csv('merged_data.csv', index=False, encoding='utf-8')
print("✓ merged_data.csv обновлен")

print("\n" + "="*60)
print(f"Итог: {len(df)} → {len(df_clean)} записей")
print(f"Удалено: {len(df) - len(df_clean)} дубликатов")
print("="*60)

