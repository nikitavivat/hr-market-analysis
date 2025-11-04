#!/usr/bin/env python3
"""Скрипт для запуска LinkedIn парсера."""
import asyncio
import sys
from pathlib import Path

# Добавляем корень проекта в путь
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Импортируем и запускаем парсер
from src.scraper.linkedin_scraper import main

if __name__ == "__main__":
    print("=" * 70)
    print("ЗАПУСК LINKEDIN ПАРСЕРА")
    print("=" * 70)
    print(f"Рабочая директория: {project_root}")
    print(f"CSV файл: data/raw/linkedin_vacancies.csv")
    print("=" * 70)
    print()
    print("ВАЖНО: Парсер откроет браузер Chrome.")
    print("Вам нужно будет войти в LinkedIn вручную.")
    print("После входа введите что-либо в консоль и нажмите Enter.")
    print()
    print("=" * 70)
    print()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nПарсинг прерван пользователем.")
    except Exception as e:
        print(f"\nКритическая ошибка: {e}")
        import traceback
        traceback.print_exc()

