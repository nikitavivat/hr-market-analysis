"""CSV writer for saving job vacancies immediately after each request."""
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from loguru import logger

class CSVWriter:
    """Writes job vacancies to CSV file with immediate save after each request."""
    
    def __init__(self, output_path: str = "data/raw/hh_vacancies.csv"):
        """Initialize CSV writer."""
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.file_exists = self.output_path.exists()
        self.csv_file = None
        self.writer = None
        
        self.fieldnames = [
            'Название',
            'Компания',
            'Зарплата',
            'Количество откликов',
            'Город',
            'Навыки',
            'Тип работы',
            'Описание вакансии'
        ]
        
    def __enter__(self):
        """Open CSV file for writing."""
        self.csv_file = open(self.output_path, 'a', newline='', encoding='utf-8')
        self.writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)
        
        if not self.file_exists:
            self.writer.writeheader()
            self.file_exists = True
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close CSV file."""
        if self.csv_file:
            self.csv_file.close()
    
    def write_vacancy(self, vacancy_data: Dict) -> None:
        """Write single vacancy to CSV and flush immediately."""
        try:
            salary = self._format_salary(
                vacancy_data.get('salary_from'),
                vacancy_data.get('salary_to'),
                vacancy_data.get('salary_currency', 'RUR')
            )
            
            skills = self._format_skills(vacancy_data.get('skills', []))
            
            work_type = self._format_work_type(
                vacancy_data.get('employment_name'),
                vacancy_data.get('schedule_name')
            )
            
            description = vacancy_data.get('description', '')
            if not description:
                description = 'Нет описания'
            
            row = {
                'Название': vacancy_data.get('name', ''),
                'Компания': vacancy_data.get('company_name', ''),
                'Зарплата': salary,
                'Количество откликов': vacancy_data.get('responses_count', 'Н/Д'),
                'Город': vacancy_data.get('area_name', ''),
                'Навыки': skills,
                'Тип работы': work_type,
                'Описание вакансии': description
            }
            
            self.writer.writerow(row)
            self.csv_file.flush()
            os.fsync(self.csv_file.fileno())
            
        except Exception as e:
            logger.error(f"Error writing vacancy to CSV: {e}")
    
    def _format_salary(self, salary_from: Optional[int], salary_to: Optional[int], currency: str) -> str:
        """Format salary information."""
        if salary_from and salary_to:
            if currency == 'RUR':
                currency_symbol = '₽'
            elif currency == 'USD':
                currency_symbol = '$'
            elif currency == 'EUR':
                currency_symbol = '€'
            else:
                currency_symbol = currency
            return f"{salary_from:,} - {salary_to:,} {currency_symbol}"
        elif salary_from:
            if currency == 'RUR':
                currency_symbol = '₽'
            elif currency == 'USD':
                currency_symbol = '$'
            elif currency == 'EUR':
                currency_symbol = '€'
            else:
                currency_symbol = currency
            return f"от {salary_from:,} {currency_symbol}"
        elif salary_to:
            if currency == 'RUR':
                currency_symbol = '₽'
            elif currency == 'USD':
                currency_symbol = '$'
            elif currency == 'EUR':
                currency_symbol = '€'
            else:
                currency_symbol = currency
            return f"до {salary_to:,} {currency_symbol}"
        else:
            return "Не указана"
    
    def _format_skills(self, skills: List[str]) -> str:
        """Format skills list as comma-separated string."""
        if skills:
            return ', '.join(skills)
        return "Не указаны"
    
    def _format_work_type(self, employment: Optional[str], schedule: Optional[str]) -> str:
        """Format work type (employment + schedule)."""
        parts = []
        if employment:
            parts.append(employment)
        if schedule:
            parts.append(schedule)
        
        if parts:
            return ', '.join(parts)
        return "Не указан"
