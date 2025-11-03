"""Main entry point for HH.ru scraper."""
import asyncio
import sys
from pathlib import Path
from loguru import logger

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from .config import Config
from .scraper import HHScraper
from search_queries_300 import SEARCH_QUERIES_300

# Configure logging
logger.remove()
logger.add(
    Config.LOG_FILE,
    level=Config.LOG_LEVEL,
    rotation="10 MB",
    retention="7 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
    encoding='utf-8'
)
logger.add(
    sys.stdout,
    level=Config.LOG_LEVEL,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:8}</level> | <level>{message}</level>",
    colorize=True
)

async def main():
    """Main function to run the scraper."""
    logger.info("Запуск HH.ru парсера...")
    
    Config.ensure_directories()
    logger.info("Проверка директорий...")
    
    logger.info("Инициализация парсера...")
    scraper = HHScraper()
    logger.info("Парсер инициализирован")
    
    # БИЗНЕС-ЗАДАЧА: Анализ рынка труда и сравнение зарплат по профессиям
    # Собираем минимум 200к вакансий для анализа
    # Используем расширенный список поисковых запросов (~395) для максимального покрытия рынка
    
    # Старый список профессий (закомментирован для избежания дублей)
    # search_queries_old = [
    #     # IT - разработка
    #     "python developer",
    #     "java developer",
    #     "javascript developer",
    #     "backend developer",
    #     "frontend developer",
    #     "full stack developer",
    #     
    #     # Data & ML
    #     "data scientist",
    #     "data analyst",
    #     "data engineer",
    #     "machine learning engineer",
    #     
    #     # DevOps & Infrastructure
    #     "devops engineer",
    #     "system administrator",
    #     "cloud engineer",
    #     
    #     # QA
    #     "qa engineer",
    #     "test engineer",
    #     
    #     # Management
    #     "product manager",
    #     "project manager",
    #     "team lead",
    #     
    #     # Business & Analytics
    #     "business analyst",
    #     "financial analyst",
    #     
    #     # Marketing & Sales
    #     "marketing manager",
    #     "sales manager",
    #     "digital marketing",
    #     
    #     # HR
    #     "hr manager",
    #     "recruiter",
    #     
    #     # Finance
    #     "accountant",
    #     "financial manager",
    #     
    #     # Design
    #     "designer",
    #     "ui designer",
    #     "ux designer",
    #     
    #     # Другие популярные профессии (русские варианты)
    #     "менеджер по продажам",
    #     "менеджер по работе с клиентами",
    #     "бухгалтер",
    #     "экономист",
    #     "юрист",
    #     "менеджер",
    #     "специалист",
    #     "инженер",
    #     "консультант"
    # ]
    
    # Новый список из 300 поисковых запросов
    search_queries = SEARCH_QUERIES_300
    
    try:
        # БИЗНЕС-ЗАДАЧА: Анализ рынка труда и сравнение зарплат по профессиям
        # Целевая переменная: Зарплата (salary_from, salary_to) - для статистического анализа
        # Признаки: Описание вакансии (текст), требования, навыки, профессия, город, компания
        # Цель: Провести EDA и выявить инсайты о рынке труда без обучения модели
        
            logger.info(
                f"\n{'='*70}\n"
            f"HH.RU VACANCY SCRAPER - MARKET ANALYSIS MODE\n"
                f"{'='*70}\n"
            f"Бизнес-задача: Анализ рынка труда и сравнение зарплат по профессиям\n"
            f"Целевая переменная: Зарплата (salary_from, salary_to) - для статистики\n"
            f"Признаки: Описание вакансии (текст), требования, навыки, профессия, город\n"
            f"Цель EDA: Выявить тренды, сравнить зарплаты, проанализировать требования\n"
                f"{'='*70}\n"
            f"Стратегия: Сбор выборки по профессиям (до 800 вакансий на профессию, максимум 40 страниц)\n"
            f"Всего профессий: {len(search_queries)}\n"
            f"Ожидаемое количество: до {len(search_queries) * 800} вакансий (минимум 200к)\n"
                f"Output CSV: data/raw/hh_vacancies.csv\n"
                f"Area: Russia (ID: 113)\n"
                f"{'='*70}\n"
            )
            
        # Собираем по 40 страниц на профессию (максимум 40 страниц, ~800 вакансий на запрос)
        # При 300 запросах это даст до 240к вакансий (минимум 200к)
        await scraper.search_and_collect_vacancies(
            search_queries=search_queries,
                area_id=113,  # Russia
            max_pages_per_query=40  # Максимум 40 страниц (требование проекта)
            )
        
        # Get final statistics
        stats = await scraper.get_collection_stats()
        logger.info(
            f"\n{'#'*70}\n"
            f"SCRAPING COMPLETED\n"
            f"{'#'*70}\n"
            f"Total vacancies collected: {stats['total_vacancies']}\n"
            f"Errors encountered: {stats['errors']}\n"
            f"CSV file saved to: data/raw/hh_vacancies.csv\n"
            f"{'#'*70}\n"
        )
        
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Scraping interrupted by user")
        logger.info("State has been saved. You can resume later.")
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
    finally:
        logger.info("Scraper shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())

