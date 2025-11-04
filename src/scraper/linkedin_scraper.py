"""LinkedIn парсер вакансий с использованием Playwright."""
import asyncio
import csv
import re
import json
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from playwright.async_api import async_playwright, Browser, Page, BrowserContext
from loguru import logger

# Определяем корень проекта
_project_root = Path(__file__).parent.parent.parent
_data_dir = _project_root / 'data'
_logs_dir = _project_root / 'logs'

# Добавляем путь для импорта CSVWriter
sys.path.insert(0, str(_project_root))
from src.api.csv_writer import CSVWriter

# Создаем директории если их нет
_data_dir.mkdir(exist_ok=True, parents=True)
_logs_dir.mkdir(exist_ok=True, parents=True)
(_data_dir / 'raw').mkdir(exist_ok=True, parents=True)
(_data_dir / 'processed').mkdir(exist_ok=True, parents=True)


class LinkedInScraper:
    """Парсер вакансий LinkedIn с использованием Playwright."""
    
    def __init__(self, csv_file: str = None):
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.processed_count = 0
        self.error_count = 0
        self.is_logged_in = False
        
        # Путь к CSV файлу (используем тот же формат что и hh.ru)
        if csv_file:
            self.csv_file = Path(csv_file)
        else:
            self.csv_file = _data_dir / 'raw' / 'linkedin_vacancies.csv'
        
        # Путь для сохранения cookies/storage state
        self.cookies_file = _data_dir / 'tokens' / 'linkedin_cookies.json'
        self.cookies_file.parent.mkdir(exist_ok=True, parents=True)
        
        # Путь для постоянного профиля браузера (persistent context)
        self.browser_profile_dir = _data_dir / 'tokens' / 'linkedin_browser_profile'
        self.browser_profile_dir.mkdir(exist_ok=True, parents=True)
        
        # Список собранных вакансий
        self.vacancies: List[Dict[str, Any]] = []
        
        # CSVWriter будет инициализирован в методе поиска
        self.csv_writer = None
        
        # StateManager для отслеживания прогресса
        from src.api.state_manager import StateManager
        self.state_manager = StateManager(state_file=str(_data_dir / 'linkedin_scraper_state.json'))
        
        # Быстрый режим ОТКЛЮЧЕН - собираем полные данные со страниц
        self.fast_mode = False
        
    
    
    def _clean_text(self, text: str) -> str:
        """Очистка текста от HTML и лишних символов."""
        if not text:
            return ""
        
        # Удаляем HTML теги
        text = re.sub(r'<[^>]+>', '', text)
        
        # Удаляем лишние пробелы
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _extract_skills(self, description: str) -> List[str]:
        """Извлечение навыков из описания."""
        if not description:
            return []
        
        skills = []
        text = description.lower()
        
        # Список популярных технологий
        technologies = [
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php',
            'ruby', 'go', 'rust', 'swift', 'kotlin', 'scala', 'r', 'matlab',
            'sql', 'html', 'css', 'react', 'angular', 'vue', 'node.js',
            'django', 'flask', 'fastapi', 'spring', 'laravel', 'rails',
            'express', 'next.js', 'docker', 'kubernetes', 'aws', 'azure',
            'gcp', 'git', 'jenkins', 'ci/cd', 'microservices', 'rest api', 'graphql'
        ]
        
        for tech in technologies:
            if tech in text:
                skills.append(tech.title())
        
        return list(set(skills))
    
    def _parse_salary(self, salary_text: str) -> tuple:
        """Парсинг зарплаты из текста LinkedIn."""
        if not salary_text:
            return None, None, 'RUR'
        
        salary_text = salary_text.lower().strip()
        
        # Пытаемся извлечь числа
        numbers = re.findall(r'[\d,]+', salary_text.replace(',', ''))
        if not numbers:
            return None, None, 'RUR'
        
        # Определяем валюту
        currency = 'RUR'
        if '$' in salary_text or 'usd' in salary_text:
            currency = 'USD'
        elif '€' in salary_text or 'eur' in salary_text:
            currency = 'EUR'
        elif '₽' in salary_text or 'rub' in salary_text or 'rur' in salary_text:
            currency = 'RUR'
        
        # Парсим диапазон
        try:
            if len(numbers) >= 2:
                salary_from = int(numbers[0].replace(',', ''))
                salary_to = int(numbers[1].replace(',', ''))
                return salary_from, salary_to, currency
            elif len(numbers) == 1:
                salary_value = int(numbers[0].replace(',', ''))
                if 'от' in salary_text or 'from' in salary_text or 'min' in salary_text:
                    return salary_value, None, currency
                elif 'до' in salary_text or 'to' in salary_text or 'max' in salary_text:
                    return None, salary_value, currency
                else:
                    return salary_value, salary_value, currency
        except ValueError:
            pass
        
        return None, None, currency
    
    async def initialize(self):
        """Инициализация браузера и открытие страницы входа."""
        import time
        start_time = time.time()
        
        logger.info("Запуск Playwright...")
        self.playwright = await async_playwright().start()
        logger.info(f"Playwright запущен за {time.time() - start_time:.1f}с")
        
        logger.info("Запуск браузера Chrome (видимый режим, без изображений)...")
        launch_start = time.time()
        
        # Используем видимый режим (LinkedIn блокирует headless), но блокируем изображения
        self.browser = await self.playwright.chromium.launch(
            headless=False,
            channel="chrome",
            args=[
                '--start-maximized',
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage',
            ],
            timeout=30000
        )
        logger.info(f"Браузер запущен за {time.time() - launch_start:.1f}с")
        
        # Создаем контекст с минимальными настройками для совместимости
        logger.info("Создание контекста браузера...")
        context_start = time.time()
        
        context_options = {
            'viewport': {'width': 1920, 'height': 1080},
            'user_agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'locale': 'ru-RU',
            'timezone_id': 'Europe/Moscow',
        }
        
        # Загружаем сохраненные cookies, если они есть
        if self.cookies_file.exists():
            try:
                context_options['storage_state'] = str(self.cookies_file)
                logger.info(f"Загружены сохраненные cookies из {self.cookies_file}")
            except Exception as e:
                logger.warning(f"Не удалось загрузить cookies: {e}")
        
        self.context = await self.browser.new_context(**context_options)
        logger.info(f"Контекст создан за {time.time() - context_start:.1f}с")
        
        async def route_handler(route):
            try:
                resource_type = route.request.resource_type
                if resource_type in ['image', 'stylesheet', 'font', 'media']:
                    try:
                        await route.abort()
                    except:
                        pass
                else:
                    try:
                        await route.continue_()
                    except:
                        pass
            except:
                pass
        
        self.page = await self.context.new_page()
        await self.context.route('**/*', route_handler)
        logger.info("Страница создана")
        
        has_cookies = self.cookies_file.exists()
        
        if has_cookies:
            logger.info("Обнаружены сохраненные cookies, проверяем авторизацию...")
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    await self.page.goto('https://www.linkedin.com/feed', 
                                       wait_until='commit',
                           timeout=30000)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Попытка {attempt + 1}/{max_retries} не удалась: {e}")
                    else:
                        logger.warning(f"Не удалось загрузить feed, потребуется вход вручную")
                        has_cookies = False
                        break
            
            await self._check_login_status()
            
            if self.is_logged_in:
                logger.info("Авторизация через сохраненные cookies успешна!")
                logger.info("Начинаем парсинг без дополнительного входа...")
                return
        
        logger.info("=" * 60)
        logger.info("ТРЕБУЕТСЯ ВХОД В LINKEDIN")
        logger.info("=" * 60)
        logger.info("Открываем страницу входа LinkedIn...")
        await self.page.goto('https://www.linkedin.com/login', wait_until='commit', timeout=30000)
        logger.info(f"Полная инициализация заняла {time.time() - start_time:.1f}с")
        
        logger.info("=" * 60)
        logger.info("Войдите в LinkedIn в браузере, затем нажмите Enter в консоли")
        logger.info("=" * 60)
        await self._wait_for_user_input()
        await self._check_login_status()
        
        if not self.is_logged_in:
            logger.warning("Похоже, вы не вошли в систему. Попробуйте еще раз.")
            return
        
        await self._save_cookies()
        
        logger.info("Вход выполнен успешно! Начинаем парсинг...")
    
    async def _wait_for_user_input(self):
        logger.info("Ожидание ввода пользователя...")
        print("\n" + "=" * 60)
        print("Введите что-либо и нажмите Enter для начала парсинга:")
        print("=" * 60)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: input())
        logger.info("Начинаем парсинг вакансий...")
    
    async def _check_login_status(self):
        try:
            current_url = self.page.url
            
            if 'login' not in current_url.lower() and 'feed' in current_url.lower():
                self.is_logged_in = True
            elif 'login' not in current_url.lower():
                await self.page.goto('https://www.linkedin.com/feed', wait_until='commit', timeout=30000)
                current_url = self.page.url
                if 'login' not in current_url.lower():
                    self.is_logged_in = True
        except Exception as e:
            logger.error(f"Ошибка при проверке статуса входа: {e}")
            self.is_logged_in = False
    
    async def _save_cookies(self):
        try:
            if self.context:
                await self.context.storage_state(path=str(self.cookies_file))
                logger.info(f"Cookies сохранены в {self.cookies_file}")
                logger.info("При следующем запуске вход не потребуется!")
            except Exception as e:
                logger.warning(f"Не удалось сохранить cookies: {e}")
    
    async def search_and_collect_vacancies(
        self,
        search_queries: List[str],
        location: str = "",
        max_pages_per_query: int = 5
    ) -> None:
        if not self.is_logged_in:
            logger.error("Необходимо войти в LinkedIn перед началом парсинга.")
            return
        
        self.csv_writer = CSVWriter(str(self.csv_file))
        self.csv_writer.__enter__()
        
        try:
            for query_idx, query in enumerate(search_queries):
                logger.info(f"Запрос {query_idx + 1}/{len(search_queries)}: '{query}'")
                
                try:
                    await self._process_search_query(
                        query, location, max_pages_per_query
                    )
                except Exception as e:
                    error_str = str(e)
                    if 'ERR_HTTP_RESPONSE_CODE_FAILURE' in error_str or 'ERR_ABORTED' in error_str:
                        logger.warning(f"LinkedIn блокирует запросы. Ждем 5 секунд...")
                        await asyncio.sleep(5)
                    elif 'AttributeError' in error_str or 'object has no attribute' in error_str or 'collected to prevent' in error_str:
                        logger.warning(f"Критическая ошибка Playwright, пересоздаем контекст...")
                        await self._recreate_context()
                    pass
                
        except Exception as e:
            logger.error(f"Критическая ошибка при парсинге: {e}")
        finally:
            logger.info(f"Всего собрано вакансий: {len(self.vacancies)}")
            if self.csv_writer:
                try:
                    self.csv_writer.__exit__(None, None, None)
                except:
                    pass
    
    async def _process_search_query(
        self,
        query: str,
        location: str,
        max_pages: int
    ) -> None:
        vacancies_collected = 0
        
        try:
            search_url = self._build_search_url(query, location)
            logger.info(f"Переходим на: {search_url}")
            
            try:
                try:
                    if self.page.is_closed():
                        logger.warning("Страница закрыта, пересоздаем...")
                        self.page = await self.context.new_page()
                except:
                    self.page = await self.context.new_page()
                
                await self.page.goto(search_url, wait_until='commit', timeout=30000)
                try:
                    await self.page.wait_for_selector('div[data-job-id], li.jobs-search-results__list-item, div.jobs-search-results__list-item', timeout=10000)
                except:
                    pass
            except Exception as e:
                error_str = str(e)
                if 'ERR_HTTP_RESPONSE_CODE_FAILURE' in error_str or 'ERR_ABORTED' in error_str:
                    logger.warning(f"LinkedIn блокирует запрос. Ждем 3 секунды и повторяем...")
                    await asyncio.sleep(3)
                    try:
                        if not self.page.is_closed():
                            await self.page.close()
                    except:
                        pass
                    self.page = await self.context.new_page()
                    await self.page.goto(search_url, wait_until='commit', timeout=30000)
                    try:
                        await self.page.wait_for_selector('div[data-job-id], li.jobs-search-results__list-item', timeout=10000)
                    except:
                        pass
                elif 'collected to prevent unbounded heap growth' in error_str or 'AttributeError' in error_str:
                    try:
                        if not self.page.is_closed():
                            await self.page.close()
                    except:
                        pass
                    self.page = await self.context.new_page()
                    await self.page.goto(search_url, wait_until='commit', timeout=30000)
                    try:
                        await self.page.wait_for_selector('div[data-job-id], li.jobs-search-results__list-item', timeout=10000)
                    except:
                        pass
                else:
                    logger.warning(f"Страница загружается медленно, продолжаем... {e}")
                    try:
                        if not self.page.is_closed():
                            await self.page.close()
                    except:
                        pass
                    self.page = await self.context.new_page()
                    await self.page.goto(search_url, wait_until='commit', timeout=30000)
                    try:
                        await self.page.wait_for_selector('div[data-job-id], li.jobs-search-results__list-item', timeout=10000)
                    except:
                        pass
            
            await self._scroll_page()
            current_url = self.page.url
            logger.info(f"Текущий URL: {current_url}")
            
            previous_page_ids = set()
            
            for page_num in range(max_pages):
                logger.info(f"Обрабатываем страницу {page_num + 1}...")
                
                try:
                    if self.page.is_closed():
                        logger.warning("Страница закрыта, пересоздаем...")
                        self.page = await self.context.new_page()
                        await self.page.goto(search_url, wait_until='commit', timeout=20000)
                    else:
                        try:
                            _ = self.page.url
                        except:
                            try:
                                if not self.page.is_closed():
                                    await self.page.close()
                            except:
                                pass
                            self.page = await self.context.new_page()
                            await self.page.goto(search_url, wait_until='commit', timeout=20000)
                except Exception as e:
                    error_str = str(e)
                    if 'collected to prevent unbounded heap growth' in error_str or 'AttributeError' in error_str:
                        try:
                            self.page = await self.context.new_page()
                            await self.page.goto(search_url, wait_until='commit', timeout=20000)
                        except:
                            pass
                
                await self._scroll_page()
                page_vacancies = await self._parse_vacancies_on_page()
                
                if not page_vacancies:
                    logger.info("Вакансии на этой странице не найдены. Завершаем.")
                    break
                
                current_page_ids = {v.get('linkedin_id') for v in page_vacancies if v.get('linkedin_id')}
                if current_page_ids and current_page_ids == previous_page_ids:
                    logger.info("Обнаружены повторяющиеся результаты. Завершаем.")
                    break
                previous_page_ids = current_page_ids
                
                for vacancy_data in page_vacancies:
                    try:
                        try:
                            if self.page.is_closed():
                                self.page = await self.context.new_page()
                                await self.page.goto(search_url, wait_until='commit', timeout=20000)
                        except:
                            try:
                                self.page = await self.context.new_page()
                                await self.page.goto(search_url, wait_until='commit', timeout=20000)
                            except:
                                continue
                        
                        try:
                            detailed_data = await self._get_vacancy_details(vacancy_data)
                            if detailed_data:
                                processed_data = self._process_linkedin_vacancy(detailed_data)
                            else:
                                processed_data = self._process_linkedin_vacancy(vacancy_data)
                        except Exception as detail_error:
                            error_str = str(detail_error)
                            if 'AttributeError' in error_str or 'object has no attribute' in error_str or 'collected to prevent' in error_str:
                                await self._recreate_context()
                                try:
                                    await self.page.goto(search_url, wait_until='commit', timeout=20000)
                                except:
                                    pass
                            processed_data = self._process_linkedin_vacancy(vacancy_data)
                        
                        if processed_data:
                            self.vacancies.append(processed_data)
                            vacancies_collected += 1
                            self.processed_count += 1
                            await self._save_to_csv(processed_data)
                    except Exception as e:
                        error_str = str(e)
                        if 'AttributeError' in error_str or 'object has no attribute' in error_str or 'collected to prevent' in error_str:
                            await self._recreate_context()
                            try:
                                await self.page.goto(search_url, wait_until='commit', timeout=20000)
                            except:
                                pass
                        self.error_count += 1
                        continue
                
                if page_num < max_pages - 1:
                    next_button = None
                    next_selectors = [
                        'button[aria-label="Далее"]',
                        'button[aria-label="Next"]',
                        'button[aria-label*="next" i]',
                        'button[aria-label*="следующ" i]',
                        'button[data-test-pagination-page-btn]',
                        'button[aria-label="Next page"]'
                    ]
                    
                    for selector in next_selectors:
                        try:
                            next_button = await self.page.query_selector(selector)
                            if next_button:
                                is_visible = await next_button.is_visible()
                                if is_visible:
                                    break
                                next_button = None
                        except:
                            continue
                    
                    if next_button:
                        try:
                            is_disabled = await next_button.get_attribute('disabled')
                            if is_disabled:
                                logger.info("Достигнута последняя страница.")
                                break
                            
                            await next_button.click()
                        except Exception as e:
                            logger.warning(f"Не удалось нажать кнопку 'Далее': {e}")
                            break
                    else:
                        logger.info("Кнопка 'Далее' не найдена. Завершаем.")
                        break
                
                logger.info(f"Обработано вакансий на странице {page_num + 1}: {len(page_vacancies)}")
            
            logger.info(f"Завершен поиск для '{query}': собрано {vacancies_collected} вакансий")
            
        except Exception as e:
            logger.error(f"Ошибка при обработке запроса '{query}': {e}")
            raise
    
    def _build_search_url(self, query: str, location: str = "") -> str:
        base_url = "https://www.linkedin.com/jobs/search/"
        params = []
        
        if query:
            params.append(f"keywords={query.replace(' ', '%20')}")
        if location:
            params.append(f"location={location.replace(' ', '%20')}")
        
        if params:
            return f"{base_url}?{'&'.join(params)}"
        return base_url
    
    async def _recreate_context(self):
        try:
            try:
                if self.context:
                    try:
                        _ = self.context.pages
                        await self._save_cookies()
                    except:
                        pass
            except:
                pass
            
            try:
                if self.page and not self.page.is_closed():
                    await self.page.close()
            except:
                pass
            
            try:
                if self.context:
                    await self.context.close()
            except:
                pass
            
            context_options = {
                'viewport': {'width': 1920, 'height': 1080},
                'user_agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'locale': 'ru-RU',
                'timezone_id': 'Europe/Moscow',
            }
            
            if self.cookies_file.exists():
                try:
                    context_options['storage_state'] = str(self.cookies_file)
                except:
                    pass
            
            self.context = await self.browser.new_context(**context_options)
            
            async def route_handler(route):
                try:
                    resource_type = route.request.resource_type
                    if resource_type in ['image', 'stylesheet', 'font', 'media']:
                        try:
                            await route.abort()
                        except:
                            pass
                    else:
                        try:
                            await route.continue_()
                        except:
                            pass
                except:
                    pass
            
            await self.context.route('**/*', route_handler)
            
            self.page = await self.context.new_page()
            logger.warning("Контекст и страница пересозданы из-за критической ошибки")
        except Exception as e:
            logger.error(f"Не удалось пересоздать контекст: {e}")
    
    async def _scroll_page(self):
        try:
            try:
                if self.page.is_closed():
                    return
            except:
                await self._recreate_context()
                return
            await self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        except Exception as e:
            error_str = str(e)
            if 'AttributeError' in error_str or 'object has no attribute' in error_str or 'collected to prevent' in error_str:
                await self._recreate_context()
    
    async def _parse_vacancies_on_page(self) -> List[Dict[str, Any]]:
        vacancies = []
        
        try:
            try:
                if self.page.is_closed():
                    return vacancies
                _ = self.page.url
            except:
                return vacancies
            
            try:
                page_content = await self.page.content()
            except Exception as e:
                error_str = str(e)
                if 'AttributeError' in error_str or 'object has no attribute' in error_str or 'collected to prevent' in error_str:
                    await self._recreate_context()
                    return vacancies
                page_content = ""
            
            vacancy_selectors = [
                'div.jobs-search-results__list-item',
                'li.jobs-search-results__list-item',
                'div[data-job-id]',
                'article.job-result-card',
                'div.base-card',
                'li.base-card',
                'div[data-test-job-card]',
                'section.jobs-search-results-list',
                'ul.jobs-search-results__list > li',
                'div.scaffold-layout__list-container > ul > li'
            ]
            
            vacancy_elements = None
            used_selector = None
            
            for selector in vacancy_selectors:
                try:
                    elements = await self.page.query_selector_all(selector)
                    if elements and len(elements) > 0:
                        vacancy_elements = elements
                        used_selector = selector
                        logger.info(f"Найдено вакансий по селектору '{selector}': {len(elements)}")
                        break
                except Exception as e:
                    error_str = str(e)
                    if 'AttributeError' in error_str or 'object has no attribute' in error_str or 'collected to prevent' in error_str:
                        await self._recreate_context()
                        return vacancies
                    logger.debug(f"Селектор '{selector}' не сработал: {e}")
                    continue
            
            if not vacancy_elements:
                logger.warning("Вакансии не найдены стандартными селекторами.")
                logger.info("Пробуем найти любые ссылки на вакансии...")
                
                job_links = await self.page.query_selector_all('a[href*="/jobs/view/"]')
                if job_links:
                    logger.info(f"Найдено {len(job_links)} ссылок на вакансии через href")
                    for link in job_links:
                        try:
                            href = await link.get_attribute('href')
                            if href and '/jobs/view/' in href:
                                parent = await link.evaluate_handle('el => el.closest("li, div, article")')
                                if parent:
                                    vacancy_elements = [parent] * len(job_links)
                                    used_selector = 'a[href*="/jobs/view/"]'
                                    break
                        except:
                            continue
                
                if not vacancy_elements:
                    logger.error("Не удалось найти вакансии на странице.")
                    logger.info("Проверьте вручную в браузере, возможно LinkedIn показывает капчу или требует дополнительной авторизации.")
                    return vacancies
            
            for idx, element in enumerate(vacancy_elements):
                try:
                    vacancy_data = {}
                    
                    try:
                        try:
                            if self.page.is_closed():
                                break
                        except:
                                break
                    
                        data = await element.evaluate('''el => {
                            const data = {};
                            
                            const link = el.querySelector('a[href*="/jobs/view/"]');
                            if (link) {
                                const href = link.getAttribute('href') || link.href;
                                if (href) {
                                    data.url = href.startsWith('http') ? href : 'https://www.linkedin.com' + href;
                                    const match = href.match(/\/jobs\/view\/(\d+)/);
                                    if (match) data.linkedin_id = match[1];
                                }
                            }
                            
                            const titleEl = el.querySelector('a[href*="/jobs/view/"] strong, a[href*="/jobs/view/"] span[aria-hidden="true"] strong, h3, h2, .job-card-list__title--link');
                            if (titleEl) {
                                data.title = titleEl.textContent.trim();
                            }
                            
                            const companyEl = el.querySelector('.artdeco-entity-lockup__subtitle, .job-card-list__company-name, h4, .qYOcoUeBMzhbXUuWXSnpXzXDDEZuEEHp');
                            if (companyEl) {
                                data.company = companyEl.textContent.trim();
                            }
                            
                            const locationEl = el.querySelector('.job-card-container__metadata-wrapper, .artdeco-entity-lockup__caption, .WSlvxnMQVsHkXzuKYRaTTeWAsAtGrFwd');
                            if (locationEl) {
                                const locationText = locationEl.textContent.trim();
                                const locationMatch = locationText.match(/^([^(]+)/);
                                if (locationMatch) {
                                    data.location = locationMatch[1].trim();
                                }
                                const workTypeMatch = locationText.match(/\(([^)]+)\)/);
                                if (workTypeMatch) {
                                    data.job_type = workTypeMatch[1].trim();
                                }
                            }
                            
                            const dateEl = el.querySelector('time, .job-card-container__footer-item time');
                            if (dateEl) {
                                const dateText = dateEl.textContent.trim();
                                data.published_date = dateText;
                                const datetime = dateEl.getAttribute('datetime');
                                if (datetime) {
                                    data.published_datetime = datetime;
                                }
                            }
                            
                            const responsesEl = el.querySelector('[aria-label*="человек нажали"], [aria-label*="applicants"]');
                            if (responsesEl) {
                                const responsesText = responsesEl.getAttribute('aria-label') || responsesEl.textContent;
                                const match = responsesText.match(/(\d+)/);
                                if (match) {
                                    data.responses_count = parseInt(match[1]);
                                }
                            }
                            
                            if (!data.linkedin_id) {
                                const jobId = el.getAttribute('data-job-id') || el.closest('[data-job-id]')?.getAttribute('data-job-id');
                                if (jobId) {
                                    data.linkedin_id = jobId;
                                }
                            }
                            
                            return data;
                        }''')
                        
                        vacancy_data.update(data)
                    except Exception as e:
                        error_str = str(e)
                        if 'AttributeError' in error_str or 'object has no attribute' in error_str or 'collected to prevent' in error_str:
                            continue
                        logger.debug(f"Ошибка при JS парсинге элемента {idx+1}: {e}")
                    
                    if not vacancy_data.get('linkedin_id'):
                        try:
                            job_id = await element.get_attribute('data-job-id')
                            if not job_id:
                                job_id = await element.evaluate('el => el.closest("[data-job-id]")?.getAttribute("data-job-id")')
                            if job_id:
                                vacancy_data['linkedin_id'] = job_id
                        except:
                            pass
                    
                    if not vacancy_data.get('url') and vacancy_data.get('linkedin_id'):
                        vacancy_data['url'] = f"https://www.linkedin.com/jobs/view/{vacancy_data['linkedin_id']}/"
                    
                    if vacancy_data.get('linkedin_id') or vacancy_data.get('title'):
                        vacancies.append(vacancy_data)
                        logger.debug(f"Парсинг вакансии {idx+1}/{len(vacancy_elements)}: {vacancy_data.get('title', 'Без названия')[:50]}")
                
                except Exception as e:
                    logger.warning(f"Ошибка при парсинге карточки вакансии {idx+1}: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Ошибка при парсинге страницы: {e}")
        
        return vacancies
    
    async def _get_vacancy_details(self, vacancy_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            url = vacancy_data.get('url')
            if not url:
                return vacancy_data
            
            original_url = self.page.url
            
            try:
                full_url = url if url.startswith('http') else f"https://www.linkedin.com{url}"
                clean_url = full_url.split('?')[0]
                logger.debug(f"Переходим на страницу вакансии: {clean_url}")
                
                try:
                    if self.page.is_closed():
                        self.page = await self.context.new_page()
                except:
                    # Если страница потеряна - пересоздаем
                    try:
                        self.page = await self.context.new_page()
                    except:
                        return vacancy_data
                
                try:
                    await self.page.goto(clean_url, wait_until='commit', timeout=15000)
                    try:
                        await self.page.wait_for_selector('.jobs-description-content__text, .jobs-description__content, #job-details', timeout=5000)
                    except:
                        pass
                except Exception as e:
                    error_str = str(e)
                    if 'collected to prevent unbounded heap growth' in error_str or 'AttributeError' in error_str or 'object has no attribute' in error_str:
                        await self._recreate_context()
                        try:
                            await self.page.goto(clean_url, wait_until='commit', timeout=15000)
                        except:
                            return vacancy_data
                    else:
                        return vacancy_data
                
                details = {}
                details.update(vacancy_data)
                
                try:
                    if self.page.is_closed():
                        # Пересоздаем страницу
                        self.page = await self.context.new_page()
                        await self.page.goto(clean_url, wait_until='commit', timeout=15000)
                except Exception as e:
                    # Если не можем пересоздать - возвращаем базовые данные
                    return vacancy_data
                
                # Получаем все данные через JavaScript
                # Обертываем в дополнительный try-except для обработки ошибок Playwright
                page_data = {}
                try:
                    # Проверяем, что страница еще валидна
                    try:
                    if self.page.is_closed():
                        raise Exception("Page closed")
                    _ = self.page.url
                except Exception as check_error:
                    try:
                            if not self.page.is_closed():
                                await self.page.close()
                        except:
                            pass
                        self.page = await self.context.new_page()
                        await self.page.goto(clean_url, wait_until='commit', timeout=15000)
                    
                    try:
                        page_data = await self.page.evaluate('''() => {
                        const data = {};
                        
                        const descSelectors = [
                            '.jobs-description-content__text',
                            '.jobs-description__content',
                            '#job-details',
                            '.jobs-box__html-content',
                            '.jobs-description__container'
                        ];
                        for (const sel of descSelectors) {
                            const el = document.querySelector(sel);
                            if (el) {
                                const text = el.textContent.trim();
                                if (text.length > 50) {
                                    data.description = text;
                                    break;
                                }
                            }
                        }
                        
                        const reqKeywords = ['требования', 'requirements', 'мы ждём', 'ожидаем', 'нужно'];
                        const allText = data.description || document.body.textContent;
                        const reqSection = Array.from(document.querySelectorAll('h2, h3, strong')).find(el => {
                            const text = el.textContent.toLowerCase();
                            return reqKeywords.some(kw => text.includes(kw));
                        });
                        if (reqSection) {
                            let reqText = '';
                            let next = reqSection.nextElementSibling;
                            while (next && !next.matches('h2, h3')) {
                                reqText += next.textContent + ' ';
                                next = next.nextElementSibling;
                            }
                            if (reqText.trim()) {
                                data.requirements = reqText.trim();
                            }
                        }
                        
                        const salaryEl = document.querySelector('[id="SALARY"], .job-details-jobs-unified-top-card__job-insight, [data-test-id*="salary"]');
                        if (salaryEl) {
                            data.salary = salaryEl.textContent.trim();
                        }
                        
                        if (!data.job_type) {
                            const workTypeEl = document.querySelector('.job-details-fit-level-preferences button, .job-details-jobs-unified-top-card__primary-description-container');
                            if (workTypeEl) {
                                const workTypes = Array.from(document.querySelectorAll('.job-details-fit-level-preferences button'))
                                    .map(btn => btn.textContent.trim())
                                    .filter(t => t);
                                if (workTypes.length > 0) {
                                    data.job_type = workTypes.join(', ');
                                }
                            }
                        }
                        
                        const skillsEl = document.querySelector('[data-test-id*="skill"], .job-details-how-you-match-card, .job-criteria__text');
                        if (skillsEl) {
                            const skills = Array.from(document.querySelectorAll('.job-criteria__text, [data-test-id*="skill"]'))
                                .map(el => el.textContent.trim())
                                .filter(s => s && s.length > 2);
                            if (skills.length > 0) {
                                data.skills_list = skills;
                            }
                        }
                        
                        // Количество откликов (если не было в списке)
                        if (!data.responses_count) {
                            const responsesText = document.body.textContent;
                            const match = responsesText.match(/(\d+)\s*(?:человек|чел|applicants?)\s*(?:нажали|нажало|clicked)/i);
                            if (match) {
                                data.responses_count = parseInt(match[1]);
                            }
                        }
                        
                        // Дата публикации (если не было в списке)
                        if (!data.published_date) {
                            const dateEl = document.querySelector('time[datetime], .job-details-jobs-unified-top-card__primary-description-container time');
                            if (dateEl) {
                                data.published_date = dateEl.textContent.trim();
                                const datetime = dateEl.getAttribute('datetime');
                                if (datetime) {
                                    data.published_datetime = datetime;
                                }
                            }
                        }
                        
                        return data;
                    }''')
                    except Exception as eval_error:
                        error_str = str(eval_error)
                        # Если критическая ошибка Playwright - пересоздаем контекст
                        if 'AttributeError' in error_str or 'object has no attribute' in error_str or 'collected to prevent' in error_str:
                            await self._recreate_context()
                        # Продолжаем с пустыми данными
                        page_data = {}
                    
                    # Обновляем details данными со страницы
                    if page_data:
                        for key, value in page_data.items():
                            if value:  # Обновляем только непустые значения
                                details[key] = value
                            
                except Exception as e:
                    error_str = str(e)
                    # Если критическая ошибка Playwright - пересоздаем контекст
                    if 'collected to prevent unbounded heap growth' in error_str or 'AttributeError' in error_str or 'object has no attribute' in error_str:
                        await self._recreate_context()
                    # Тихий пропуск - продолжаем с базовыми данными
                
                # Запасной вариант через селекторы для описания
                if not details.get('description'):
                    description_selectors = [
                        '.jobs-description-content__text',
                        '.jobs-description__content',
                        '#job-details',
                        '.jobs-box__html-content'
                    ]
                    
                    for selector in description_selectors:
                        try:
                            desc_element = await self.page.query_selector(selector)
                            if desc_element:
                                desc_text = await desc_element.inner_text()
                                if desc_text and len(desc_text.strip()) > 50:
                                    details['description'] = desc_text.strip()
                                    break
                        except:
                            continue
                
                return details
                
            finally:
                # Возвращаемся на страницу поиска
                try:
                    if original_url and '/jobs/search' in original_url:
                        await self.page.goto(original_url, wait_until='commit', timeout=20000)
                except:
                    pass
        
        except Exception as e:
            error_str = str(e)
            # Если объект потерян - пересоздаем страницу и возвращаем базовые данные
            if 'collected to prevent unbounded heap growth' in error_str or 'AttributeError' in error_str:
                try:
                    if not self.page.is_closed():
                        await self.page.close()
                except:
                    pass
                try:
                    self.page = await self.context.new_page()
                except:
                    pass
            # Возвращаем базовые данные, которые уже есть
            return vacancy_data
    
    def _process_linkedin_vacancy(self, raw_data: Dict) -> Optional[Dict]:
        """Обработка данных вакансии LinkedIn в формате hh.ru."""
        try:
            linkedin_id = raw_data.get('linkedin_id', '')
            if not linkedin_id:
                return None
            
            # Обрабатываем описание
            description = raw_data.get('description', '')
            if description:
                description = self._clean_text(description)
            if not description:
                description = 'Нет описания'
            
            # Обрабатываем требования
            requirements = raw_data.get('requirements', '')
            if requirements:
                requirements = self._clean_text(requirements)
            
            # Извлекаем навыки
            skills = []
            # Сначала пробуем получить навыки из списка (если были найдены на странице)
            if raw_data.get('skills_list'):
                skills = raw_data.get('skills_list', [])
            else:
                # Иначе извлекаем из описания
                skills = self._extract_skills(description + ' ' + requirements)
            
            # Парсим зарплату
            salary_text = raw_data.get('salary', '')
            salary_from, salary_to, salary_currency = self._parse_salary(salary_text)
            
            # Парсим тип работы
            job_type = raw_data.get('job_type', '')
            employment_name = ''
            schedule_name = ''
            
            if job_type:
                job_type_lower = job_type.lower()
                # Определяем тип занятости
                if 'full-time' in job_type_lower or 'полная' in job_type_lower or 'полный рабочий день' in job_type_lower:
                    employment_name = 'Полная занятость'
                elif 'part-time' in job_type_lower or 'частичная' in job_type_lower:
                    employment_name = 'Частичная занятость'
                elif 'contract' in job_type_lower or 'контракт' in job_type_lower or 'проект' in job_type_lower:
                    employment_name = 'Проектная работа'
                elif 'стажировка' in job_type_lower or 'internship' in job_type_lower:
                    employment_name = 'Стажировка'
                
                # Определяем график работы
                if 'remote' in job_type_lower or 'удаленн' in job_type_lower or 'удаленная работа' in job_type_lower:
                    schedule_name = 'Удаленная работа'
                elif 'hybrid' in job_type_lower or 'гибридн' in job_type_lower or 'гибридный формат' in job_type_lower:
                    schedule_name = 'Гибридный формат работы'
                elif 'flexible' in job_type_lower or 'гибк' in job_type_lower:
                    schedule_name = 'Гибкий график'
                elif 'full day' in job_type_lower or 'полный день' in job_type_lower or 'работа в офисе' in job_type_lower:
                    schedule_name = 'Полный день'
            
            # Парсим локацию (убираем тип работы из локации если он там есть)
            location = raw_data.get('location', '')
            if location and '(' in location:
                # Убираем часть в скобках (тип работы)
                location = location.split('(')[0].strip()
            
            # Получаем количество откликов
            responses_count = raw_data.get('responses_count')
            if responses_count is None:
                # Пробуем извлечь из текста
                responses_text = raw_data.get('responses_text', '')
                if responses_text:
                    import re
                    match = re.search(r'(\d+)\s*(?:человек|чел|applicants?)', responses_text, re.IGNORECASE)
                    if match:
                        responses_count = int(match.group(1))
            
            # Формируем данные в формате hh.ru
            return {
                'name': raw_data.get('title', ''),
                'company_name': raw_data.get('company', ''),
                'salary_from': salary_from,
                'salary_to': salary_to,
                'salary_currency': salary_currency,
                'area_name': location,  # Используем очищенную локацию
                'skills': skills,
                'description': description,
                'requirements': requirements,
                'responses_count': responses_count,
                'employment_name': employment_name,
                'schedule_name': schedule_name,
                'vacancy_url': raw_data.get('url', ''),
                'source': 'linkedin'  # Для идентификации источника
            }
        
        except Exception as e:
            logger.error(f"Ошибка при обработке данных вакансии: {e}")
            return None
    
    async def _save_to_csv(self, vacancy_data: Dict):
        """Сохранение вакансии в CSV файл через CSVWriter в формате hh.ru."""
        if not self.csv_writer:
            logger.error("CSVWriter не инициализирован")
            return
        
        if not vacancy_data:
            logger.debug("Нет данных для сохранения.")
            return
        
        try:
            # Модифицируем данные для CSVWriter (добавляем источник)
            vacancy_for_csv = vacancy_data.copy()
            # CSVWriter ожидает 'Источник' в поле, но мы передаем через отдельный параметр
            # Нужно модифицировать CSVWriter или передавать данные как есть
            
            # Сохраняем через CSVWriter
            await self.csv_writer.write_vacancy(vacancy_for_csv)
            
        except Exception as e:
            logger.error(f"Ошибка при сохранении в CSV: {e}")
    
    def get_collection_stats(self) -> Dict[str, int]:
        """Получение статистики сбора данных."""
        return {
            'total_vacancies': len(self.vacancies),
            'processed': self.processed_count,
            'errors': self.error_count
        }
    
    async def close(self):
        """Закрытие браузера и очистка ресурсов."""
        try:
            # Сохраняем cookies перед закрытием (на случай если они обновились)
            if self.is_logged_in and self.context:
                await self._save_cookies()
            
            if self.page:
                await self.page.close()
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            if hasattr(self, 'playwright'):
                await self.playwright.stop()
            logger.info("Браузер закрыт.")
        except Exception as e:
            logger.error(f"Ошибка при закрытии браузера: {e}")


async def main():
    """Главная функция для запуска парсера."""
    # Импортируем поисковые запросы (пробуем разные варианты)
    try:
        from search_queries import SEARCH_QUERIES
        search_queries = SEARCH_QUERIES
    except ImportError:
        try:
            from search_queries_300 import SEARCH_QUERIES_300
            search_queries = SEARCH_QUERIES_300
        except ImportError:
            try:
                from search_queries_friend import SEARCH_QUERIES
                search_queries = SEARCH_QUERIES
            except ImportError:
                logger.warning("Не удалось импортировать поисковые запросы, используем базовый список")
                search_queries = [
                    "python developer",
                    "data scientist",
                    "machine learning engineer",
                    "backend developer",
                    "frontend developer"
                ]
    
    scraper = LinkedInScraper()
    
    try:
        # Инициализация браузера и ожидание входа
        await scraper.initialize()
        
        if not scraper.is_logged_in:
            logger.error("Не удалось войти в систему. Завершение работы.")
            return
        
        logger.info(f"Начинаем парсинг для {len(search_queries)} поисковых запросов")
        
        # Начинаем парсинг (ограничиваем количество страниц для теста)
        await scraper.search_and_collect_vacancies(
            search_queries=search_queries,
            location="",
            max_pages_per_query=20  # Увеличено для сбора 10к полных вакансий
        )
        
        # Выводим статистику
        stats = scraper.get_collection_stats()
        logger.info("=" * 60)
        logger.info("СТАТИСТИКА:")
        logger.info(f"Всего вакансий: {stats['total_vacancies']}")
        logger.info(f"Обработано: {stats['processed']}")
        logger.info(f"Ошибок: {stats['errors']}")
        logger.info(f"Файл сохранен: {scraper.csv_file}")
        logger.info("=" * 60)
        
    except KeyboardInterrupt:
        logger.info("Парсинг прерван пользователем.")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await scraper.close()


if __name__ == "__main__":
    asyncio.run(main())
