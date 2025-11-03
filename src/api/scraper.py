"""Main scraper class for HH.ru job vacancies."""
import asyncio
from typing import List, Dict, Any
from datetime import datetime
from loguru import logger

from .hh_client import HHAPIClient
from .data_processor import DataProcessor
from .csv_writer import CSVWriter
from .config import Config

class HHScraper:
    """Main scraper for HH.ru job vacancies."""
    
    def __init__(self, csv_output_path: str = "data/raw/hh_vacancies.csv"):
        self.data_processor = DataProcessor()
        self.csv_output_path = csv_output_path
        self.processed_count = 0
        self.error_count = 0
    
    async def search_and_collect_vacancies(
        self,
        search_queries: List[str],
        area_id: int = 1,
        max_pages_per_query: int = None
    ) -> None:
        """Search and collect vacancies for given queries."""
        if max_pages_per_query is None:
            max_pages_per_query = Config.MAX_PAGES_PER_SEARCH
        
        with CSVWriter(self.csv_output_path) as csv_writer:
            async with HHAPIClient() as client:
                for query_idx, query in enumerate(search_queries, 1):
                    logger.info(
                        f"\n{'#'*70}\n"
                        f"SEARCH {query_idx}/{len(search_queries)}: '{query}'\n"
                        f"{'#'*70}\n"
                    )
                    
                    try:
                        await self._process_search_query(
                            client, query, area_id, max_pages_per_query, csv_writer
                        )
                    except Exception as e:
                        logger.error(f"\nFATAL ERROR processing query '{query}': {e}\n")
                        logger.error("Continuing with next query...\n")
                        import traceback
                        logger.debug(traceback.format_exc())
    
    async def _process_search_query(
        self,
        client: HHAPIClient,
        query: str,
        area_id: int,
        max_pages: int,
        csv_writer: CSVWriter
    ) -> None:
        """Process a single search query."""
        page = 0
        total_found = 0
        vacancies_collected = 0
        query_errors_before = self.error_count
        
        while page < max_pages:
            try:
                if page > 0:
                    await asyncio.sleep(Config.PAGE_DELAY)
                search_result = await client.search_vacancies(
                    text=query,
                    area=area_id,
                    per_page=100,
                    page=page
                )
                
                if page == 0:
                    total_found = search_result.get('found', 0)
                    logger.info(f"Found {total_found} vacancies for query: '{query}'")
                
                vacancies = search_result.get('items', [])
                if not vacancies:
                    break
                
                vacancies_saved_before_page = vacancies_collected
                for idx, vacancy_item in enumerate(vacancies, 1):
                    try:
                        vacancy_id = str(vacancy_item.get('id', ''))
                        vacancy_name = vacancy_item.get('name', 'Unknown')
                        
                        logger.info(f"[Page {page+1}, Vacancy {idx}/{len(vacancies)}] Processing: {vacancy_name[:60]}... (ID: {vacancy_id})")
                        
                        if idx > 1:
                            await asyncio.sleep(Config.VACANCY_DETAIL_DELAY)
                        detailed_data = await client.get_vacancy_details(vacancy_id)
                        
                        if 'counters' in vacancy_item and vacancy_item.get('counters'):
                            if 'counters' not in detailed_data:
                                detailed_data['counters'] = {}
                            detailed_data['counters'].update(vacancy_item.get('counters', {}))
                        
                        if 'counters' not in detailed_data:
                            detailed_data['counters'] = {}
                        
                        processed_data = self.data_processor.process_vacancy_data(detailed_data)
                        if processed_data:
                            csv_writer.write_vacancy(processed_data)
                            
                            vacancies_collected += 1
                            self.processed_count += 1
                            
                            company = processed_data.get('company_name', 'N/A')
                            salary = processed_data.get('salary_from', 'N/A')
                            if processed_data.get('salary_to'):
                                salary = f"{salary}-{processed_data.get('salary_to')}"
                            city = processed_data.get('area_name', 'N/A')
                            responses = processed_data.get('responses_count', 'N/A')
                            
                            logger.success(
                                f"Saved: {vacancy_name[:50]} | "
                                f"Company: {company[:30]} | "
                                f"Salary: {salary} | "
                                f"City: {city} | "
                                f"Responses: {responses}"
                            )
                        else:
                            logger.warning(f"Failed to process vacancy {vacancy_id}: Empty data")
                            self.error_count += 1
                        
                    except Exception as e:
                        vacancy_id = vacancy_item.get('id', 'unknown')
                        logger.error(f"Error processing vacancy {vacancy_id}: {str(e)}")
                        self.error_count += 1
                        continue
                
                page += 1
                vacancies_saved_this_page = vacancies_collected - vacancies_saved_before_page
                errors_this_query = self.error_count - query_errors_before
                logger.info(
                    f"\n{'='*60}\n"
                    f"Page {page} completed for query '{query}'\n"
                    f"  - Vacancies processed this page: {len(vacancies)}\n"
                    f"  - Vacancies saved this page: {vacancies_saved_this_page}\n"
                    f"  - Total saved for this query: {vacancies_collected}\n"
                    f"  - Total found: {total_found}\n"
                    f"  - Errors this query: {errors_this_query}\n"
                    f"{'='*60}\n"
                )
                
            except Exception as e:
                logger.error(f"Error on page {page} for query '{query}': {e}")
                break
        
        success_rate = (vacancies_collected / max(page * 100, 1)) * 100 if page > 0 else 0
        logger.info(
            f"\n{'#'*60}\n"
            f"QUERY COMPLETED: '{query}'\n"
            f"  - Total vacancies found: {total_found}\n"
            f"  - Vacancies saved: {vacancies_collected}\n"
            f"  - Pages processed: {page}\n"
            f"  - Success rate: {success_rate:.1f}%\n"
            f"{'#'*60}\n"
        )
    
    
    async def get_collection_stats(self) -> Dict[str, int]:
        """Get collection statistics."""
        return {
            'total_vacancies': self.processed_count,
            'processed_vacancies': self.processed_count,
            'errors': self.error_count
        }
    

