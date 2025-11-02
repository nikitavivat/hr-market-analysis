"""HH.ru API client with async support."""
import asyncio
import aiohttp
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
from .config import Config
from .auth import ensure_authorization, get_access_token

class HHAPIClient:
    """Async client for HH.ru API."""
    
    def __init__(self):
        self.base_url = Config.API_BASE_URL
        self.session = None
        self.access_token = None
        self.semaphore = asyncio.Semaphore(Config.MAX_CONCURRENT_REQUESTS)
    
    async def authorize(self):
        """Get and set access token."""
        logger.info("Получение токена доступа...")
        loop = asyncio.get_event_loop()
        self.access_token = await loop.run_in_executor(None, ensure_authorization)
        logger.info("Токен получен")
    
    async def __aenter__(self):
        """Async context manager entry."""
        logger.info("Открытие соединения с HH.ru API...")
        if not self.access_token:
            await self.authorize()
        logger.info("Соединение установлено")
        
        headers = {
            'User-Agent': 'HH.ru Job Scraper/1.0',
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.access_token}'
        }
        
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers=headers
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make HTTP request with retry logic."""
        async with self.semaphore:
            # Remove leading slash if present
            endpoint = endpoint.lstrip('/')
            url = f"{self.base_url}/{endpoint}"
            
            try:
                async with self.session.get(url, params=params) as response:
                    if response.status == 429:  # Rate limit
                        await asyncio.sleep(2)
                        raise Exception("Rate limited")
                    
                    if response.status == 403:
                        logger.warning(f"403 Forbidden: {url} - Token may be invalid or expired")
                        # Try to refresh token
                        await self.authorize()
                        headers = dict(self.session.headers)
                        headers['Authorization'] = f'Bearer {self.access_token}'
                        # Retry with new token
                        async with self.session.get(url, params=params, headers=headers) as retry_response:
                            retry_response.raise_for_status()
                            data = await retry_response.json()
                            await asyncio.sleep(Config.REQUEST_DELAY)
                            return data
                    
                    if response.status == 404:
                        logger.error(f"404 Not Found: {url}")
                        raise Exception(f"Endpoint not found: {endpoint}")
                    
                    response.raise_for_status()
                    data = await response.json()
                    
                    # Respect rate limits
                    await asyncio.sleep(Config.REQUEST_DELAY)
                    
                    return data
                    
            except aiohttp.ClientError as e:
                logger.error(f"Request failed: {url}, error: {e}")
                raise
            except Exception as e:
                logger.error(f"Request failed: {url}, error: {e}")
                raise
    
    async def search_vacancies(
        self,
        text: str = None,
        area: int = 1,  # Moscow by default
        per_page: int = 100,
        page: int = 0,
        experience: str = None,
        employment: str = None,
        schedule: str = None,
        order_by: str = 'publication_time',  # Sort by date instead of relevance
        industry: str = None  # Industry ID for filtering
    ) -> Dict:
        """Search for job vacancies."""
        params = {
            'area': area,
            'per_page': per_page,
            'page': page,
            'order_by': order_by
        }
        
        if text:
            params['text'] = text
        
        if industry:
            params['industry'] = industry
        
        if experience:
            params['experience'] = experience
        if employment:
            params['employment'] = employment
        if schedule:
            params['schedule'] = schedule
        
        return await self._make_request('vacancies', params)
    
    async def get_vacancy_details(self, vacancy_id: str) -> Dict:
        """Get detailed information about a specific vacancy."""
        return await self._make_request(f'vacancies/{vacancy_id}')
    
    async def get_areas(self) -> List[Dict]:
        """Get list of areas (cities/regions)."""
        return await self._make_request('areas')
    
    async def get_employers(self, text: str = "", per_page: int = 100, page: int = 0) -> Dict:
        """Search for employers."""
        params = {
            'text': text,
            'per_page': per_page,
            'page': page
        }
        return await self._make_request('employers', params)
    
    async def get_employer_details(self, employer_id: str) -> Dict:
        """Get detailed information about an employer."""
        return await self._make_request(f'employers/{employer_id}')
    
    async def get_specializations(self) -> List[Dict]:
        """Get list of specializations."""
        return await self._make_request('specializations')
    
    async def get_industries(self) -> List[Dict]:
        """Get list of industries and sub-industries."""
        return await self._make_request('industries')
    
    async def get_dictionaries(self) -> Dict:
        """Get all dictionaries including schedules, working hours, etc."""
        return await self._make_request('dictionaries')
    
    async def get_skills(self, text: str = "") -> List[Dict]:
        """Search for skills."""
        params = {'text': text} if text else {}
        return await self._make_request('skills', params)

