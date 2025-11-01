"""Configuration management for HH.ru API scraper."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for HH.ru API scraper."""
    
    # HH.ru API settings
    CLIENT_ID = os.getenv('HH_CLIENT_ID')
    CLIENT_SECRET = os.getenv('HH_CLIENT_SECRET')
    REDIRECT_URI = os.getenv('HH_REDIRECT_URI', 'http://127.0.0.1:5000/callback')
    API_BASE_URL = os.getenv('HH_API_BASE_URL', 'https://api.hh.ru')
    
    # Database settings
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///data/hh_jobs.db')
    
    # Request settings
    REQUEST_DELAY = float(os.getenv('REQUEST_DELAY', '0.2'))  # Delay between API requests (seconds)
    VACANCY_DETAIL_DELAY = float(os.getenv('VACANCY_DETAIL_DELAY', '0.2'))  # Delay before each vacancy detail request
    PAGE_DELAY = float(os.getenv('PAGE_DELAY', '0.3'))  # Delay between pages
    MAX_CONCURRENT_REQUESTS = int(os.getenv('MAX_CONCURRENT_REQUESTS', '3'))  # Reduced to avoid rate limits
    
    # Logging settings
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'logs/hh_scraper.log')
    
    # Data collection settings
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', '100'))
    MAX_PAGES_PER_SEARCH = int(os.getenv('MAX_PAGES_PER_SEARCH', '100'))
    
    # Ensure directories exist
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist."""
        Path('data').mkdir(exist_ok=True)
        Path('logs').mkdir(exist_ok=True)
        Path('data/raw').mkdir(exist_ok=True)
        Path('data/processed').mkdir(exist_ok=True)

