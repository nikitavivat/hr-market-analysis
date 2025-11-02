"""OAuth authorization for HH.ru API."""
import json
import webbrowser
from pathlib import Path
from typing import Optional, Dict
import requests
from loguru import logger

from .config import Config

TOKEN_DIR = Path("data/tokens")
TOKEN_PATH = TOKEN_DIR / "hh_token.json"

def get_access_token() -> Optional[str]:
    """Load access token from file."""
    if not TOKEN_PATH.exists():
        return None
    
    try:
        with open(TOKEN_PATH, 'r', encoding='utf-8') as f:
            token_data = json.load(f)
            return token_data.get('access_token')
    except Exception as e:
        logger.error(f"Error loading token: {e}")
        return None

def save_token(token_data: Dict) -> None:
    """Save token to file."""
    TOKEN_DIR.mkdir(parents=True, exist_ok=True)
    with open(TOKEN_PATH, 'w', encoding='utf-8') as f:
        json.dump(token_data, f, ensure_ascii=False, indent=2)
    logger.info("Token saved successfully")

def authorize_user() -> str:
    
    auth_url = (
        f"https://hh.ru/oauth/authorize?"
        f"response_type=code&"
        f"client_id={Config.CLIENT_ID}&"
        f"redirect_uri={Config.REDIRECT_URI}"
    )
    
    print(auth_url)
    print("-" * 60)
    
    # Try to open browser
    try:
        webbrowser.open(auth_url)
        print("\nBrowser opened automatically.")
    except:
        print("\nPlease copy the URL above and open it in your browser.")
    
    print("\nStep 2: Authorize the application and copy the CODE from the redirect URL.")
    print("The redirect URL will look like:")
    print(f"{Config.REDIRECT_URI}?code=YOUR_CODE_HERE")
    print("\nStep 3: Paste the CODE here:")
    
    code = input("Enter authorization code: ").strip()
    
    if not code:
        raise ValueError("Authorization code is required")
    
    # Exchange code for token
    logger.info("\nОбмен кода на токен...")
    token_url = "https://hh.ru/oauth/token"
    
    data = {
        "grant_type": "authorization_code",
        "client_id": Config.CLIENT_ID,
        "client_secret": Config.CLIENT_SECRET,
        "redirect_uri": Config.REDIRECT_URI,
        "code": code
    }
    
    response = requests.post(token_url, data=data, timeout=15)
    
    if response.status_code != 200:
        error_text = response.text
        raise Exception(f"Failed to get token: {response.status_code} - {error_text}")
    
    token_data = response.json()
    save_token(token_data)
    
    access_token = token_data.get('access_token')
    expires_in = token_data.get('expires_in', 'unknown')
    
    logger.info(f"\nAuthorization successful!")
    logger.info(f"  Token expires in: {expires_in} seconds")
    logger.info(f"  Token saved to: {TOKEN_PATH}")
    logger.info("="*60 + "\n")
    
    return access_token

def ensure_authorization() -> str:
    """Ensure we have valid access token, authorize if needed."""
    logger.info("Проверка авторизации...")
    token = get_access_token()
    
    if not token:
        logger.warning("Токен не найден. Требуется авторизация...")
        logger.info("Следуйте инструкциям ниже для авторизации")
        token = authorize_user()
    
    # Verify token is valid by making a test request
    logger.info("Проверка валидности токена...")
    if not verify_token(token):
        logger.warning("Токен невалиден. Требуется повторная авторизация...")
        token = authorize_user()
    
    logger.info("Авторизация успешна")
    return token

def verify_token(token: str) -> bool:
    """Verify if token is valid by making a test request."""
    try:
        headers = {
            'Authorization': f'Bearer {token}',
            'User-Agent': 'HH.ru Job Scraper/1.0'
        }
        response = requests.get(
            'https://api.hh.ru/me',
            headers=headers,
            timeout=10
        )
        return response.status_code == 200
    except:
        return False
