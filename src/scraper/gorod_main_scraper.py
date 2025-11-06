"""
GorodRabot main‑page scraper (version 3)

This script collects vacancy listings from the public search results of
``gorodrabot.ru`` without following links to individual vacancy
pages.  It uses Selenium to render each search results page and
BeautifulSoup to extract the vacancy snippets.  The goal of this
version is to generate a dataset with **at least 15 000 entries** while
staying within reasonable request limits.  To achieve this we:

* Expand the list of city subdomains to cover many regions of Russia.
  Each subdomain corresponds to a different geographic region on
  GorodRabot.  By scraping multiple cities we can collect more
  vacancies without stressing any single subdomain too heavily.
* Increase the number of pages scraped per city.  Each results page
  typically lists 20 vacancies.  Scraping 80 pages per city yields
  1 600 vacancies per city on average.
* Randomise delays between requests and insert periodic long breaks
  after a set number of pages.  This reduces the likelihood of
  encountering HTTP 429 or temporary bans.  You can adjust the
  ``SLEEP_RANGE``, ``BREAK_INTERVAL`` and ``LONG_BREAK_DURATION``
  constants below based on your network environment.
* Log progress and errors to a file (``list_scraper_v3.log``) and the
  console.  Logging includes counts of vacancies found per page and
  per city, as well as total rows scraped.

The output dataset is saved in ``gorodrabot_listings_v3.csv``.  Each
row contains these columns:

    * ``name`` – vacancy title
    * ``company`` – employer name
    * ``city`` – region/city name (derived from the subdomain)
    * ``salary_raw`` – textual salary field as seen on the site
    * ``salary_from`` – numeric lower bound of salary if present
    * ``salary_to`` – numeric upper bound of salary if present
    * ``salary_avg`` – average of salary_from and salary_to
    * ``work_type`` – employment type (if available)
    * ``description`` – short description snippet from the listing
    * ``url`` – hyperlink to the vacancy (for reference)

Usage::

    python gorod_main_scraper_v3.py

To customise the scope of scraping (for example, to reduce runtime),
edit the ``CITY_SLUGS``, ``PAGE_START`` and ``PAGE_END`` variables.

Dependencies::

    pip install selenium webdriver-manager beautifulsoup4 pandas

Note: Running this script will open a visible Chrome window unless
``--headless`` is added in the code.  Headless mode can be toggled
via the commented argument in the script below.
"""

from __future__ import annotations

import logging
import os
import random
import re
import time
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager


# ---------------------------------------------------------------------------
# Configuration
#
# List of city subdomains.  Extending this list increases the number of
# vacancies collected.  To meet the 15 k target, we've included a wide
# selection of Russian cities.  Feel free to remove or add slugs to
# balance runtime and volume.
CITY_SLUGS: List[str] = [
    "moskva", "sankt-peterburg", "nijni-novgorod", "ekaterinburg",
    "novosibirsk", "kazan", "samara", "rostov-na-donu", "ufa",
    "perm", "voronezh", "volgograd", "sochi", "krasnodar", "tula",
    "yaroslavl", "chelyabinsk", "krasnoyarsk", "tyumen", "barnaul",
    "irkutsk", "omsk", "vladivostok", "vologda", "tver", "kirov",
    "smolensk", "belgorod", "ivanovo", "kaluga", "lipetsk", "orel",
    "ryazan", "saratov", "penza", "ulan-ude", "pskov", "murmansk",
    "sevastopol", "ulyanovsk", "grozny", "naberezhnye-chelny", "kursk",
    "sergiev-posad", "taganrog", "astrakhan", "magnitogorsk", "rybinsk",
]

# Page range to scrape per city.  Each page contains around 20 vacancy
# snippets.  With PAGE_END=80, we expect roughly 20 × 80 = 1 600 rows
# per city; across 50 cities this can yield ~80 000 rows.  Adjust
# downwards if your network is slow or you encounter rate limits.
PAGE_START: int = 1
PAGE_END: int = 80

# Random delay range between page requests (seconds).  Increase
# ``SLEEP_RANGE`` to be more polite; decrease to speed up scraping at
# the risk of hitting 429 errors.
SLEEP_RANGE: Tuple[float, float] = (3.0, 6.0)

# Pause for a longer break after every BREAK_INTERVAL pages to avoid
# triggering anti‑bot protection.  When BREAK_INTERVAL pages have
# been processed, the scraper will sleep for LONG_BREAK_DURATION seconds.
BREAK_INTERVAL: int = 10
LONG_BREAK_DURATION: float = 25.0

# Filenames for CSV output and logs.
CSV_FILENAME: str = "gorodrabot_listings_v3.csv"
LOG_FILENAME: str = "list_scraper_v3.log"

# Configure logging to file and console simultaneously.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILENAME, mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)


# ---------------------------------------------------------------------------
# Data model definition
#
@dataclass
class Listing:
    name: Optional[str]
    company: Optional[str]
    city: Optional[str]
    salary_raw: Optional[str]
    salary_from: Optional[float]
    salary_to: Optional[float]
    salary_avg: Optional[float]
    work_type: Optional[str]
    description: Optional[str]
    url: Optional[str]


# ---------------------------------------------------------------------------
# Helper functions
#
def parse_salary(text: Optional[str]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Extract numeric salary bounds from a salary string.

    Returns a tuple ``(salary_from, salary_to, salary_avg)``.  If only a
    single value is present, it will be used for all three.  Returns
    ``(None, None, None)`` if no numbers are found.
    """
    if not text:
        return None, None, None
    t = text.replace("\xa0", " ").replace("\u202f", " ")
    nums = [float("".join(re.findall(r"\d", part))) for part in re.findall(r"\d[\d\s]*", t)]
    salary_from = salary_to = salary_avg = None
    if "от" in t and nums:
        salary_from = nums[0]
    if "до" in t and nums:
        salary_to = nums[0]
    if ("–" in t or "-" in t) and len(nums) >= 2:
        salary_from, salary_to = nums[0], nums[1]
    if salary_from and salary_to:
        salary_avg = (salary_from + salary_to) / 2.0
    elif salary_from:
        salary_avg = salary_from
    elif salary_to:
        salary_avg = salary_to
    return salary_from, salary_to, salary_avg


def random_sleep() -> None:
    """Sleep for a random duration within SLEEP_RANGE."""
    time.sleep(random.uniform(*SLEEP_RANGE))


def accept_cookie(driver: webdriver.Chrome) -> None:
    """Click the cookie consent button if present."""
    try:
        btn = driver.find_element(By.XPATH, "//button[contains(text(), 'Понятно')]")
        btn.click()
        time.sleep(0.3)
    except Exception:
        pass


def collect_listings(driver: webdriver.Chrome, city_slug: str) -> List[Listing]:
    """Parse all vacancy snippets on the current page into Listing objects."""
    soup = BeautifulSoup(driver.page_source, "html.parser")
    listings: List[Listing] = []
    snippets = soup.find_all("div", class_="vacancy")
    for snippet in snippets:
        classes = snippet.get("class", [])
        if "snippet" not in classes:
            continue
        title_el = snippet.find("a", class_="snippet__title-link")
        name = title_el.get_text(strip=True) if title_el else None
        url = title_el.get("href") if title_el else None
        company_el = snippet.find("li", class_="snippet__meta-item_company")
        company = (
            company_el.find("span", class_="snippet__meta-value").get_text(strip=True)
            if company_el
            else None
        )
        city_el = snippet.find("li", class_="snippet__meta-item_location")
        city_name = (
            city_el.find("span", class_="snippet__meta-value").get_text(strip=True)
            if city_el
            else city_slug.replace("-", " ")
        )
        salary_el = snippet.find("span", class_="snippet__salary")
        salary_raw = salary_el.get_text(" ", strip=True) if salary_el else None
        work_type_el = snippet.find("li", class_="snippet__meta-item_type")
        work_type = (
            work_type_el.find("span", class_="snippet__meta-value").get_text(strip=True)
            if work_type_el
            else None
        )
        desc_el = snippet.find("div", class_="snippet__desc")
        description = desc_el.get_text(" ", strip=True) if desc_el else None
        s_from, s_to, s_avg = parse_salary(salary_raw)
        listings.append(
            Listing(
                name=name,
                company=company,
                city=city_name,
                salary_raw=salary_raw,
                salary_from=s_from,
                salary_to=s_to,
                salary_avg=s_avg,
                work_type=work_type,
                description=description,
                url=url,
            )
        )
    return listings


# ---------------------------------------------------------------------------
# Main scraping routine
#
def main() -> None:
    logging.info("Starting GorodRabot listing scraper (v3)")
    options = Options()
    # Uncomment the next line to run headless (no visible browser window)
    # options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    all_rows: List[Listing] = []
    try:
        processed_pages = 0
        for city_slug in CITY_SLUGS:
            logging.info("Scraping city: %s", city_slug.replace("-", " "))
            for page in range(PAGE_START, PAGE_END + 1):
                url = f"https://{city_slug}.gorodrabot.ru/"
                if page != 1:
                    url += f"?p={page}"
                logging.info("Opening page %d of %s: %s", page, city_slug, url)
                try:
                    driver.get(url)
                except Exception as e:
                    logging.warning("Failed to open %s: %s", url, e)
                    random_sleep()
                    continue
                accept_cookie(driver)
                try:
                    WebDriverWait(driver, 15).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, "div.vacancy.snippet"))
                    )
                except Exception:
                    logging.warning("Timeout waiting for content on %s", url)
                    random_sleep()
                    continue
                rows = collect_listings(driver, city_slug)
                logging.info(
                    "Found %d listings on page %d of %s",
                    len(rows),
                    page,
                    city_slug,
                )
                all_rows.extend(rows)
                processed_pages += 1
                # Take a longer break every BREAK_INTERVAL pages
                if processed_pages % BREAK_INTERVAL == 0:
                    logging.info(
                        "Taking a long break (%d pages processed so far). Sleeping %.1f seconds",
                        processed_pages,
                        LONG_BREAK_DURATION,
                    )
                    time.sleep(LONG_BREAK_DURATION)
                random_sleep()
    finally:
        try:
            driver.quit()
        except Exception:
            pass
    logging.info("Scraped a total of %d rows (before deduplication)", len(all_rows))
    df = pd.DataFrame([asdict(r) for r in all_rows])
    df.drop_duplicates(subset=["name", "company", "city", "url"], inplace=True)
    logging.info("Total rows after deduplication: %d", len(df))
    df.to_csv(CSV_FILENAME, index=False)
    logging.info("Saved deduplicated data to %s", CSV_FILENAME)


if __name__ == "__main__":
    main()