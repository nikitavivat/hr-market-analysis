# HR Market Analysis

Comprehensive job market analysis system through data collection from multiple platforms to identify effective directions for HR companies.

## Project Goal

Compare salary expectations and requirements across professions to identify market niches and optimize HR strategies. The project includes data collection, data merging, exploratory data analysis (EDA), and building salary prediction models.

## Data Sources

The project collects data from four main platforms:

- **HH.ru** - via official API (OAuth 2.0)
- **LinkedIn** - via Playwright (browser automation)
- **Rabota.ru** - via Playwright
- **GorodRabot** - via Selenium

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
playwright install chromium
```

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```env
# HH.ru API settings
HH_CLIENT_ID=your_client_id
HH_CLIENT_SECRET=your_client_secret
HH_REDIRECT_URI=http://127.0.0.1:5000/callback

# Delay settings (seconds)
REQUEST_DELAY=0.2
VACANCY_DETAIL_DELAY=0.2
PAGE_DELAY=0.3
MAX_CONCURRENT_REQUESTS=3

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/hh_scraper.log

# OpenAI API (for EDA with embeddings)
OPENAI_API_KEY=your_openai_api_key
```

### 3. Get HH.ru API Tokens

1. Register an application on [dev.hh.ru](https://dev.hh.ru)
2. Get `CLIENT_ID` and `CLIENT_SECRET`
3. Specify `REDIRECT_URI` in application settings

## Usage

### HH.ru Scraper

Run the main scraper to collect vacancies from HH.ru:

```bash
python run_scraper.py
```

**Features:**
- Collects up to 800 vacancies per profession (maximum 40 pages)
- Uses a list of 300+ search queries (`search_queries_300.py`)
- Automatically saves progress to `data/scraper_state.json`
- Resumes from the last processed page if interrupted
- Output: `data/raw/hh_vacancies.csv`

### LinkedIn Scraper

Run the scraper to collect vacancies from LinkedIn:

```bash
python run_linkedin_scraper.py
```

**Important:**
- The scraper will automatically open Chrome browser
- You need to log in to LinkedIn manually
- After logging in, type anything in the console and press Enter
- Output: `data/raw/linkedin_vacancies.csv`
- Saves state to `data/linkedin_scraper_state.json`

### Rabota.ru Scraper

Run the scraper to collect vacancies from Rabota.ru:

```bash
python -m src.scraper.rabota_scraper
```

**Features:**
- Collects up to 600 vacancies per profession (maximum 20 pages)
- Uses the same query list from `search_queries_300.py`
- Output: `data/raw/rabota_vacancies.csv`

### GorodRabot Scraper

Run the scraper to collect vacancies from GorodRabot:

```bash
python -m src.scraper.gorod_main_scraper
```

**Features:**
- Collects data for 50+ cities in Russia
- Output: `src/EDA/gorodrabot_listings_v3.csv`

## Output Data Structure

All scrapers save data in a unified CSV format with the following fields:

- **Название** - Job title
- **Компания** - Company name
- **Зарплата_от** - Minimum salary
- **Зарплата_до** - Maximum salary
- **Город** - City where the vacancy is located
- **Навыки** - List of required skills
- **Тип работы** - Remote/Office/Hybrid etc.
- **Описание вакансии** - Full description
- **Требования** - Candidate requirements
- **Количество откликов** - Number of responses (for HH.ru)
- **URL** - Vacancy link

## Data Analysis (EDA)

The project includes modules for exploratory data analysis in the `src/EDA/` directory:

### Data Merging

```bash
cd src/EDA
python merge_datasets.py
```

Merges data from all sources into a single dataset `merged_data.csv`:
- Removes duplicates
- Cleans data
- Creates target variable `salary_target` (average salary value)
- Removes outliers using IQR method

### Basic EDA

```bash
cd src/EDA
python eda.py
```

Performs basic data analysis:
- Missing values statistics
- Salary distribution
- City analysis
- Company analysis
- Work type analysis
- Salary transparency index

Results are saved in `plots/`:
- `01_missing_values.png` - Missing values distribution
- `02_salary_distribution.png` - Salary distribution
- `03_city_analysis.png` - City analysis
- `04_company_analysis.png` - Company analysis
- `05_work_type_analysis.png` - Work type analysis
- `06_transparency_index.png` - Transparency index

### Salary Prediction

```bash
cd src/EDA
python salary_prediction.py
```

Builds machine learning models for salary prediction:
- Uses embeddings from OpenAI API for text data
- Tests various models: RandomForest, GradientBoosting, Ridge, Lasso, ElasticNet
- Visualizes results and feature importance
- Saves plots to `plots/`

### Clustering

```bash
cd src/EDA
python clustering.py
```

Performs vacancy clustering to identify groups of similar positions.

### NLP Analysis

```bash
cd src/EDA
python nlp_analysis.py
```

Conducts analysis of textual data (job descriptions, requirements).

## Project Structure

```
GP-python/
├── src/
│   ├── api/                    # Modules for HH.ru API
│   │   ├── auth.py            # OAuth 2.0 authorization
│   │   ├── config.py          # Configuration
│   │   ├── csv_writer.py      # CSV writing
│   │   ├── data_processor.py  # Data processing
│   │   ├── hh_client.py       # HH.ru API client
│   │   ├── main.py            # Entry point
│   │   ├── scraper.py         # Main scraper class
│   │   └── state_manager.py   # State management
│   │
│   ├── scraper/               # Scrapers for other platforms
│   │   ├── linkedin_scraper.py    # LinkedIn scraper
│   │   ├── rabota_scraper.py     # Rabota.ru scraper
│   │   └── gorod_main_scraper.py # GorodRabot scraper
│   │
│   └── EDA/                   # Data analysis modules
│       ├── eda.py             # Basic EDA
│       ├── merge_datasets.py  # Data merging
│       ├── salary_prediction.py # ML models
│       ├── clustering.py     # Clustering
│       ├── nlp_analysis.py    # NLP analysis
│       └── plots/             # Plots and visualizations
│
├── data/
│   ├── raw/                   # Raw data from scrapers
│   │   ├── hh_vacancies.csv
│   │   ├── linkedin_vacancies.csv
│   │   └── rabota_vacancies.csv
│   ├── processed/             # Processed data
│   ├── backup/                # Backups
│   └── tokens/                # Tokens and cookies
│       ├── hh_token.json
│       ├── linkedin_cookies.json
│       └── linkedin_browser_profile/
│
├── logs/                      # Scraper logs
│
├── run_scraper.py            # HH.ru scraper launch script
├── run_linkedin_scraper.py   # LinkedIn scraper launch script
├── search_queries_300.py     # Search queries list (300+)
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Technical Details

### HH.ru API Scraper
- Asynchronous requests with `aiohttp`
- Parallel vacancy processing (controlled by `MAX_CONCURRENT_REQUESTS`)
- OAuth 2.0 authorization
- Rate limiting with configurable delays
- State persistence for resuming interrupted sessions

### LinkedIn Scraper
- Uses Playwright for browser automation
- Saves cookies for reuse
- Supports persistent browser context
- Automatic skill extraction from descriptions

### GorodRabot Scraper
- Selenium WebDriver for automation
- Support for multiple cities
- Random delays to simulate human behavior

## Data Scale

The project is designed to collect large volumes of data:
- **HH.ru**: up to 240,000 vacancies (300 queries × 800 vacancies)
- **Rabota.ru**: up to 180,000 vacancies (300 queries × 600 vacancies)
- **LinkedIn**: depends on data availability
- **GorodRabot**: depends on number of cities

## Logging

All scrapers use `loguru` for logging:
- Logs are saved to `logs/`
- Log rotation at 10 MB
- Log retention for 7 days
- Colored console output

## Resume Capability

All scrapers support resuming work after interruption:
- **HH.ru**: state in `data/scraper_state.json`
- **LinkedIn**: state in `data/linkedin_scraper_state.json`
- **Rabota.ru**: checks existing records in CSV

## License

MIT
