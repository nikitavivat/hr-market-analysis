"""Data processing and transformation utilities."""
import re
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from loguru import logger

class DataProcessor:
    """Processes and transforms job vacancy data."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text data."""
        if not text:
            return ""
        
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:()-]', '', text)
        
        return text.strip()
    
    @staticmethod
    def extract_skills_from_api(raw_data: Dict) -> List[str]:
        """Extract skills from HH.ru API key_skills field."""
        key_skills = raw_data.get('key_skills', [])
        if not key_skills:
            return []
        
        skills = []
        for skill in key_skills:
            if isinstance(skill, dict):
                skill_name = skill.get('name', '')
                if skill_name:
                    skills.append(skill_name)
            elif isinstance(skill, str):
                skills.append(skill)
        
        return skills
    
    @staticmethod
    def parse_salary(salary_data: Dict) -> tuple:
        """Parse salary information from HH.ru API response."""
        if not salary_data:
            return None, None, None, None
        
        salary_from = salary_data.get('from')
        salary_to = salary_data.get('to')
        currency = salary_data.get('currency', 'RUR')
        gross = salary_data.get('gross', True)
        
        return salary_from, salary_to, currency, gross
    
    @staticmethod
    def extract_company_info(employer_data: Dict) -> tuple:
        """Extract company information."""
        if not employer_data:
            return None, None, None
        
        company_name = employer_data.get('name')
        company_id = employer_data.get('id')
        company_url = employer_data.get('alternate_url')
        
        return company_name, company_id, company_url
    
    @staticmethod
    def process_vacancy_data(raw_data: Dict) -> Dict:
        """Process raw vacancy data from HH.ru API."""
        try:
            hh_id = str(raw_data.get('id', ''))
            name = raw_data.get('name', '')
            
            description = raw_data.get('description', '')
            requirements = raw_data.get('requirement', '')
            responsibilities = raw_data.get('responsibility', '')
            
            description_cleaned = DataProcessor.clean_text(description)
            requirements_cleaned = DataProcessor.clean_text(requirements)
            responsibilities_cleaned = DataProcessor.clean_text(responsibilities)
            
            skills = DataProcessor.extract_skills_from_api(raw_data)
            
            salary_data = raw_data.get('salary')
            salary_from, salary_to, salary_currency, salary_gross = DataProcessor.parse_salary(salary_data)
            
            employer_data = raw_data.get('employer', {})
            company_name, company_id, company_url = DataProcessor.extract_company_info(employer_data)
            
            area_data = raw_data.get('area', {})
            area_name = area_data.get('name', '')
            area_id = area_data.get('id', '')
            
            experience_data = raw_data.get('experience', {})
            experience_name = experience_data.get('name', '')
            
            employment_data = raw_data.get('employment', {})
            employment_name = employment_data.get('name', '')
            
            schedule_data = raw_data.get('schedule', {})
            schedule_name = schedule_data.get('name', '')
            
            specializations = []
            for spec in raw_data.get('specializations', []):
                specializations.append(spec.get('name', ''))
            
            published_at = None
            if raw_data.get('published_at'):
                try:
                    published_at = datetime.fromisoformat(
                        raw_data['published_at'].replace('Z', '+00:00')
                    )
                except:
                    pass
            
            responses_count = None
            counters = raw_data.get('counters', {})
            if isinstance(counters, dict):
                responses_count = counters.get('responses')
            elif isinstance(counters, list) and len(counters) > 0:
                for counter in counters:
                    if isinstance(counter, dict) and counter.get('type') == 'responses':
                        responses_count = counter.get('value')
                        break
            
            if responses_count is None:
                responses_count = raw_data.get('responses_count')
            
            if responses_count is None:
                logger.debug(f"Responses count not available for vacancy {raw_data.get('id', 'unknown')}")
            
            return {
                'hh_id': hh_id,
                'name': name,
                'description': description_cleaned,
                'requirements': requirements_cleaned,
                'responsibilities': responsibilities_cleaned,
                'skills': skills,
                'salary_from': salary_from,
                'salary_to': salary_to,
                'salary_currency': salary_currency,
                'salary_gross': salary_gross,
                'company_name': company_name,
                'company_id': company_id,
                'company_url': company_url,
                'area_name': area_name,
                'area_id': area_id,
                'experience_name': experience_name,
                'employment_name': employment_name,
                'schedule_name': schedule_name,
                'specializations': specializations,
                'published_at': published_at,
                'responses_count': responses_count
            }
            
        except Exception as e:
            logger.error(f"Error processing vacancy data: {e}")
            return None
