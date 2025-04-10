"""
Fix the project structure and ensure the marketsense package is properly organized.
"""
import os
import shutil
from pathlib import Path
import sys

# Current directory (should be your project root)
current_dir = Path.cwd()
print(f"Current directory: {current_dir}")

# Create the marketsense package directory if it doesn't exist
marketsense_dir = current_dir / "marketsense"
if not marketsense_dir.exists():
    marketsense_dir.mkdir()
    print(f"Created directory: {marketsense_dir}")

# Create required subdirectories
subdirs = ["core", "modules", "utils", "data", "data/cache", "data/outputs"]
for subdir in subdirs:
    subdir_path = marketsense_dir / subdir
    if not subdir_path.exists():
        subdir_path.mkdir(parents=True)
        print(f"Created directory: {subdir_path}")

# Create __init__.py files
init_files = [
    marketsense_dir / "__init__.py",
    marketsense_dir / "core" / "__init__.py",
    marketsense_dir / "modules" / "__init__.py",
    marketsense_dir / "utils" / "__init__.py"
]

for init_file in init_files:
    if not init_file.exists():
        with open(init_file, "w") as f:
            f.write('"""MarketSense AI package."""\n\n__version__ = "1.0.0"\n')
        print(f"Created file: {init_file}")

# Create a minimal config.py if it doesn't exist
config_path = marketsense_dir / "config.py"
if not config_path.exists():
    with open(config_path, "w") as f:
        f.write('''"""
Configuration module for MarketSense AI.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")

# Application settings
FLASK_ENV = os.getenv("FLASK_ENV", "development")
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "1") == "1"
SECRET_KEY = os.getenv("SECRET_KEY", "default_secret_key")

# Paths
CACHE_DIR = Path(os.getenv("CACHE_DIR", DATA_DIR / "cache"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", DATA_DIR / "outputs"))

# Ensure directories exist
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# AI models configuration
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
NLP_BATCH_SIZE = int(os.getenv("NLP_BATCH_SIZE", "10"))
CACHE_EXPIRY_DAYS = int(os.getenv("CACHE_EXPIRY_DAYS", "7"))
''')
    print(f"Created file: {config_path}")

# Create minimal cache.py if it doesn't exist
cache_path = marketsense_dir / "utils" / "cache.py"
if not cache_path.exists():
    with open(cache_path, "w") as f:
        f.write('''"""
Caching utilities for MarketSense AI.
"""

import os
import json
import time
import hashlib
from functools import wraps

def get_cache_path(cache_key, cache_dir="./data/cache", suffix=".json"):
    """Get path for a cache file with proper sanitization."""
    # Create a safe filename from the cache key
    safe_key = hashlib.md5(cache_key.encode()).hexdigest()
    return os.path.join(cache_dir, f"{safe_key}{suffix}")

def is_cache_valid(cache_path, expiry_days=7):
    """Check if a cache file exists and is still valid."""
    if not os.path.exists(cache_path):
        return False
    
    # Check if cache is still valid
    file_age = time.time() - os.path.getmtime(cache_path)
    return file_age < expiry_days * 86400  # Convert days to seconds

def save_to_cache(data, cache_path):
    """Save data to a cache file."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    with open(cache_path, 'w') as f:
        json.dump(data, f)

def load_from_cache(cache_path):
    """Load data from a cache file."""
    with open(cache_path, 'r') as f:
        return json.load(f)

def cached(key_fn=None, expiry_days=7):
    """
    Decorator to cache function results.
    
    Args:
        key_fn: Function to generate cache key from args and kwargs
        expiry_days: Cache expiry in days
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_fn:
                cache_key = key_fn(*args, **kwargs)
            else:
                # Default cache key based on function name and args
                params = str(args) + str(sorted(kwargs.items()))
                cache_key = f"{func.__name__}_{hash(params)}"
            
            cache_path = get_cache_path(cache_key)
            
            # Return cached result if valid
            if is_cache_valid(cache_path, expiry_days):
                return load_from_cache(cache_path)
            
            # Calculate result and cache it
            result = func(*args, **kwargs)
            save_to_cache(result, cache_path)
            return result
        
        return wrapper
    
    return decorator
''')
    print(f"Created file: {cache_path}")

# Save the ai_engine.py content
ai_engine_path = marketsense_dir / "core" / "ai_engine.py"
ai_engine_content = """# AI-powered analysis engine
\"\"\"
AI Engine for market analysis using OpenAI API.
\"\"\"

import os
import json
import openai
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import time

from marketsense import config
from marketsense.utils.cache import cached, get_cache_path, is_cache_valid, save_to_cache, load_from_cache

class AIEngine:
    \"\"\"Advanced AI Engine to interact with OpenAI API for market analysis.\"\"\"
    
    def __init__(self):
        \"\"\"Initialize the AI Engine with OpenAI API key.\"\"\"
        openai.api_key = config.OPENAI_API_KEY
        self.model = config.OPENAI_MODEL
    
    @cached(lambda self, company_name, industry, depth: f"{company_name}_{industry}_{depth}_analysis")
    def generate_market_analysis(self, company_name: str, industry: str, 
                               depth: str = "comprehensive") -> Dict[str, Any]:
        \"\"\"
        Generate market analysis for a specific company in an industry.
        
        Args:
            company_name: Name of the company to analyze
            industry: Industry sector of the company
            depth: Analysis depth ("basic", "comprehensive", "expert")
            
        Returns:
            Dictionary containing market analysis data
        \"\"\"
        # Create prompt based on requested depth
        if depth == "basic":
            sections = ["Company Overview", "SWOT Analysis", "Key Competitors"]
        elif depth == "comprehensive":
            sections = ["Company Overview", "SWOT Analysis", "Market Position", 
                      "Key Competitors", "Growth Opportunities", "Risks and Challenges"]
        else:  # expert
            sections = ["Company Overview", "SWOT Analysis", "Market Position", 
                      "Key Competitors", "Growth Opportunities", "Risks and Challenges",
                      "Financial Analysis", "Strategic Recommendations", "Technology Assessment"]
        
        # Format sections for prompt
        sections_text = "\\n".join([f"{i+1}. {section}" for i, section in enumerate(sections)])
        
        # Create prompt for market analysis
        prompt = f\"\"\"
        As a market intelligence expert, provide a {depth} analysis for {company_name} in the {industry} industry.
        Include the following sections:
        {sections_text}
        
        For the SWOT analysis, be specific and detailed with actionable insights.
        Format the response as structured JSON with these sections as keys.
        \"\"\"
        
        try:
            # For demo purposes, create mock data (in production, call OpenAI API)
            result = self._generate_mock_analysis(company_name, industry, depth, sections)
            
            # In production, use:
            # response = openai.ChatCompletion.create(
            #     model=self.model,
            #     messages=[
            #         {"role": "system", "content": "You are a market analysis expert providing detailed insights."},
            #         {"role": "user", "content": prompt}
            #     ],
            #     temperature=0.2,
            #     max_tokens=1500
            # )
            # result = json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error generating analysis: {str(e)}")
            # Create basic fallback data
            result = {section: f"No {section.lower()} data available" for section in sections}
            if "SWOT Analysis" in sections:
                result["SWOT Analysis"] = {"Strengths": [], "Weaknesses": [], 
                                         "Opportunities": [], "Threats": []}
        
        return result
    
    @cached(lambda self, company, competitors, industry="": f"{company}_vs_{'_'.join(competitors)}_{industry}")
    def analyze_competitors(self, company: str, competitors: List[str], 
                          industry: str = "") -> Dict[str, Any]:
        \"\"\"
        Generate advanced comparative analysis between a company and its competitors.
        
        Args:
            company: Main company name
            competitors: List of competitor company names
            industry: Optional industry context
            
        Returns:
            Dictionary containing competitive analysis data
        \"\"\"
        # Create prompt for competitor analysis with industry context
        industry_context = f" in the {industry} industry" if industry else ""
        competitors_list = ", ".join(competitors)
        
        prompt = f\"\"\"
        As a competitive intelligence expert, provide a detailed analysis comparing {company} with these competitors: {competitors_list}{industry_context}.
        
        Include the following in your analysis:
        1. Comparative market share and positioning
        2. Detailed strengths and weaknesses assessment for each company
        3. Product/service portfolio comparison with feature analysis
        4. Pricing strategy differences and price-to-value analysis
        5. Technology and innovation comparison with forward-looking assessment
        6. Customer experience and brand perception comparison
        7. Distribution channels and market reach assessment
        
        Format the response as structured JSON with these sections as keys and include specific metrics where possible.
        \"\"\"
        
        # For demo purposes, create mock data (in production, call OpenAI API)
        # This would be replaced with actual API call in production version
        result = self._generate_mock_competitor_analysis(company, competitors, industry)
        
        return result
    
    @cached(lambda self, industry, timeframe: f"{industry}_{timeframe}_trends")
    def identify_market_trends(self, industry: str, timeframe: str = "current") -> Dict[str, Any]:
        \"\"\"
        Identify key market trends in a specific industry.
        
        Args:
            industry: Industry to analyze
            timeframe: Trends timeframe ("current", "emerging", "future")
            
        Returns:
            Dictionary containing trend analysis data
        \"\"\"
        # Create prompt based on timeframe
        if timeframe == "current":
            scope = "currently impacting"
        elif timeframe == "emerging":
            scope = "emerging in the next 1-2 years"
        else:  # future
            scope = "likely to shape the industry in the next 3-5 years"
        
        prompt = f\"\"\"
        As a market trends analyst, identify the key trends {scope} the {industry} industry.
        
        Include:
        1. Major market trends with impact assessment (high/medium/low)
        2. Technology trends changing the industry
        3. Consumer behavior shifts
        4. Regulatory developments
        5. Competitive landscape evolution
        
        For each trend, provide specific examples, impact assessment, and strategic implications.
        Format the response as structured JSON with these categories as keys.
        \"\"\"
        
        # For demo purposes, create mock data (in production, call OpenAI API)
        result = self._generate_mock_trends(industry, timeframe)
        
        return result
    
    def generate_competitive_strategy(self, company: str, industry: str, 
                                   competitors: List[str], objective: str) -> Dict[str, Any]:
        \"\"\"
        Generate a competitive strategy based on market analysis.
        
        Args:
            company: Company name
            industry: Industry sector
            competitors: List of main competitors
            objective: Strategic objective (e.g., "market_share_growth", "product_innovation")
            
        Returns:
            Dictionary containing strategic recommendations
        \"\"\"
        # Check cache
        cache_key = f"{company}_{industry}_{objective}_{'_'.join(competitors)}_strategy"
        cache_path = get_cache_path(cache_key)
        
        if is_cache_valid(cache_path):
            return load_from_cache(cache_path)
        
        competitors_str = ", ".join(competitors)
        
        # Map objective to specific prompt focus
        objective_focus = {
            "market_share_growth": "increasing market share and competitive positioning",
            "product_innovation": "accelerating product innovation and development",
            "cost_efficiency": "improving operational efficiency and cost structure",
            "geographic_expansion": "expanding into new geographic markets",
            "customer_retention": "improving customer retention and loyalty",
            "digital_transformation": "implementing digital transformation initiatives"
        }.get(objective, "improving overall competitive position")
        
        prompt = f\"\"\"
        As a strategic consultant, develop a competitive strategy for {company} in the {industry} industry focused on {objective_focus}.
        
        Consider these competitors: {competitors_str}
        
        Include:
        1. Strategic overview and objectives
        2. Competitive advantages to leverage
        3. Key initiatives and action items (short-term and long-term)
        4. Resource requirements and implementation approach
        5. Success metrics and KPIs
        6. Risk assessment and mitigation strategies
        
        Format the response as structured JSON with these sections as keys.
        \"\"\"
        
        # For demo purposes, create mock data (in production, call OpenAI API)
        result = self._generate_mock_strategy(company, industry, competitors, objective)
        
        # Cache the result
        save_to_cache(result, cache_path)
        
        return result
    
    # Private methods for generating mock data (would be replaced with API calls in production)
    
    def _generate_mock_analysis(self, company: str, industry: str, depth: str, sections: List[str]) -> Dict[str, Any]:
        \"\"\"Generate mock analysis data for demonstration purposes.\"\"\"
        result = {}
        
        if "Company Overview" in sections:
            result["Company Overview"] = f"{company} is a leading company in the {industry} industry, known for innovative products and solutions."
        
        if "SWOT Analysis" in sections:
            result["SWOT Analysis"] = {
                "Strengths": [
                    f"Strong brand recognition within the {industry} space",
                    "Innovative product development capabilities",
                    "Robust supply chain network",
                    "Strong financial position"
                ],
                "Weaknesses": [
                    "Higher production costs compared to key competitors",
                    "Limited product diversification",
                    "Geographic market concentration",
                    "Aging technology infrastructure"
                ],
                "Opportunities": [
                    "Expansion into emerging markets",
                    "Strategic partnerships with complementary businesses",
                    "Digital transformation opportunities",
                    "Development of subscription-based revenue streams"
                ],
                "Threats": [
                    "Increasing competition from established players",
                    "Emerging disruptive technologies",
                    "Changing regulatory environment",
                    "Economic uncertainty"
                ]
            }
        
        if "Market Position" in sections:
            result["Market Position"] = f"{company} currently holds approximately 15-18% market share in the {industry} sector."
        
        if "Key Competitors" in sections:
            result["Key Competitors"] = [
                {"name": "Industry Leader Corp", "market_share": "22%", "primary_strength": "Extensive market reach"},
                {"name": "Innovation Tech", "market_share": "14%", "primary_strength": "Cutting-edge technology"},
                {"name": "Value Solutions", "market_share": "12%", "primary_strength": "Competitive pricing"}
            ]
        
        if "Growth Opportunities" in sections:
            result["Growth Opportunities"] = [
                {"opportunity": "Expansion into Asian markets", "potential_impact": "High", "timeframe": "Medium-term"},
                {"opportunity": "Development of AI-enhanced products", "potential_impact": "High", "timeframe": "Long-term"},
                {"opportunity": "Strategic acquisitions of smaller competitors", "potential_impact": "Medium", "timeframe": "Short-term"}
            ]
        
        if "Risks and Challenges" in sections:
            result["Risks and Challenges"] = [
                {"risk": "Increasing raw material costs", "severity": "Medium", "probability": "High"},
                {"risk": "New market entrants with disruptive models", "severity": "High", "probability": "Medium"},
                {"risk": "Changing consumer preferences", "severity": "Medium", "probability": "Medium"}
            ]
        
        # Expert-level sections
        if depth == "expert":
            if "Financial Analysis" in sections:
                result["Financial Analysis"] = {
                    "Revenue Growth": "8.5% YoY",
                    "Profit Margin": "15.3%",
                    "ROI": "12.7%",
                    "Debt-to-Equity": "0.38",
                    "Assessment": f"{company} shows strong financial performance with above-industry-average growth."
                }
                
            if "Strategic Recommendations" in sections:
                result["Strategic Recommendations"] = [
                    {"recommendation": "Accelerate digital transformation initiatives", "priority": "High", "expected_impact": "Improved operational efficiency"},
                    {"recommendation": "Expand product portfolio through targeted acquisitions", "priority": "Medium", "expected_impact": "Increased market share"},
                    {"recommendation": "Invest in sustainable manufacturing practices", "priority": "Medium", "expected_impact": "Enhanced brand reputation"}
                ]
                
            if "Technology Assessment" in sections:
                result["Technology Assessment"] = {
                    "Current State": f"{company}'s technology stack is moderately advanced with some legacy systems.",
                    "Key Technologies": ["Cloud infrastructure", "Data analytics", "Automation systems", "CRM"],
                    "Gap Analysis": "Significant opportunities exist in AI implementation and advanced analytics.",
                    "Recommendations": [
                        "Implement AI-driven predictive maintenance",
                        "Migrate core systems to cloud architecture",
                        "Invest in advanced data analytics capabilities"
                    ]
                }
        
        return result
    
    def _generate_mock_competitor_analysis(self, company: str, competitors: List[str], industry: str) -> Dict[str, Any]:
        \"\"\"Generate simplified mock competitor analysis data.\"\"\"
        all_companies = [company] + competitors
        
        # Generate mock market shares - simplified version
        company_share = 25
        competitor_shares = [15] * len(competitors)  # Equal shares for competitors
        
        # Structure the result (simplified version)
        result = {
            "Comparative Market Share": {
                company: f"{company_share}%",
                **{competitors[i]: f"{share}%" for i, share in enumerate(competitor_shares)}
            },
            "Strengths and Weaknesses": {
                company: {
                    "Strengths": [
                        "Strong brand reputation",
                        "Innovative product development",
                        "Robust customer loyalty"
                    ],
                    "Weaknesses": [
                        "Higher cost structure",
                        "Limited product range in some segments",
                        "Regional market concentration"
                    ]
                }
            },
            "Product/Service Comparison": {},
            "Pricing Strategy Differences": {},
            "Technology and Innovation": {},
            "Customer Experience": {},
            "Distribution Channels": {}
        }
        
        # Generate simplified data for competitors
        strength_options = [
            "Competitive pricing", "Wide distribution network", "Strong digital presence",
            "Cost-efficient operations", "Extensive product range", "R&D focus"
        ]
        
        weakness_options = [
            "Limited market reach", "Product quality inconsistencies", "Weak brand recognition",
            "High employee turnover", "Outdated technology systems", "Limited financial resources"
        ]
        
        # Generate unique strengths and weaknesses for each competitor
        for competitor in competitors:
            # Pick 3 random strengths and weaknesses without repetition
            selected_strengths = np.random.choice(strength_options, size=3, replace=False)
            selected_weaknesses = np.random.choice(weakness_options, size=3, replace=False)
            
            result["Strengths and Weaknesses"][competitor] = {
                "Strengths": selected_strengths.tolist(),
                "Weaknesses": selected_weaknesses.tolist()
            }
        
        # Add basic product comparison data
        aspects = ["Features", "Quality", "Support", "Innovation"]
        ratings = ["Excellent", "Good", "Average"]
        
        for aspect in aspects:
            result["Product/Service Comparison"][aspect] = {}
            for comp in all_companies:
                result["Product/Service Comparison"][aspect][comp] = np.random.choice(ratings)
        
        # Add basic pricing strategy data
        strategies = ["Premium", "Value-based", "Competitive", "Economy"]
        for comp in all_companies:
            strategy = np.random.choice(strategies)
            result["Pricing Strategy Differences"][comp] = f"{strategy} pricing strategy"
        
        # Add basic technology data
        tech_areas = ["AI", "Cloud", "Mobile", "IoT", "Automation"]
        for comp in all_companies:
            focus_areas = np.random.choice(tech_areas, size=3, replace=False)
            result["Technology and Innovation"][comp] = {
                "Investment Level": np.random.choice(["High", "Medium", "Low"]),
                "Focus Areas": focus_areas.tolist(),
                "Innovation Rate": np.random.choice(["Above average", "Average", "Below average"])
            }
        
        # Add basic customer experience data
        for comp in all_companies:
            result["Customer Experience"][comp] = {
                "Service Quality": np.random.choice(ratings),
                "Support Responsiveness": np.random.choice(ratings),
                "User Interface": np.random.choice(ratings)
            }
        
        # Add basic distribution channel data
        channels = ["Direct Sales", "Retail", "E-commerce", "Distributors"]
        for comp in all_companies:
            result["Distribution Channels"][comp] = {}
            for channel in channels:
                result["Distribution Channels"][comp][channel] = np.random.choice(["Strong", "Moderate", "Limited"])
        
        return result
    
    def _generate_mock_trends(self, industry: str, timeframe: str) -> Dict[str, Any]:
        \"\"\"Generate simplified mock trends data.\"\"\"
        # Define basic trends by industry (shortened version)
        industry_trends = {
            "Technology": [
                "Cloud computing adoption",
                "AI and machine learning integration",
                "Cybersecurity investment",
                "Remote work technology"
            ],
            "Healthcare": [
                "Telemedicine expansion",
                "AI in diagnostics",
                "Wearable health devices",
                "Personalized medicine"
            ],
            "Retail": [
                "E-commerce growth",
                "Omnichannel strategies",
                "Contactless payment",
                "AI-driven personalization"
            ],
            "Finance": [
                "Digital payment growth",
                "Blockchain adoption",
                "Robo-advisory services",
                "Open banking"
            ],
            "Automotive": [
                "Electric vehicle adoption",
                "Autonomous driving",
                "Connected car services",
                "Mobility-as-a-Service"
            ]
        }
        
        # Default trends if industry not found
        default_trends = [
            "Digital transformation",
            "Data analytics adoption",
            "Sustainability focus",
            "Process automation"
        ]
        
        # Select trends based on industry and timeframe
        all_trends = industry_trends.get(industry, default_trends)
        
        # Select 3 trends regardless of timeframe (simplified)
        selected_trends = np.random.choice(all_trends, size=min(3, len(all_trends)), replace=False)
        
        # Build simplified result structure
        result = {
            "Major Market Trends": [],
            "Technology Trends": [],
            "Consumer Behavior Shifts": [],
            "Regulatory Developments": [],
            "Competitive Landscape Evolution": []
        }
        
        # Populate major market trends
        impact_levels = ["High", "Medium", "Low"]
        probabilities = ["Very Likely", "Likely", "Possible"]
        
        for trend in selected_trends:
            impact = np.random.choice(impact_levels)
            probability = np.random.choice(probabilities)
            
            result["Major Market Trends"].append({
                "trend": trend,
                "impact": impact,
                "probability": probability,
                "description": f"This trend is reshaping the {industry} industry.",
                "strategic_implications": [
                    f"Companies should evaluate capabilities related to {trend.lower()}",
                    f"Investment in training and technology may be required"
                ]
            })
        
        # Add basic tech trends
        tech_trends = ["AI adoption", "Cloud migration", "Automation", "Data analytics"]
        for _ in range(2):
            trend = np.random.choice(tech_trends)
            tech_trends.remove(trend)
            
            result["Technology Trends"].append({
                "trend": trend,
                "impact": np.random.choice(impact_levels),
                "adoption_stage": np.random.choice(["Early", "Growing", "Mainstream"])
            })
        
        # Add basic consumer behaviors
        behaviors = ["Personalization demand", "Sustainability focus", "Digital-first interactions"]
        for behavior in behaviors[:2]:
            result["Consumer Behavior Shifts"].append({
                "shift": behavior,
                "impact": np.random.choice(impact_levels),
                "pace_of_change": np.random.choice(["Rapid", "Moderate", "Gradual"])
            })
        
        # Add basic regulatory trends
        result["Regulatory Developments"].append({
            "development": "Data privacy regulation expansion",
            "impact": np.random.choice(impact_levels),
            "timeline": np.random.choice(["Current", "1-2 years", "3-5 years"])
        })
        
        # Add basic competitive landscape
        result["Competitive Landscape Evolution"].append({
            "shift": "Market consolidation through M&A",
            "impact": np.random.choice(impact_levels),
            "timeline": np.random.choice(["Ongoing", "Emerging", "Future"])
        })
        
        return result
    
    def _generate_mock_strategy(self, company: str, industry: str, competitors: List[str], objective: str) -> Dict[str, Any]:
        \"\"\"Generate simplified mock strategy data.\"\"\"
        # Map objective to strategy focus
        objective_mapping = {
            "market_share_growth": "increasing market share",
            "product_innovation": "accelerating product innovation",
            "cost_efficiency": "improving operational efficiency",
            "geographic_expansion": "expanding into new markets",
            "customer_retention": "improving customer retention",
            "digital_transformation": "implementing digital transformation"
        }
        
        strategy_focus = objective_mapping.get(objective, "improving competitive position")
        
        # Create strategic overview based on objective
        overview = f"A strategy for {company} to achieve growth through {strategy_focus} in the {industry} industry."
        
        # Generic advantages for all objectives
        advantages = [
            "Market expertise",
            "Customer relationships",
            "Operational capabilities",
            "Financial strength"
        ]
        
        # Generic initiatives for all objectives
        initiatives = [
            {"name": "Strategic initiative 1", "timeframe": "Short-term", "priority": "High"},
            {"name": "Strategic initiative 2", "timeframe": "Short-term", "priority": "Medium"},
            {"name": "Strategic initiative 3", "timeframe": "Medium-term", "priority": "High"},
            {"name": "Strategic initiative 4", "timeframe": "Long-term", "priority": "Medium"}
        ]
        
        # Generic metrics for all objectives
        metrics = [
            "Key performance indicator 1",
            "Key performance indicator 2",
            "Key performance indicator 3",
            "Key performance indicator 4"
        ]
        
        # Build the strategy response
        result = {
            "Strategic Overview": overview,
            "Competitive Advantages": advantages,
            "Key Initiatives": initiatives,
            "Resource Requirements": {
                "Financial": "Investment of $X million over Y years",
                "Human Resources": "Cross-functional team with expertise in key areas",
                "Technology": "System upgrades and new platform implementations",
                "Timeline": "Phased implementation over 18-24 months"
            },
            "Success Metrics": metrics,
            "Risk Assessment": [
                {"risk": "Competitive response", "severity": "High", "mitigation": "Monitor competitor actions"},
                {"risk": "Implementation delays", "severity": "Medium", "mitigation": "Establish project management"},
                {"risk": "Resource constraints", "severity": "Medium", "mitigation": "Prioritize initiatives"}
            ]
        }
        
        return result
"""

with open(ai_engine_path, "w") as f:
    f.write(ai_engine_content)
print(f"Created file: {ai_engine_path}")

# Create a basic NLP processor module if it doesn't exist
nlp_processor_path = marketsense_dir / "core" / "nlp_processor.py"
if not nlp_processor_path.exists():
    with open(nlp_processor_path, "w") as f:
        f.write('''"""
Natural Language Processing module for text analysis.
"""

class NLPProcessor:
    """NLP processor for text analysis and insight extraction."""
    
    def __init__(self):
        """Initialize the NLP Processor."""
        self.batch_size = 10
    
    def extract_key_insights(self, text, num_insights=5):
        """Extract key insights from text."""
        # Mock implementation
        sentences = text.split('. ')
        return sentences[:num_insights] if len(sentences) >= num_insights else sentences
    
    def analyze_sentiment(self, texts):
        """Analyze sentiment of text snippets."""
        # Mock implementation
        results = []
        for text in texts:
            results.append({
                "text": text[:100] + "..." if len(text) > 100 else text,
                "sentiment": "positive",
                "score": 0.8,
                "confidence": 0.9
            })
        return results
    
    def extract_entities(self, text):
        """Extract named entities from text."""
        # Mock implementation
        return {
            "companies": ["Example Corp", "Tech Inc"],
            "locations": ["New York", "San Francisco"],
            "technologies": ["AI", "Cloud"]
        }
    
    def summarize_text(self, text, max_sentences=3):
        """Generate a summary of text."""
        # Mock implementation
        sentences = text.split('. ')
        return '. '.join(sentences[:max_sentences]) + '.'
''')
    print(f"Created file: {nlp_processor_path}")

# Create a basic data analyzer module if it doesn't exist
data_analyzer_path = marketsense_dir / "core" / "data_analyzer.py"
if not data_analyzer_path.exists():
    with open(data_analyzer_path, "w") as f:
        f.write('''"""
Data analysis module for processing and visualizing market data.
"""

class DataAnalyzer:
    """Data analysis module for market data."""
    
    def __init__(self):
        """Initialize the Data Analyzer."""
        self.output_dir = "./data/outputs"
    
    def analyze_market_data(self, data, analysis_type="trend"):
        """Analyze market data."""
        # Mock implementation
        if analysis_type == "trend":
            return {
                "analysis_type": "trend",
                "metrics_analyzed": ["revenue", "profit"],
                "trends": {
                    "revenue": {
                        "direction": "upward",
                        "percentage_change": 15.3
                    }
                }
            }
        else:
            return {
                "analysis_type": analysis_type,
                "metrics_analyzed": ["revenue", "profit"]
            }
    
    def create_market_visualization(self, data, viz_type, title=""):
        """Create market visualization."""
        # Mock implementation
        return f"path/to/{viz_type}_{title.replace(' ', '_').lower()}.html"
''')
    print(f"Created file: {data_analyzer_path}")

# Create a minimal market_research.py module if it doesn't exist
market_research_path = marketsense_dir / "modules" / "market_research.py"
if not market_research_path.exists():
    with open(market_research_path, "w") as f:
        f.write('''"""
Market research module for company and competitor analysis.
"""

from marketsense.core.ai_engine import AIEngine
from marketsense.core.data_analyzer import DataAnalyzer

class MarketResearch:
    """Market research module for company and competitor analysis."""
    
    def __init__(self):
        """Initialize the Market Research module."""
        self.ai_engine = AIEngine()
        self.data_analyzer = DataAnalyzer()
    
    def analyze_company(self, company_name, industry, depth="comprehensive"):
        """Analyze a company."""
        # Get market analysis from AI Engine
        market_analysis = self.ai_engine.generate_market_analysis(company_name, industry, depth)
        return market_analysis
    
    def analyze_competitors(self, company, competitors, industry=""):
        """Analyze a company against its competitors."""
        # Get competitor analysis from AI Engine
        competitor_analysis = self.ai_engine.analyze_competitors(company, competitors, industry)
        return competitor_analysis
    
    def analyze_industry_trends(self, industry, timeframe="current"):
        """Analyze trends in a specific industry."""
        # Get trend analysis from AI Engine
        trends = self.ai_engine.identify_market_trends(industry, timeframe)
        return trends
''')
    print(f"Created file: {market_research_path}")

# Create a minimal sentiment_analyzer.py module if it doesn't exist
sentiment_analyzer_path = marketsense_dir / "modules" / "sentiment_analyzer.py"
if not sentiment_analyzer_path.exists():
    with open(sentiment_analyzer_path, "w") as f:
        f.write('''"""
Sentiment analysis module for market and competitor sentiment tracking.
"""

from marketsense.core.nlp_processor import NLPProcessor

class SentimentAnalyzer:
    """Advanced sentiment analysis module."""
    
    def __init__(self):
        """Initialize the Sentiment Analyzer."""
        self.nlp_processor = NLPProcessor()
    
    def analyze_market_sentiment(self, texts):
        """Analyze sentiment in market-related texts."""
        # Extract text content and prepare for batch processing
        text_contents = [item.get('text', '') for item in texts]
        
        # Perform sentiment analysis
        sentiment_results = self.nlp_processor.analyze_sentiment(text_contents)
        
        # Combine results with metadata
        for i, result in enumerate(sentiment_results):
            result.update({k: v for k, v in texts[i].items() if k != 'text'})
        
        # Return simplified results
        return {
            "total_texts": len(texts),
            "detailed_results": sentiment_results
        }
    
    def track_competitor_sentiment(self, competitor_texts):
        """Track sentiment around different competitors."""
        results = {}
        
        # Analyze sentiment for each competitor
        for competitor, texts in competitor_texts.items():
            if texts:
                competitor_sentiment = self.analyze_market_sentiment(texts)
                results[competitor] = competitor_sentiment
        
        return {
            "competitor_results": results
        }
    
    def identify_sentiment_drivers(self, texts, sentiment_type="all"):
        """Identify key phrases and topics driving sentiment."""
        # Extract key phrases using NLP processor
        all_text = " ".join([t.get('text', '') for t in texts])
        key_phrases = self.nlp_processor.extract_key_insights(all_text, num_insights=5)
        
        return {
            "sentiment_type": sentiment_type,
            "text_count": len(texts),
            "key_phrases": key_phrases
        }
''')
    print(f"Created file: {sentiment_analyzer_path}")

# Create a minimal trend_forecaster.py module if it doesn't exist
trend_forecaster_path = marketsense_dir / "modules" / "trend_forecaster.py"
if not trend_forecaster_path.exists():
    with open(trend_forecaster_path, "w") as f:
        f.write('''"""
Market trend forecasting module.
"""

from marketsense.core.data_analyzer import DataAnalyzer

class TrendForecaster:
    """Advanced market trend forecasting module."""
    
    def __init__(self):
        """Initialize the Trend Forecaster."""
        self.data_analyzer = DataAnalyzer()
    
    def forecast_market_trends(self, historical_data, periods=4, confidence_interval=0.95):
        """Forecast future market trends based on historical data."""
        # Mock implementation
        return {
            "forecast_date": "2023-05-01",
            "metrics_forecasted": ["revenue", "profit"],
            "forecast_periods": periods,
            "confidence_interval": confidence_interval,
            "forecasts": {
                "revenue": {
                    "model_type": "linear_regression",
                    "metrics": {
                        "rmse": 0.15,
                        "mae": 0.12
                    },
                    "forecast_data": [
                        {"period": "2023-06-01", "forecast": 120.5, "lower_bound": 110.2, "upper_bound": 130.8}
                    ]
                }
            }
        }
    
    def analyze_seasonality(self, time_series_data):
        """Analyze seasonality patterns in time series data."""
        # Mock implementation
        return {
            "metrics_analyzed": ["revenue"],
            "seasonality_results": {
                "revenue": {
                    "monthly_seasonality": {
                        "strength": "Moderate",
                        "peak_month": 12,
                        "trough_month": 7
                    },
                    "quarterly_seasonality": {
                        "strength": "Strong",
                        "peak_quarter": 4,
                        "trough_quarter": 3
                    }
                }
            }
        }
''')
    print(f"Created file: {trend_forecaster_path}")

# Update the setup.py file
setup_path = current_dir / "setup.py"
setup_content = '''
from setuptools import setup, find_packages

setup(
    name="marketsense",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "streamlit",
        "flask",
        "pandas",
        "numpy",
        "openai",
        "python-dotenv",
        "plotly",
        "scikit-learn",
        "matplotlib",
        "requests",
        "scipy",
    ],
    python_requires=">=3.7",
)
'''

with open(setup_path, "w") as f:
    f.write(setup_content)
print(f"Updated file: {setup_path}")

# Fix the app.py file
app_path = current_dir / "app.py"
app_content = '''"""
Flask API for MarketSense AI.
"""

import os
import sys
from pathlib import Path
import logging

# Add the current directory to Python path to ensure marketsense can be found
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from flask import Flask, jsonify, request
    from marketsense.core.ai_engine import AIEngine
    from marketsense.core.nlp_processor import NLPProcessor
    from marketsense.core.data_analyzer import DataAnalyzer
    from marketsense.modules.market_research import MarketResearch
    from marketsense.modules.sentiment_analyzer import SentimentAnalyzer
    from marketsense.modules.trend_forecaster import TrendForecaster
    from marketsense import config
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Please make sure the marketsense package is installed correctly.")
    logger.error("Try running: pip install -e .")
    sys.exit(1)

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(config)

# Initialize services lazily (only when needed)
_ai_engine = None
_nlp_processor = None
_data_analyzer = None
_market_research = None
_sentiment_analyzer = None
_trend_forecaster = None

# Lazy loading functions to reduce memory usage
def ai_engine():
    global _ai_engine
    if _ai_engine is None:
        _ai_engine = AIEngine()
    return _ai_engine

def nlp_processor():
    global _nlp_processor
    if _nlp_processor is None:
        _nlp_processor = NLPProcessor()
    return _nlp_processor

def data_analyzer():
    global _data_analyzer
    if _data_analyzer is None:
        _data_analyzer = DataAnalyzer()
    return _data_analyzer

def market_research():
    global _market_research
    if _market_research is None:
        _market_research = MarketResearch()
    return _market_research

def sentiment_analyzer():
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        _sentiment_analyzer = SentimentAnalyzer()
    return _sentiment_analyzer

def trend_forecaster():
    global _trend_forecaster
    if _trend_forecaster is None:
        _trend_forecaster = TrendForecaster()
    return _trend_forecaster

@app.route('/api/health', methods=['GET'])
def health_check():
    """API health check endpoint."""
    return jsonify({"status": "healthy", "version": "1.0.0"})

@app.route('/api/company-analysis', methods=['POST'])
def company_analysis_api():
    """API endpoint for company analysis."""
    data = request.json
    company_name = data.get('company_name', '')
    industry = data.get('industry', '')
    depth = data.get('depth', 'comprehensive')
    
    if not company_name:
        return jsonify({"success": False, "error": "Company name is required"}), 400
    
    try:
        result = market_research().analyze_company(company_name, industry, depth)
        return jsonify({"success": True, "data": result})
    except Exception as e:
        logger.error(f"Error in company analysis: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/competitor-analysis', methods=['POST'])
def competitor_analysis_api():
    """API endpoint for competitor analysis."""
    data = request.json
    company = data.get('company', '')
    competitors = data.get('competitors', [])
    industry = data.get('industry', '')
    
    if not company or not competitors:
        return jsonify({"success": False, "error": "Company and competitors are required"}), 400
    
    try:
        result = market_research().analyze_competitors(company, competitors, industry)
        return jsonify({"success": True, "data": result})
    except Exception as e:
        logger.error(f"Error in competitor analysis: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/market-trends', methods=['POST'])
def market_trends_api():
    """API endpoint for market trends analysis."""
    data = request.json
    industry = data.get('industry', '')
    timeframe = data.get('timeframe', 'current')
    
    if not industry:
        return jsonify({"success": False, "error": "Industry is required"}), 400
    
    try:
        result = market_research().analyze_industry_trends(industry, timeframe)
        return jsonify({"success": True, "data": result})
    except Exception as e:
        logger.error(f"Error in market trends analysis: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/sentiment-analysis', methods=['POST'])
def sentiment_analysis_api():
    """API endpoint for sentiment analysis."""
    data = request.json
    texts = data.get('texts', [])
    
    if not texts:
        return jsonify({"success": False, "error": "Text content is required"}), 400
    
    try:
        result = sentiment_analyzer().analyze_market_sentiment(texts)
        return jsonify({"success": True, "data": result})
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/strategic-forecast', methods=['POST'])
def strategic_forecast_api():
    """API endpoint for strategic forecasting."""
    data = request.json
    company = data.get('company', '')
    industry = data.get('industry', '')
    competitors = data.get('competitors', [])
    objective = data.get('objective', 'market_share_growth')
    
    if not company or not industry:
        return jsonify({"success": False, "error": "Company and industry are required"}), 400
    
    try:
        result = ai_engine().generate_competitive_strategy(company, industry, competitors, objective)
        return jsonify({"success": True, "data": result})
    except Exception as e:
        logger.error(f"Error in strategic forecasting: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    # Print where marketsense is imported from
    try:
        import marketsense
        print(f"Using marketsense from: {marketsense.__file__}")
    except ImportError:
        print("WARNING: marketsense module not found in Python path!")
    
    print(f"Starting Flask server on port {5000}...")
    app.run(debug=True, host='0.0.0.0', port=5000)
'''

with open(app_path, "w") as f:
    f.write(app_content)
print(f"Updated file: {app_path}")

# Fix the streamlit_app.py file
streamlit_app_path = current_dir / "streamlit_app.py"
streamlit_app_content = '''"""
Streamlit web application for MarketSense AI.
"""

import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Add the current directory to Python path to ensure marketsense can be found
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

try:
    from marketsense.core.ai_engine import AIEngine
    from marketsense.core.nlp_processor import NLPProcessor
    from marketsense.core.data_analyzer import DataAnalyzer
    from marketsense.modules.market_research import MarketResearch
    from marketsense.modules.sentiment_analyzer import SentimentAnalyzer
    from marketsense.modules.trend_forecaster import TrendForecaster
    from marketsense import config
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please make sure the marketsense package is installed correctly.")
    st.error("Try running: pip install -e .")
    st.stop()

# Initialize services lazily (only when needed)
@st.cache_resource
def load_ai_engine():
    return AIEngine()

@st.cache_resource
def load_nlp_processor():
    return NLPProcessor()

@st.cache_resource
def load_data_analyzer():
    return DataAnalyzer()

@st.cache_resource
def load_market_research():
    return MarketResearch()

@st.cache_resource
def load_sentiment_analyzer():
    return SentimentAnalyzer()

@st.cache_resource
def load_trend_forecaster():
    return TrendForecaster()

# Set page configuration
st.set_page_config(
    page_title="MarketSense AI - Market Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Print where marketsense is imported from
try:
    import marketsense
    st.sidebar.info(f"Using marketsense from: {marketsense.__file__}")
except ImportError:
    st.sidebar.warning("WARNING: marketsense module not found in Python path!")

# Sidebar
st.sidebar.title("MarketSense AI")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "Navigate", 
    ["Dashboard", "Company Analysis", "Competitor Analysis", "Market Trends", "Sentiment Analysis", "Trend Forecasting"]
)

# Cache for storing analysis results
if "analysis_cache" not in st.session_state:
    st.session_state.analysis_cache = {}

if "recent_analyses" not in st.session_state:
    st.session_state.recent_analyses = {}

if page == "Dashboard":
    st.title("📊 Market Intelligence Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Welcome to MarketSense AI")
        st.write("""
        MarketSense AI provides intelligent market analysis and competitive intelligence 
        to help businesses make data-driven decisions. Use the sidebar to navigate 
        through different analysis tools.
        """)

elif page == "Company Analysis":
    st.title("🏢 Company Analysis")
    
    # Input form
    with st.form("company_analysis_form"):
        company_name = st.text_input("Company Name", placeholder="e.g. Tesla")
        industry = st.selectbox(
            "Industry", 
            ["Technology", "Automotive", "Healthcare", "Finance", "Retail", 
             "Energy", "Manufacturing", "Consumer Goods", "Telecommunications", "Other"]
        )
        analysis_depth = st.select_slider(
            "Analysis Depth",
            options=["basic", "comprehensive", "expert"],
            value="comprehensive"
        )
        
        submit_button = st.form_submit_button("Analyze Company")
    
    if submit_button and company_name:
        with st.spinner(f"Analyzing {company_name}..."):
            # Perform analysis
            research = load_market_research()
            analysis = research.analyze_company(company_name, industry, analysis_depth)
            
            # Store in session state
            cache_key = f"{company_name}_{industry}_{analysis_depth}"
            st.session_state.analysis_cache[cache_key] = analysis
            
            # Update recent analyses
            st.session_state.recent_analyses[f"{company_name} ({analysis_depth})"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            
            # Display analysis results
            st.json(analysis)

elif page == "Competitor Analysis":
    st.title("⚔️ Competitor Analysis")
    
    # Placeholder implementation
    st.write("Competitor analysis functionality goes here")

elif page == "Market Trends":
    st.title("📈 Market Trends Analysis")
    
    # Placeholder implementation
    st.write("Market trends analysis functionality goes here")

elif page == "Sentiment Analysis":
    st.title("💬 Market Sentiment Analysis")
    
    # Placeholder implementation
    st.write("Sentiment analysis functionality goes here")

elif page == "Trend Forecasting":
    st.title("🔮 Market Trend Forecasting")
    
    # Placeholder implementation
    st.write("Trend forecasting functionality goes here")
'''

with open(streamlit_app_path, "w") as f:
    f.write(streamlit_app_content)
print(f"Updated file: {streamlit_app_path}")

# Create a verification script
verify_script_path = current_dir / "verify_install.py"
verify_script_content = '''"""
Verify that the marketsense package is correctly installed and importable.
"""

import sys
import os
from pathlib import Path

print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")
print("\nPython path:")
for path in sys.path:
    print(f"  - {path}")

print("\nTrying to import marketsense...")
try:
    import marketsense
    print(f"✓ Successfully imported marketsense from {marketsense.__file__}")
    
    # Try importing key modules
    modules_to_check = [
        "marketsense.core.ai_engine",
        "marketsense.core.nlp_processor",
        "marketsense.core.data_analyzer",
        "marketsense.modules.market_research",
        "marketsense.modules.sentiment_analyzer",
        "marketsense.modules.trend_forecaster",
        "marketsense.utils.cache"
    ]
    
    for module_name in modules_to_check:
        try:
            module = __import__(module_name, fromlist=[""])
            print(f"✓ Successfully imported {module_name}")
        except ImportError as e:
            print(f"✗ Failed to import {module_name}: {e}")
    
except ImportError as e:
    print(f"✗ Failed to import marketsense: {e}")
    
    # Look for marketsense in the current directory structure
    print("\nLooking for marketsense directory in the current path...")
    current_dir = Path(os.getcwd())
    marketsense_dir = current_dir / "marketsense"
    
    if marketsense_dir.exists():
        print(f"Found marketsense directory at: {marketsense_dir}")
        
        # Check for __init__.py file
        init_file = marketsense_dir / "__init__.py"
        if init_file.exists():
            print(f"  ✓ Found __init__.py file")
        else:
            print(f"  ✗ Missing __init__.py file")
        
        # Check for core directory
        core_dir = marketsense_dir / "core"
        if core_dir.exists():
            print(f"  ✓ Found core directory")
            
            # Check for core/__init__.py file
            core_init = core_dir / "__init__.py"
            if core_init.exists():
                print(f"    ✓ Found core/__init__.py file")
            else:
                print(f"    ✗ Missing core/__init__.py file")
            
            # Check for ai_engine.py file
            ai_engine = core_dir / "ai_engine.py"
            if ai_engine.exists():
                print(f"    ✓ Found core/ai_engine.py file")
            else:
                print(f"    ✗ Missing core/ai_engine.py file")
        else:
            print(f"  ✗ Missing core directory")
    else:
        print(f"Could not find marketsense directory at: {marketsense_dir}")

print("\nNext steps:")
print("1. Run 'pip install -e .'")
print("2. Run 'python app.py' or 'python streamlit_app.py'")
'''

with open(verify_script_path, "w") as f:
    f.write(verify_script_content)
print(f"Created file: {verify_script_path}")

print("\nDirectory structure and files have been created successfully!")
print("Next steps:")
print("1. Run 'pip install -e .'")
print("2. Run 'python verify_install.py' to verify the installation")
print("3. Run 'python app.py' or 'python streamlit_app.py'")