"""
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
