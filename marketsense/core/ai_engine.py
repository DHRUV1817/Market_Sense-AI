# AI-powered analysis engine
"""
AI Engine for market analysis using Groq API.
"""

import os
import json
import groq
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import time

from marketsense import config
from marketsense.utils.cache import cached, get_cache_path, is_cache_valid, save_to_cache, load_from_cache

class AIEngine:
    """Advanced AI Engine to interact with Groq API for market analysis."""
    
    def __init__(self):
        """Initialize the AI Engine with Groq API key."""
        self.client = groq.Groq(api_key=config.GROQ_API_KEY)
        self.model = config.GROQ_MODEL
    
    @cached(lambda self, company_name, industry, depth: f"{company_name}_{industry}_{depth}_analysis")
    def generate_market_analysis(self, company_name: str, industry: str, 
                               depth: str = "comprehensive") -> Dict[str, Any]:
        """
        Generate market analysis for a specific company in an industry.
        
        Args:
            company_name: Name of the company to analyze
            industry: Industry sector of the company
            depth: Analysis depth ("basic", "comprehensive", "expert")
            
        Returns:
            Dictionary containing market analysis data
        """
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
        sections_text = "\n".join([f"{i+1}. {section}" for i, section in enumerate(sections)])
        
        # Create prompt for market analysis
        prompt = f"""
        As a market intelligence expert, provide a {depth} analysis for {company_name} in the {industry} industry.
        Include the following sections:
        {sections_text}
        
        For the SWOT analysis, be specific and detailed with actionable insights.
        Format the response as structured JSON with these sections as keys.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a market analysis expert providing detailed insights."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1500,
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
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
        """
        Generate advanced comparative analysis between a company and its competitors.
        
        Args:
            company: Main company name
            competitors: List of competitor company names
            industry: Optional industry context
            
        Returns:
            Dictionary containing competitive analysis data
        """
        # Create prompt for competitor analysis with industry context
        industry_context = f" in the {industry} industry" if industry else ""
        competitors_list = ", ".join(competitors)
        
        prompt = f"""
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
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a competitive intelligence expert providing detailed insights."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error generating competitor analysis: {str(e)}")
            result = {"error": "Could not generate competitor analysis."}

        return result
    
    @cached(lambda self, industry, timeframe: f"{industry}_{timeframe}_trends")
    def identify_market_trends(self, industry: str, timeframe: str = "current") -> Dict[str, Any]:
        """
        Identify key market trends in a specific industry.
        
        Args:
            industry: Industry to analyze
            timeframe: Trends timeframe ("current", "emerging", "future")
            
        Returns:
            Dictionary containing trend analysis data
        """
        # Create prompt based on timeframe
        if timeframe == "current":
            scope = "currently impacting"
        elif timeframe == "emerging":
            scope = "emerging in the next 1-2 years"
        else:  # future
            scope = "likely to shape the industry in the next 3-5 years"
        
        prompt = f"""
        As a market trends analyst, identify the key trends {scope} the {industry} industry.
        
        Include:
        1. Major market trends with impact assessment (high/medium/low)
        2. Technology trends changing the industry
        3. Consumer behavior shifts
        4. Regulatory developments
        5. Competitive landscape evolution
        
        For each trend, provide specific examples, impact assessment, and strategic implications.
        Format the response as structured JSON with these categories as keys.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a market trends analyst providing detailed insights."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1500,
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error identifying market trends: {str(e)}")
            result = {"error": "Could not identify market trends."}

        return result
    
    @cached(lambda self, company, industry, competitors, objective: f"{company}_{industry}_{objective}_{'_'.join(competitors)}_strategy")
    def generate_competitive_strategy(self, company: str, industry: str, 
                                   competitors: List[str], objective: str) -> Dict[str, Any]:
        """
        Generate a competitive strategy based on market analysis.
        
        Args:
            company: Company name
            industry: Industry sector
            competitors: List of main competitors
            objective: Strategic objective (e.g., "market_share_growth", "product_innovation")
            
        Returns:
            Dictionary containing strategic recommendations
        """
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
        
        prompt = f"""
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
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a strategic consultant providing detailed insights."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error generating competitive strategy: {str(e)}")
            result = {"error": "Could not generate competitive strategy."}
        
        return result
