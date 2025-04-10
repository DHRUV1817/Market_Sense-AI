# AI-powered analysis engine
"""
AI Engine for market analysis using OpenAI API.
"""

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
    """Advanced AI Engine to interact with OpenAI API for market analysis."""
    
    def __init__(self):
        """Initialize the AI Engine with OpenAI API key."""
        openai.api_key = config.OPENAI_API_KEY
        self.model = config.OPENAI_MODEL
    
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
        
        # For demo purposes, create mock data (in production, call OpenAI API)
        # This would be replaced with actual API call in production version
        result = self._generate_mock_competitor_analysis(company, competitors, industry)
        
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
        
        # For demo purposes, create mock data (in production, call OpenAI API)
        result = self._generate_mock_trends(industry, timeframe)
        
        return result
    
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
        
        # For demo purposes, create mock data (in production, call OpenAI API)
        result = self._generate_mock_strategy(company, industry, competitors, objective)
        
        # Cache the result
        save_to_cache(result, cache_path)
        
        return result
    
    # Private methods for generating mock data (would be replaced with API calls in production)
    
    def _generate_mock_analysis(self, company: str, industry: str, depth: str, sections: List[str]) -> Dict[str, Any]:
        """Generate mock analysis data for demonstration purposes."""
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
        """Generate simplified mock competitor analysis data."""
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
        """Generate simplified mock trends data."""
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
        """Generate simplified mock strategy data."""
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