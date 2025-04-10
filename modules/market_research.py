# Company and competitor analysis
"""
Market research module for company and competitor analysis.
"""

import pandas as pd
from typing import Dict, List, Any, Optional
import plotly.graph_objects as go

from marketsense import config
from marketsense.core.ai_engine import AIEngine
from marketsense.core.data_analyzer import DataAnalyzer
from marketsense.utils.visualization import save_visualization, create_swot_visualization

class MarketResearch:
    """Market research module for company and competitor analysis."""
    
    def __init__(self):
        """Initialize the Market Research module."""
        self.ai_engine = AIEngine()
        self.data_analyzer = DataAnalyzer()
    
    def analyze_company(self, company_name: str, industry: str, depth: str = "comprehensive") -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a company.
        
        Args:
            company_name: Name of the company to analyze
            industry: Industry sector of the company
            depth: Analysis depth ("basic", "comprehensive", "expert")
            
        Returns:
            Dictionary containing analysis results
        """
        # Get market analysis from AI Engine
        market_analysis = self.ai_engine.generate_market_analysis(company_name, industry, depth)
        
        # Create SWOT visualization
        swot_data = market_analysis.get("SWOT Analysis", {})
        if swot_data:
            fig = create_swot_visualization(company_name, swot_data)
            swot_viz_path = save_visualization(fig, f"{company_name}_swot")
            market_analysis["swot_visualization"] = swot_viz_path
        
        # Enhance with additional data
        enhanced_data = self._enhance_with_market_data(company_name, industry, market_analysis)
        
        return enhanced_data
    
    def analyze_competitors(self, company: str, competitors: List[str], industry: str = "") -> Dict[str, Any]:
        """
        Analyze a company against its competitors.
        
        Args:
            company: Main company name
            competitors: List of competitor company names
            industry: Optional industry context
            
        Returns:
            Dictionary containing competitive analysis results
        """
        # Get competitor analysis from AI Engine
        competitor_analysis = self.ai_engine.analyze_competitors(company, competitors, industry)
        
        # Create visualizations
        viz_paths = self.create_competitor_visualizations(company, competitors, competitor_analysis)
        competitor_analysis["visualizations"] = viz_paths
        
        return competitor_analysis
    
    def _enhance_with_market_data(self, company_name: str, industry: str, 
                               base_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance basic analysis with additional market data.
        
        Args:
            company_name: Company name
            industry: Industry sector
            base_analysis: Base analysis to enhance
            
        Returns:
            Enhanced analysis with additional data
        """
        # This would normally fetch real market data
        # For now, we'll use placeholder data
        enhanced = base_analysis.copy()
        
        # Add market size data
        enhanced["Market Data"] = {
            "Industry Size": f"$XXX Billion",
            "Growth Rate": "X.X% YoY",
            "Company Market Share": "XX%",
            "Market Trends": [
                "Increasing digitalization",
                "Growing demand for sustainable solutions",
                "Regulatory changes affecting market dynamics"
            ]
        }
        
        return enhanced
    
    def create_competitor_visualizations(self, company: str, competitors: List[str], 
                                      analysis: Dict[str, Any]) -> List[str]:
        """
        Create visualizations for competitor analysis.
        
        Args:
            company: Main company name
            competitors: List of competitor names
            analysis: Competitor analysis data
            
        Returns:
            List of paths to saved visualizations
        """
        output_paths = []
        all_companies = [company] + competitors
        
        # Create market share bar chart
        market_share_data = analysis.get("Comparative Market Share", {})
        if market_share_data:
            # Extract market shares
            market_shares = []
            for comp in all_companies:
                if comp in market_share_data:
                    # Extract numeric value from percentage string (e.g., "25%" -> 25)
                    share_str = market_share_data[comp]
                    try:
                        share = float(share_str.strip('%'))
                    except:
                        # Generate value if conversion fails
                        share = 20 if comp == company else 15
                else:
                    share = 20 if comp == company else 15
                
                market_shares.append(share)
            
            # Create the bar chart
            fig = go.Figure(data=[
                go.Bar(name='Market Share (%)', 
                      x=all_companies, 
                      y=market_shares,
                      textposition='auto',
                      texttemplate='%{y:.1f}%')
            ])
            
            # Update layout
            fig.update_layout(
                title=f"Market Share Comparison: {company} vs Competitors",
                xaxis_title="Companies",
                yaxis_title="Market Share (%)",
                yaxis=dict(range=[0, max(market_shares) * 1.2])
            )
            
            # Save figure
            output_path = save_visualization(fig, f"{company}_vs_competitors")
            output_paths.append(output_path)
        
        # Create product comparison radar chart if data is available
        product_comparison = analysis.get("Product/Service Comparison", {})
        if product_comparison and all(isinstance(product_comparison[aspect], dict) for aspect in product_comparison):
            try:
                # Convert ratings to numeric values for visualization
                rating_map = {
                    "Excellent": 5,
                    "Very Good": 4,
                    "Good": 3,
                    "Average": 2,
                    "Below Average": 1,
                    "Poor": 0
                }
                
                # Prepare data for radar chart
                categories = list(product_comparison.keys())
                
                fig = go.Figure()
                
                for comp in all_companies:
                    comp_ratings = []
                    for category in categories:
                        rating_str = product_comparison[category].get(comp, "Average")
                        rating = rating_map.get(rating_str, 2)  # Default to 2 (Average) if not found
                        comp_ratings.append(rating)
                    
                    # Add the first rating at the end to close the loop
                    comp_ratings.append(comp_ratings[0])
                    radar_categories = categories + [categories[0]]
                    
                    fig.add_trace(go.Scatterpolar(
                        r=comp_ratings,
                        theta=radar_categories,
                        fill='toself',
                        name=comp
                    ))
                
                fig.update_layout(
                    title=f"Product/Service Comparison: {company} vs Competitors",
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 5]
                        )
                    ),
                    showlegend=True
                )
                
                radar_path = save_visualization(fig, f"{company}_product_comparison")
                output_paths.append(radar_path)
            except Exception as e:
                print(f"Error creating product comparison visualization: {str(e)}")
        
        return output_paths
    
    def analyze_industry_trends(self, industry: str, timeframe: str = "current") -> Dict[str, Any]:
        """
        Analyze trends in a specific industry.
        
        Args:
            industry: Industry to analyze
            timeframe: Timeframe for trend analysis ("current", "emerging", "future")
            
        Returns:
            Dictionary containing trend analysis
        """
        # Get trend analysis from AI Engine
        trends = self.ai_engine.identify_market_trends(industry, timeframe)
        
        # Create visualizations
        viz_paths = self.create_trend_visualizations(industry, trends)
        trends["visualizations"] = viz_paths
        
        return trends
    
    def create_trend_visualizations(self, industry: str, trends: Dict[str, Any]) -> List[str]:
        """
        Create visualizations for industry trends.
        
        Args:
            industry: Industry being analyzed
            trends: Trend analysis data
            
        Returns:
            List of paths to saved visualizations
        """
        output_paths = []
        
        # Create impact visualization for major market trends
        major_trends = trends.get("Major Market Trends", [])
        if major_trends:
            trend_names = [t.get('trend', 'Unknown')[:30] + '...' if len(t.get('trend', 'Unknown')) > 30 
                          else t.get('trend', 'Unknown') for t in major_trends]
            
            # Map impact levels to numeric values
            impact_map = {"High": 3, "Medium": 2, "Low": 1}
            impact_values = [impact_map.get(t.get('impact', 'Medium'), 2) for t in major_trends]
            
            # Create horizontal bar chart
            fig = go.Figure(data=[
                go.Bar(
                    y=trend_names,
                    x=impact_values,
                    orientation='h',
                    marker=dict(
                        color=impact_values,
                        colorscale='Blues',
                        colorbar=dict(
                            title="Impact",
                            tickvals=[1, 2, 3],
                            ticktext=["Low", "Medium", "High"]
                        )
                    )
                )
            ])
            
            fig.update_layout(
                title=f"{industry} Industry Major Trends - Impact Assessment",
                xaxis=dict(
                    tickvals=[1, 2, 3],
                    ticktext=["Low", "Medium", "High"],
                    title="Impact Level"
                ),
                yaxis=dict(title="Trend"),
                height=500 + len(major_trends) * 30  # Dynamic height based on number of trends
            )
            
            output_path = save_visualization(fig, f"{industry}_trend_impact")
            output_paths.append(output_path)
        
        # Create technology adoption visualization
        tech_trends = trends.get("Technology Trends", [])
        if tech_trends:
            tech_names = [t.get('trend', 'Unknown') for t in tech_trends]
            
            # Map adoption stages to numeric values
            adoption_map = {"Early": 1, "Growing": 2, "Mainstream": 3}
            adoption_values = [adoption_map.get(t.get('adoption_stage', 'Growing'), 2) for t in tech_trends]
            
            # Map impact to sizes
            impact_map = {"High": 25, "Medium": 15, "Low": 8}
            sizes = [impact_map.get(t.get('impact', 'Medium'), 15) for t in tech_trends]
            
            # Create bubble chart
            fig = go.Figure(data=[
                go.Scatter(
                    x=adoption_values,
                    y=range(len(tech_names)),
                    text=tech_names,
                    mode='markers',
                    marker=dict(
                        size=sizes,
                        sizemode='area',
                        sizeref=0.1,
                        color=adoption_values,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(
                            title="Adoption Stage",
                            tickvals=[1, 2, 3],
                            ticktext=["Early", "Growing", "Mainstream"]
                        )
                    )
                )
            ])
            
            fig.update_layout(
                title=f"{industry} Industry Technology Adoption",
                xaxis=dict(
                    tickvals=[1, 2, 3],
                    ticktext=["Early", "Growing", "Mainstream"],
                    title="Adoption Stage"
                ),
                yaxis=dict(
                    tickvals=list(range(len(tech_names))),
                    ticktext=tech_names,
                    title="Technology"
                ),
                height=400
            )
            
            output_path = save_visualization(fig, f"{industry}_tech_adoption")
            output_paths.append(output_path)
        
        return output_paths