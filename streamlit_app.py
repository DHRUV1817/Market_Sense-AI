"""
Streamlit web application for MarketSense AI.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import json
from datetime import datetime
from pathlib import Path

from marketsense.core.ai_engine import AIEngine
from marketsense.core.nlp_processor import NLPProcessor
from marketsense.core.data_analyzer import DataAnalyzer
from marketsense.modules.market_research import MarketResearch
from marketsense.modules.sentiment_analyzer import SentimentAnalyzer
from marketsense.modules.trend_forecaster import TrendForecaster
from marketsense import config

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

def render_dashboard():
    """Render the main dashboard page."""
    st.title("📊 Market Intelligence Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Welcome to MarketSense AI")
        st.write("""
        MarketSense AI provides intelligent market analysis and competitive intelligence 
        to help businesses make data-driven decisions. Use the sidebar to navigate 
        through different analysis tools.
        """)
        
        st.info("""
        **Getting Started**:
        1. Navigate to 'Company Analysis' for a deep-dive on a specific company
        2. Use 'Competitor Analysis' to compare companies
        3. Explore 'Market Trends' to identify industry patterns
        4. Analyze sentiment in 'Sentiment Analysis'
        5. Forecast future trends in 'Trend Forecasting'
        """)
    
    with col2:
        st.subheader("Recent Analysis")
        if not st.session_state.recent_analyses:
            st.write("No recent analyses. Start by analyzing a company or competitors.")
        else:
            for item, timestamp in st.session_state.recent_analyses.items():
                st.write(f"- {item} (analyzed at {timestamp})")
    
    # Sample charts for the dashboard
    st.subheader("Industry Overview")
    
    # Sample data
    industries = ["Technology", "Healthcare", "Finance", "Retail", "Manufacturing"]
    growth_rates = [7.2, 5.1, 3.8, 2.9, 4.3]
    market_sizes = [2.5, 1.8, 3.2, 1.5, 2.1]
    
    # Create columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Industry growth rates
        fig1 = px.bar(
            x=industries,
            y=growth_rates,
            labels={"x": "Industry", "y": "Growth Rate (%)"},
            title="Industry Growth Rates",
            color=growth_rates,
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Market sizes
        fig2 = px.pie(
            names=industries,
            values=market_sizes,
            title="Market Size Distribution (Trillion USD)",
            hole=0.4
        )
        st.plotly_chart(fig2, use_container_width=True)

def render_company_analysis():
    """Render the company analysis page."""
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
            # Check cache first
            cache_key = f"{company_name}_{industry}_{analysis_depth}"
            if cache_key in st.session_state.analysis_cache:
                analysis = st.session_state.analysis_cache[cache_key]
                st.success("Retrieved analysis from cache")
            else:
                # Perform new analysis
                analysis = load_market_research().analyze_company(company_name, industry, analysis_depth)
                st.session_state.analysis_cache[cache_key] = analysis
                
                # Update recent analyses
                st.session_state.recent_analyses[f"{company_name} ({analysis_depth})"] = datetime.now().strftime("%Y-%m-%d %H:%M")
                st.success(f"Analysis of {company_name} completed!")

            # Display analysis results
            display_company_analysis(company_name, analysis, analysis_depth)

def display_company_analysis(company_name, analysis, analysis_depth):
    """Display company analysis results."""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Company Overview")
        st.write(analysis.get("Company Overview", "No overview available"))
        
        st.subheader("Market Position")
        st.write(analysis.get("Market Position", "No market position data available"))
        
        st.subheader("Growth Opportunities")
        opportunities = analysis.get("Growth Opportunities", [])
        if isinstance(opportunities, list):
            for opportunity in opportunities:
                if isinstance(opportunity, dict):
                    st.write(f"- **{opportunity.get('opportunity', '')}** (Impact: {opportunity.get('potential_impact', 'Unknown')})")
                else:
                    st.write(f"- {opportunity}")
        else:
            st.write(opportunities)
        
        st.subheader("Risks and Challenges")
        risks = analysis.get("Risks and Challenges", [])
        if isinstance(risks, list):
            for risk in risks:
                if isinstance(risk, dict):
                    st.write(f"- **{risk.get('risk', '')}** (Severity: {risk.get('severity', 'Unknown')}, Probability: {risk.get('probability', 'Unknown')})")
                else:
                    st.write(f"- {risk}")
        else:
            st.write(risks)
        
        # Expert-level sections
        if analysis_depth == "expert":
            display_expert_analysis(analysis)
    
    with col2:
        # SWOT Analysis
        st.subheader("SWOT Analysis")
        swot = analysis.get("SWOT Analysis", {})
        
        # Create tabs for SWOT components
        swot_tabs = st.tabs(["Strengths", "Weaknesses", "Opportunities", "Threats"])
        
        with swot_tabs[0]:
            for item in swot.get("Strengths", ["No data"]):
                st.write(f"- {item}")
        
        with swot_tabs[1]:
            for item in swot.get("Weaknesses", ["No data"]):
                st.write(f"- {item}")
        
        with swot_tabs[2]:
            for item in swot.get("Opportunities", ["No data"]):
                st.write(f"- {item}")
        
        with swot_tabs[3]:
            for item in swot.get("Threats", ["No data"]):
                st.write(f"- {item}")
        
        # Key Competitors
        st.subheader("Key Competitors")
        competitors = analysis.get("Key Competitors", [])
        if isinstance(competitors, list):
            if competitors and isinstance(competitors[0], dict):
                for comp in competitors:
                    st.write(f"- **{comp.get('name', '')}** (Market Share: {comp.get('market_share', 'Unknown')})")
                    st.write(f"  *Primary Strength:* {comp.get('primary_strength', '')}")
            else:
                for competitor in competitors:
                    st.write(f"- {competitor}")
        else:
            st.write(competitors)
        
        # Show SWOT visualization if available
        if "swot_visualization" in analysis:
            viz_path = analysis["swot_visualization"]
            try:
                with open(viz_path, 'r') as f:
                    html_content = f.read()
                    st.components.v1.html(html_content, height=600)
            except Exception as e:
                st.warning(f"Could not load visualization: {str(e)}")

def display_expert_analysis(analysis):
    """Display expert-level analysis sections."""
    if "Financial Analysis" in analysis:
        st.subheader("Financial Analysis")
        fin_analysis = analysis["Financial Analysis"]
        
        metrics = {k: v for k, v in fin_analysis.items() if k != "Assessment"}
        st.json(metrics)
        
        if "Assessment" in fin_analysis:
            st.write("**Assessment:**")
            st.write(fin_analysis["Assessment"])
    
    if "Strategic Recommendations" in analysis:
        st.subheader("Strategic Recommendations")
        recommendations = analysis["Strategic Recommendations"]
        for rec in recommendations:
            if isinstance(rec, dict):
                st.write(f"- **{rec.get('recommendation', '')}** (Priority: {rec.get('priority', 'Unknown')})")
                st.write(f"  *Expected Impact:* {rec.get('expected_impact', '')}")
            else:
                st.write(f"- {rec}")
    
    if "Technology Assessment" in analysis:
        st.subheader("Technology Assessment")
        tech_assessment = analysis["Technology Assessment"]
        
        st.write("**Current State:**")
        st.write(tech_assessment.get("Current State", "N/A"))
        
        st.write("**Key Technologies:**")
        for tech in tech_assessment.get("Key Technologies", []):
            st.write(f"- {tech}")
        
        st.write("**Gap Analysis:**")
        st.write(tech_assessment.get("Gap Analysis", "N/A"))
        
        st.write("**Recommendations:**")
        for rec in tech_assessment.get("Recommendations", []):
            st.write(f"- {rec}")

def render_competitor_analysis():
    """Render the competitor analysis page."""
    st.title("⚔️ Competitor Analysis")
    
    # Input form
    with st.form("competitor_analysis_form"):
        company = st.text_input("Your Company", placeholder="e.g. Tesla")
        industry = st.selectbox(
            "Industry", 
            ["", "Technology", "Automotive", "Healthcare", "Finance", "Retail", 
             "Energy", "Manufacturing", "Consumer Goods", "Telecommunications", "Other"]
        )
        
        # Competitor inputs
        st.subheader("Competitors")
        cols = st.columns(2)
        competitors = []
        
        with cols[0]:
            competitor1 = st.text_input("Competitor 1", placeholder="e.g. Ford")
            competitor3 = st.text_input("Competitor 3 (optional)", placeholder="e.g. BMW")
        
        with cols[1]:
            competitor2 = st.text_input("Competitor 2", placeholder="e.g. Toyota")
            competitor4 = st.text_input("Competitor 4 (optional)", placeholder="e.g. BYD")
        
        for comp in [competitor1, competitor2, competitor3, competitor4]:
            if comp:
                competitors.append(comp)
        
        submit_button = st.form_submit_button("Analyze Competitors")
    
    if submit_button and company and len(competitors) > 0:
        with st.spinner(f"Comparing {company} with {', '.join(competitors)}..."):
            # Generate the analysis
            cache_key = f"{company}_vs_{'_'.join(competitors)}"
            if cache_key in st.session_state.analysis_cache:
                analysis = st.session_state.analysis_cache[cache_key]
                st.success("Retrieved analysis from cache")
            else:
                analysis = load_market_research().analyze_competitors(company, competitors, industry)
                st.session_state.analysis_cache[cache_key] = analysis
                
                # Update recent analyses
                st.session_state.recent_analyses[f"{company} vs Competitors"] = datetime.now().strftime("%Y-%m-%d %H:%M")
                st.success(f"Competitor analysis completed!")
            
            # Display analysis results
            display_competitor_analysis(company, competitors, analysis)

def display_competitor_analysis(company, competitors, analysis):
    """Display competitor analysis results."""
    all_companies = [company] + competitors
    
    # Market Share
    st.subheader("Comparative Market Share")
    market_share = analysis.get("Comparative Market Share", {})
    
    # Check if market share data exists
    if market_share:
        # Create data for visualization
        companies = list(market_share.keys())
        shares = []
        
        for comp in companies:
            share_str = market_share.get(comp, "0%")
            try:
                share = float(share_str.strip('%'))
            except:
                share = 0
            shares.append(share)
        
        # Create the bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=companies, 
                y=shares,
                text=[f"{share:.1f}%" for share in shares],
                textposition='auto'
            )
        ])
        
        # Update layout
        fig.update_layout(
            title=f"Market Share Comparison",
            xaxis_title="Companies",
            yaxis_title="Market Share (%)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Display visualizations if available
    for viz_path in analysis.get("visualizations", []):
        try:
            with open(viz_path, 'r') as f:
                html_content = f.read()
                st.components.v1.html(html_content, height=500)
        except Exception as e:
            st.warning(f"Could not load visualization: {str(e)}")
    
    # Strengths & Weaknesses
    st.subheader("Strengths and Weaknesses")
    strengths_weaknesses = analysis.get("Strengths and Weaknesses", {})
    
    tabs = st.tabs(all_companies)
    for i, comp in enumerate(all_companies):
        with tabs[i]:
            if comp in strengths_weaknesses:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Strengths:**")
                    strengths = strengths_weaknesses[comp].get("Strengths", ["No data"])
                    for strength in strengths:
                        st.write(f"- {strength}")
                
                with col2:
                    st.write("**Weaknesses:**")
                    weaknesses = strengths_weaknesses[comp].get("Weaknesses", ["No data"])
                    for weakness in weaknesses:
                        st.write(f"- {weakness}")
            else:
                st.write("No strengths and weaknesses data available")
    
    # Other analysis sections
    display_competitor_comparison_sections(all_companies, analysis)

def display_competitor_comparison_sections(companies, analysis):
    """Display additional competitor analysis sections."""
    sections = [
        ("Product/Service Comparison", "Features and capabilities comparison"),
        ("Pricing Strategy Differences", "Pricing approaches"),
        ("Technology and Innovation", "Tech focus and innovation rates"),
        ("Customer Experience", "Service quality and user experience"),
        ("Distribution Channels", "Market reach and distribution methods")
    ]
    
    for section_name, description in sections:
        if section_name in analysis:
            st.subheader(section_name)
            section_data = analysis[section_name]
            
            # Create a table if possible
            if all(isinstance(section_data.get(company, {}), dict) for company in companies):
                # Try to create a DataFrame for better display
                try:
                    # Extract all possible factors
                    all_factors = set()
                    for company in section_data:
                        all_factors.update(section_data[company].keys())
                    
                    # Create a DataFrame
                    comparison_data = []
                    for factor in all_factors:
                        row = {"Factor": factor}
                        for company in companies:
                            if company in section_data:
                                row[company] = section_data[company].get(factor, "N/A")
                            else:
                                row[company] = "N/A"
                        comparison_data.append(row)
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True)
                except:
                    # Fallback to simple display
                    st.json(section_data)
            else:
                # Simple text display
                for company in companies:
                    if company in section_data:
                        st.write(f"**{company}:** {section_data[company]}")

def render_market_trends():
    """Render the market trends analysis page."""
    st.title("📈 Market Trends Analysis")
    
    # Input form
    with st.form("market_trends_form"):
        industry = st.selectbox(
            "Industry", 
            ["Technology", "Automotive", "Healthcare", "Finance", "Retail", 
             "Energy", "Manufacturing", "Consumer Goods", "Telecommunications", "Other"]
        )
        
        timeframe = st.selectbox(
            "Timeframe",
            ["current", "emerging", "future"]
        )
        
        submit_button = st.form_submit_button("Analyze Market Trends")
    
    if submit_button:
        with st.spinner(f"Analyzing {industry} market trends..."):
            # Check cache first
            cache_key = f"{industry}_{timeframe}_trends"
            if cache_key in st.session_state.analysis_cache:
                trends = st.session_state.analysis_cache[cache_key]
                st.success("Retrieved analysis from cache")
            else:
                # Generate trends analysis
                trends = load_market_research().analyze_industry_trends(industry, timeframe)
                st.session_state.analysis_cache[cache_key] = trends
                
                # Update recent analyses
                st.session_state.recent_analyses[f"{industry} {timeframe} trends"] = datetime.now().strftime("%Y-%m-%d %H:%M")
                st.success(f"Market trends analysis for {industry} completed!")
            
            # Display trends analysis
            display_market_trends(industry, trends, timeframe)

def display_market_trends(industry, trends, timeframe):
    """Display market trends analysis results."""
    # Display visualizations if available
    for viz_path in trends.get("visualizations", []):
        try:
            with open(viz_path, 'r') as f:
                html_content = f.read()
                st.components.v1.html(html_content, height=500)
        except Exception as e:
            st.warning(f"Could not load visualization: {str(e)}")
    
    # Major Market Trends
    st.subheader("Major Market Trends")
    major_trends = trends.get("Major Market Trends", [])
    
    for i, trend in enumerate(major_trends):
        with st.expander(f"{i+1}. {trend.get('trend', 'Trend')} (Impact: {trend.get('impact', 'Unknown')})"):
            st.write(f"**Probability:** {trend.get('probability', 'Unknown')}")
            st.write(f"**Description:** {trend.get('description', 'No description')}")
            
            st.write("**Strategic Implications:**")
            for implication in trend.get("strategic_implications", []):
                st.write(f"- {implication}")
    
    # Technology Trends
    st.subheader("Technology Trends")
    tech_trends = trends.get("Technology Trends", [])
    
    col1, col2 = st.columns(2)
    
    with col1:
        for trend in tech_trends:
            st.write(f"**{trend.get('trend', 'Unknown')}**")
            st.write(f"- Impact: {trend.get('impact', 'Unknown')}")
            st.write(f"- Adoption Stage: {trend.get('adoption_stage', 'Unknown')}")
            st.write("")
    
    # Display other trend categories
    display_trend_categories(trends)

def display_trend_categories(trends):
    """Display additional trend categories."""
    categories = [
        ("Consumer Behavior Shifts", "shift"),
        ("Regulatory Developments", "development"),
        ("Competitive Landscape Evolution", "shift")
    ]
    
    for category_name, item_key in categories:
        st.subheader(category_name)
        items = trends.get(category_name, [])
        
        for item in items:
            with st.expander(f"{item.get(item_key, 'Item')} (Impact: {item.get('impact', 'Unknown')})"):
                for key, value in item.items():
                    if key not in [item_key, 'impact'] and not isinstance(value, list):
                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                
                for key, value in item.items():
                    if isinstance(value, list) and value:
                        st.write(f"**{key.replace('_', ' ').title()}:**")
                        for v in value:
                            st.write(f"- {v}")

def render_sentiment_analysis():
    """Render the sentiment analysis page."""
    st.title("💬 Market Sentiment Analysis")
    
    st.write("""
    This tool analyzes sentiment in market-related content such as news articles, social media posts, 
    and customer feedback to gauge market perception.
    """)
    
    # Setup tabs for different sentiment analysis tools
    sentiment_tabs = st.tabs(["General Analysis", "Competitor Sentiment", "Sentiment Drivers"])
    
    with sentiment_tabs[0]:
        render_general_sentiment_analysis()
    
    with sentiment_tabs[1]:
        render_competitor_sentiment_analysis()
    
    with sentiment_tabs[2]:
        render_sentiment_drivers_analysis()

def render_general_sentiment_analysis():
    """Render the general sentiment analysis section."""
    st.subheader("General Sentiment Analysis")
    
    # Text input area for sentiment analysis
    text_input = st.text_area(
        "Enter market-related text content for sentiment analysis:",
        height=150,
        placeholder="Paste news articles, social media posts, or other content for analysis..."
    )
    
    # Source selection
    source_type = st.selectbox(
        "Content Source",
        ["News Article", "Social Media", "Customer Feedback", "Analyst Report", "Other"]
    )
    
    if st.button("Analyze Sentiment"):
        if text_input:
            with st.spinner("Analyzing sentiment..."):
                # Prepare input for analysis
                text_entry = {
                    "text": text_input,
                    "source": source_type,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
                }
                
                # Perform sentiment analysis
                sentiment_result = load_sentiment_analyzer().analyze_market_sentiment([text_entry])
                
                # Display results
                display_sentiment_analysis_results(sentiment_result, text_input)
        else:
            st.warning("Please enter some text to analyze")

def render_competitor_sentiment_analysis():
    """Render the competitor sentiment analysis section."""
    st.subheader("Competitor Sentiment Comparison")
    
    st.write("""
    Compare sentiment across multiple competitors to understand market perception differences.
    Enter text content related to each competitor (news, social media, reviews, etc.).
    """)
    
    # Setup for multiple competitors
    num_competitors = st.number_input("Number of competitors to analyze", min_value=2, max_value=5, value=2)
    
    # Create input fields for each competitor
    competitor_names = []
    competitor_texts = {}
    
    for i in range(num_competitors):
        st.markdown(f"### Competitor {i+1}")
        name = st.text_input(f"Competitor {i+1} Name", key=f"comp_name_{i}")
        text = st.text_area(
            f"Enter text content related to {name if name else f'Competitor {i+1}'}",
            height=100,
            key=f"comp_text_{i}"
        )
        
        if name and text:
            competitor_names.append(name)
            if name not in competitor_texts:
                competitor_texts[name] = []
            
            competitor_texts[name].append({
                "text": text,
                "source": "User Input",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
            })
    
    if st.button("Compare Sentiment"):
        if len(competitor_texts) >= 2:
            with st.spinner("Analyzing competitor sentiment..."):
                # Perform comparison analysis
                comparison_results = load_sentiment_analyzer().track_competitor_sentiment(competitor_texts)
                
                # Display results
                display_competitor_sentiment_results(comparison_results)
        else:
            st.warning("Please enter information for at least two competitors")

def render_sentiment_drivers_analysis():
    """Render the sentiment drivers analysis section."""
    st.subheader("Sentiment Drivers Analysis")
    
    st.write("""
    Identify key factors driving positive or negative sentiment in market-related content.
    This helps understand what aspects are influencing market perception.
    """)
    
    text_input = st.text_area(
        "Enter a larger body of text for sentiment driver analysis:",
        height=200,
        placeholder="Paste multiple paragraphs, articles, or comments for analysis..."
    )
    
    sentiment_focus = st.radio(
        "Focus on specific sentiment type:",
        ["all", "positive", "negative", "neutral"]
    )
    
    if st.button("Identify Sentiment Drivers"):
        if text_input and len(text_input.split()) >= 50:
            with st.spinner("Analyzing sentiment drivers..."):
                # First get sentiment
                sentiment_result = load_sentiment_analyzer().analyze_market_sentiment([
                    {"text": text_input, "source": "User Input"}
                ])
                
                # Then identify drivers
                driver_result = load_sentiment_analyzer().identify_sentiment_drivers(
                    sentiment_result.get("detailed_results", []),
                    sentiment_type=sentiment_focus
                )
                
                # Display results
                display_sentiment_drivers_results(driver_result)
        else:
            st.warning("Please enter a substantial amount of text (at least 50 words) for meaningful analysis")

def display_sentiment_analysis_results(sentiment_result, text_input):
    """Display general sentiment analysis results."""
    if "error" in sentiment_result:
        st.error(sentiment_result["error"])
        return
    
    detailed_results = sentiment_result.get("detailed_results", [])
    if not detailed_results:
        st.warning("No sentiment results returned")
        return
    
    result = detailed_results[0]
    
    # Display sentiment result
    sentiment = result.get("sentiment", "neutral")
    score = result.get("score", 0)
    
    if sentiment == "positive":
        st.success(f"Sentiment: Positive (Score: {score:.2f})")
    elif sentiment == "negative":
        st.error(f"Sentiment: Negative (Score: {score:.2f})")
    else:
        st.info(f"Sentiment: Neutral (Score: {score:.2f})")
    
    # Extract key insights
    st.subheader("Key Insights")
    insights = load_nlp_processor().extract_key_insights(text_input, num_insights=3)
    for insight in insights:
        st.write(f"- {insight}")
    
    # Extract entities
    entities = load_nlp_processor().extract_entities(text_input)
    
    # Display entities
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Companies Mentioned")
        companies = entities.get("companies", [])
        if companies:
            for company in companies:
                st.write(f"- {company}")
        else:
            st.write("No companies detected")
    
    with col2:
        st.subheader("Technologies Mentioned")
        technologies = entities.get("technologies", [])
        if technologies:
            for tech in technologies:
                st.write(f"- {tech}")
        else:
            st.write("No technologies detected")

def display_competitor_sentiment_results(comparison_results):
    """Display competitor sentiment comparison results."""
    if "error" in comparison_results:
        st.error(comparison_results["error"])
        return
    
    # Show visualization if available
    viz_path = comparison_results.get("visualization_path")
    if viz_path and os.path.exists(viz_path):
        try:
            with open(viz_path, 'r') as f:
                html_content = f.read()
                st.components.v1.html(html_content, height=500)
        except Exception as e:
            st.warning(f"Could not load visualization: {str(e)}")
    
    # Display comparative analysis
    comparative = comparison_results.get("comparative_analysis", {})
    companies = comparative.get("competitors", [])
    
    # Display detailed results for each competitor
    for comp in companies:
        with st.expander(f"Detailed Analysis: {comp}"):
            comp_results = comparison_results.get("competitor_results", {}).get(comp, {})
            
            st.write(f"**Total Texts Analyzed:** {comp_results.get('total_texts', 0)}")
            st.write(f"**Average Sentiment Score:** {comp_results.get('average_sentiment_score', 0):.2f}")
            
            st.write("**Sentiment Distribution:**")
            distribution = comp_results.get("sentiment_distribution", {})
            
            # Create distribution chart
            fig = px.pie(
                names=["Positive", "Neutral", "Negative"],
                values=[
                    distribution.get("positive", 0) * 100,
                    distribution.get("neutral", 0) * 100,
                    distribution.get("negative", 0) * 100
                ],
                title="Sentiment Distribution",
                color_discrete_sequence=["green", "gray", "red"]
            )
            
            st.plotly_chart(fig, use_container_width=True)

def display_sentiment_drivers_results(driver_result):
    """Display sentiment drivers analysis results."""
    if "error" in driver_result:
        st.error(driver_result["error"])
        return
    
    st.write(f"**Analysis Focus:** {driver_result.get('sentiment_type', 'all')} sentiment")
    st.write(f"**Text segments analyzed:** {driver_result.get('text_count', 0)}")
    
    # Display key phrases
    st.subheader("Key Sentiment-Driving Phrases")
    for phrase in driver_result.get("key_phrases", []):
        st.write(f"- {phrase}")
    
    # Display top entities
    top_entities = driver_result.get("top_entities", {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Companies Driving Sentiment")
        companies = top_entities.get("companies", [])
        if companies:
            # Create DataFrame for better display
            company_names = [c[0] for c in companies]
            company_counts = [c[1] for c in companies]
            
            # Create chart
            fig = px.bar(
                x=company_names,
                y=company_counts,
                title="Companies Mentioned"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No companies detected")
    
    with col2:
        st.subheader("Technologies Driving Sentiment")
        technologies = top_entities.get("technologies", [])
        if technologies:
            # Create DataFrame for better display
            tech_names = [t[0] for t in technologies]
            tech_counts = [t[1] for t in technologies]
            
            # Create chart
            fig = px.bar(
                x=tech_names,
                y=tech_counts,
                title="Technologies Mentioned"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No technologies detected")

def render_trend_forecasting():
    """Render the trend forecasting page."""
    st.title("🔮 Market Trend Forecasting")
    
    st.write("""
    Forecast future market trends based on historical data and identify seasonal patterns to inform strategic planning.
    """)
    
    # Setup tabs for different forecasting tools
    forecast_tabs = st.tabs(["Data Upload & Forecast", "Seasonality Analysis"])
    
    with forecast_tabs[0]:
        render_trend_forecast()
    
    with forecast_tabs[1]:
        render_seasonality_analysis()

def render_trend_forecast():
    """Render the trend forecast section."""
    st.subheader("Market Trend Forecasting")
    
    st.write("""
    Upload a CSV file with historical market data to generate forecasts. 
    The file should include a date/time column and at least one numeric metric column.
    """)
    
    # File upload
    uploaded_file = st.file_uploader("Upload historical data (CSV)", type=["csv"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        forecast_periods = st.number_input("Forecast Periods", min_value=1, max_value=12, value=4)
    
    with col2:
        confidence = st.slider("Confidence Interval", min_value=0.80, max_value=0.99, value=0.95, step=0.01)
    
    if uploaded_file is not None:
        try:
            # Load the data
            data = pd.read_csv(uploaded_file)
            
            # Show data preview
            st.subheader("Data Preview")
            st.dataframe(data.head(), use_container_width=True)
            
            # Column selection
            st.subheader("Column Selection")
            
            date_col = find_date_column(data)
            if not date_col:
                date_col = st.selectbox("Select Date/Time Column", list(data.columns))
            else:
                date_col = st.selectbox("Select Date/Time Column", list(data.columns), index=list(data.columns).index(date_col))
            
            numeric_cols = find_numeric_columns(data, exclude_cols=[date_col])
            if not numeric_cols:
                st.warning("No numeric columns found in the data")
            else:
                selected_metrics = st.multiselect("Select Metrics to Forecast", numeric_cols, default=[numeric_cols[0]] if numeric_cols else [])
                
                if selected_metrics and st.button("Generate Forecast"):
                    with st.spinner("Generating forecast..."):
                        # Filter to only selected columns
                        forecast_data = data[[date_col] + selected_metrics].copy()
                        
                        # Generate forecast
                        forecast_result = load_trend_forecaster().forecast_market_trends(
                            forecast_data,
                            periods=forecast_periods,
                            confidence_interval=confidence
                        )
                        
                        # Display results
                        display_forecast_results(forecast_result, selected_metrics)
        except Exception as e:
            st.error(f"Error processing the data: {str(e)}")

def render_seasonality_analysis():
    """Render the seasonality analysis section."""
    st.subheader("Seasonality Analysis")
    
    st.write("""
    Identify seasonal patterns in your market data to better understand cyclical trends and plan accordingly.
    Upload time series data with date/time information.
    """)
    
    # File upload
    uploaded_file = st.file_uploader("Upload time series data (CSV)", type=["csv"], key="seasonality_upload")
    
    if uploaded_file is not None:
        try:
            # Load the data
            data = pd.read_csv(uploaded_file)
            
            # Show data preview
            st.subheader("Data Preview")
            st.dataframe(data.head(), use_container_width=True)
            
            # Column selection
            st.subheader("Column Selection")
            
            date_col = find_date_column(data)
            if not date_col:
                date_col = st.selectbox("Select Date/Time Column", list(data.columns), key="seasonality_date")
            else:
                date_col = st.selectbox("Select Date/Time Column", list(data.columns), index=list(data.columns).index(date_col), key="seasonality_date")
            
            numeric_cols = find_numeric_columns(data, exclude_cols=[date_col])
            if not numeric_cols:
                st.warning("No numeric columns found in the data")
            else:
                selected_metrics = st.multiselect("Select Metrics to Analyze", numeric_cols, default=[numeric_cols[0]] if numeric_cols else [], key="seasonality_metrics")
                
                if selected_metrics and st.button("Analyze Seasonality"):
                    with st.spinner("Analyzing seasonality patterns..."):
                        # Ensure date column is datetime
                        try:
                            data[date_col] = pd.to_datetime(data[date_col])
                        except:
                            st.error("Could not convert the selected column to datetime format")
                            st.stop()
                        
                        # Filter to only selected columns
                        seasonality_data = data[[date_col] + selected_metrics].copy()
                        
                        # Analyze seasonality
                        seasonality_result = load_trend_forecaster().analyze_seasonality(seasonality_data)
                        
                        # Display results
                        display_seasonality_results(seasonality_result, selected_metrics)
        except Exception as e:
            st.error(f"Error processing the data: {str(e)}")

def display_forecast_results(forecast_result, selected_metrics):
    """Display forecast results."""
    if "error" in forecast_result:
        st.error(forecast_result["error"])
        return
    
    st.success(f"Forecast generated successfully for {len(selected_metrics)} metrics")
    
    # Display each metric forecast
    for metric in selected_metrics:
        st.subheader(f"{metric} Forecast")
        
        metric_forecast = forecast_result.get("forecasts", {}).get(metric, {})
        
        # Show forecast metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Forecast Model:** Linear Regression")
            st.write(f"**Confidence Interval:** {forecast_result.get('confidence_interval', 0.95):.0%}")
        
        with col2:
            metrics = metric_forecast.get("metrics", {})
            st.write(f"**RMSE:** {metrics.get('rmse', 'N/A'):.2f}")
            st.write(f"**MAE:** {metrics.get('mae', 'N/A'):.2f}")
        
        # Show forecast data
        forecast_data = metric_forecast.get("forecast_data", [])
        if forecast_data:
            # Create DataFrame for display
            forecast_df = pd.DataFrame(forecast_data)
            forecast_df = forecast_df[["period", "forecast", "lower_bound", "upper_bound"]]
            forecast_df.columns = ["Period", "Forecast", "Lower Bound", "Upper Bound"]
            
            st.dataframe(forecast_df, use_container_width=True)
            
            # Display visualization
            viz_path = metric_forecast.get("visualization_path")
            if viz_path and os.path.exists(viz_path):
                try:
                    with open(viz_path, 'r') as f:
                        html_content = f.read()
                        st.components.v1.html(html_content, height=500)
                except Exception as e:
                    st.warning(f"Could not load visualization: {str(e)}")

def display_seasonality_results(seasonality_result, selected_metrics):
    """Display seasonality analysis results."""
    if "error" in seasonality_result:
        st.error(seasonality_result["error"])
        return
    
    st.success("Seasonality analysis completed")
    
    # Display results for each metric
    seasonality_results = seasonality_result.get("seasonality_results", {})
    
    for metric in selected_metrics:
        if metric in seasonality_results:
            st.subheader(f"{metric} Seasonality Analysis")
            
            metric_seasonality = seasonality_results[metric]
            
            # Monthly seasonality
            monthly = metric_seasonality.get("monthly_seasonality", {})
            st.write(f"**Monthly Seasonality:** {monthly.get('strength', 'Unknown')}")
            st.write(f"**Peak Month:** {monthly.get('peak_month', 'N/A')}")
            st.write(f"**Trough Month:** {monthly.get('trough_month', 'N/A')}")
            st.write(f"**Variation:** {monthly.get('variation', 0):.1f}%")
            
            # Quarterly seasonality
            quarterly = metric_seasonality.get("quarterly_seasonality", {})
            st.write(f"**Quarterly Seasonality:** {quarterly.get('strength', 'Unknown')}")
            st.write(f"**Peak Quarter:** Q{quarterly.get('peak_quarter', 'N/A')}")
            st.write(f"**Trough Quarter:** Q{quarterly.get('trough_quarter', 'N/A')}")
            
            # Display visualization if available
            viz_path = metric_seasonality.get("visualization_path")
            if viz_path and os.path.exists(viz_path):
                try:
                    with open(viz_path, 'r') as f:
                        html_content = f.read()
                        st.components.v1.html(html_content, height=600)
                except Exception as e:
                    st.warning(f"Could not load visualization: {str(e)}")
            
            # Day of week seasonality if available
            if "day_of_week_seasonality" in metric_seasonality:
                dow = metric_seasonality["day_of_week_seasonality"]
                
                st.write(f"**Day of Week Seasonality:** {dow.get('strength', 'Unknown')}")
                
                # Map day numbers to names
                day_names = {
                    0: "Monday", 1: "Tuesday", 2: "Wednesday", 
                    3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"
                }
                
                peak_day = dow.get('peak_day', 0)
                trough_day = dow.get('trough_day', 0)
                
                st.write(f"**Peak Day:** {day_names.get(peak_day, peak_day)}")
                st.write(f"**Trough Day:** {day_names.get(trough_day, trough_day)}")

# Main router
if page == "Dashboard":
    render_dashboard()
elif page == "Company Analysis":
    render_company_analysis()
elif page == "Competitor Analysis":
    render_competitor_analysis()
elif page == "Market Trends":
    render_market_trends()
elif page == "Sentiment Analysis":
    render_sentiment_analysis()
elif page == "Trend Forecasting":
    render_trend_forecasting()

if __name__ == "__main__":
    # This runs when the script is executed directly
    pass