"""
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
