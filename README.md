# MarketSense AI

MarketSense AI is a market intelligence platform that provides AI-powered analysis of companies, competitors, market trends, and sentiment to help businesses make data-driven decisions.

## Features

- **Company Analysis**: Get detailed insights about a company including SWOT analysis, market position, growth opportunities, and risks.
- **Competitor Analysis**: Compare a company against its competitors across multiple dimensions including market share, strengths/weaknesses, and product offerings.
- **Market Trends**: Identify current, emerging, and future trends in specific industries with impact assessments.
- **Sentiment Analysis**: Analyze sentiment in market-related content and track sentiment around different competitors.
- **Trend Forecasting**: Generate forecasts based on historical data and identify seasonality patterns.

## Project Structure

```
marketsense/
├── __init__.py           # Package initialization
├── config.py             # Configuration settings
├── core/                 # Core functionality modules
│   ├── __init__.py
│   ├── ai_engine.py      # AI-powered analysis engine
│   ├── data_analyzer.py  # Data analysis and visualization
│   └── nlp_processor.py  # Natural language processing
├── modules/              # Application modules
│   ├── __init__.py
│   ├── market_research.py    # Company and competitor analysis
│   ├── sentiment_analyzer.py # Sentiment analysis
│   └── trend_forecaster.py   # Trend forecasting
└── utils/                # Utility functions
    ├── __init__.py
    ├── cache.py          # Caching utilities
    └── data_helpers.py   # Data processing utilities
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Market_Sense-AI.git
   cd Market_Sense-AI
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`

4. Install dependencies:
   ```bash
   pip install -e .
   ```

5. Create a `.env` file in the project root with the following content:
   ```
   # API Keys
   OPENAI_API_KEY=your_openai_api_key_here

   # Application settings
   FLASK_ENV=development
   FLASK_DEBUG=1
   SECRET_KEY=your_secret_key_here

   # Other configurations
   CACHE_DIR=./marketsense/data/cache
   OUTPUT_DIR=./marketsense/data/outputs
   ```

## Usage

### Running the Flask API

```bash
python app.py
```

The API will be available at http://localhost:5000.

### Running the Streamlit Web App

```bash
python streamlit_app.py
```

The Streamlit app will be available at http://localhost:8501.

## API Endpoints

- `/api/health`: Health check endpoint
- `/api/company-analysis`: Analyze a company
- `/api/competitor-analysis`: Analyze competitors
- `/api/market-trends`: Analyze market trends
- `/api/sentiment-analysis`: Analyze sentiment in text content
- `/api/strategic-forecast`: Generate strategic forecasts

## Development

- Run tests: `pytest`
- Format code: `black .`

## License

[MIT License](LICENSE)