# Configuration settings
"""
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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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
HF_MODEL_SENTIMENT = os.getenv("HF_MODEL_SENTIMENT", "distilbert-base-uncased-finetuned-sst-2-english")
HF_MODEL_NER = os.getenv("HF_MODEL_NER", "dslim/bert-base-NER")

# Processing parameters
NLP_BATCH_SIZE = int(os.getenv("NLP_BATCH_SIZE", "10"))
CACHE_EXPIRY_DAYS = int(os.getenv("CACHE_EXPIRY_DAYS", "7"))