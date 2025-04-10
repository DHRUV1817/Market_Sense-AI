"""
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
