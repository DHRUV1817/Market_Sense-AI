# Sentiment analysis
"""
Sentiment analysis module for market and competitor sentiment tracking.
"""

import pandas as pd
from typing import List, Dict, Any, Optional
import plotly.express as px
import plotly.graph_objects as go

from marketsense import config
from marketsense.core.nlp_processor import NLPProcessor
from marketsense.utils.visualization import save_visualization

class SentimentAnalyzer:
    """Advanced sentiment analysis module for market and competitor sentiment tracking."""
    
    def __init__(self):
        """Initialize the Sentiment Analyzer."""
        self.nlp_processor = NLPProcessor()
    
    def analyze_market_sentiment(self, texts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze sentiment in market-related texts.
        
        Args:
            texts: List of dictionaries with text content and metadata
                  Each dict should have at least 'text' and 'source' keys
            
        Returns:
            Dictionary containing sentiment analysis results
        """
        if not texts:
            return {"error": "No texts provided for analysis"}
        
        # Extract text content and prepare for batch processing
        text_contents = [item.get('text', '') for item in texts]
        
        # Perform sentiment analysis
        sentiment_results = self.nlp_processor.analyze_sentiment(text_contents)
        
        # Combine results with metadata
        for i, result in enumerate(sentiment_results):
            result.update({k: v for k, v in texts[i].items() if k != 'text'})
        
        # Aggregate statistics
        sentiment_counts = {
            "positive": len([r for r in sentiment_results if r['sentiment'] == 'positive']),
            "neutral": len([r for r in sentiment_results if r['sentiment'] == 'neutral']),
            "negative": len([r for r in sentiment_results if r['sentiment'] == 'negative'])
        }
        
        sentiment_distribution = {
            "positive": sentiment_counts["positive"] / len(sentiment_results) if sentiment_results else 0,
            "neutral": sentiment_counts["neutral"] / len(sentiment_results) if sentiment_results else 0,
            "negative": sentiment_counts["negative"] / len(sentiment_results) if sentiment_results else 0
        }
        
        average_score = sum(r['score'] for r in sentiment_results) / len(sentiment_results) if sentiment_results else 0
        
        # Group sentiment by source if available
        source_sentiment = self._aggregate_by_source(sentiment_results)
        
        # Return consolidated results
        return {
            "total_texts": len(texts),
            "sentiment_counts": sentiment_counts,
            "sentiment_distribution": sentiment_distribution,
            "average_sentiment_score": average_score,
            "source_sentiment": source_sentiment,
            "detailed_results": sentiment_results
        }
    
    def track_competitor_sentiment(self, competitor_texts: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Track sentiment around different competitors.
        
        Args:
            competitor_texts: Dictionary mapping competitor names to lists of text dictionaries
            
        Returns:
            Dictionary containing comparative sentiment analysis
        """
        results = {}
        
        # Analyze sentiment for each competitor
        for competitor, texts in competitor_texts.items():
            if texts:
                competitor_sentiment = self.analyze_market_sentiment(texts)
                results[competitor] = competitor_sentiment
        
        if not results:
            return {"error": "No valid competitor texts provided"}
        
        # Create comparative analysis
        comparative = {
            "competitors": list(results.keys()),
            "sentiment_scores": {comp: results[comp]["average_sentiment_score"] for comp in results},
            "positive_mentions": {comp: results[comp]["sentiment_counts"]["positive"] for comp in results},
            "negative_mentions": {comp: results[comp]["sentiment_counts"]["negative"] for comp in results},
            "sentiment_ratios": {}
        }
        
        # Calculate positive-to-negative ratios
        for comp in results:
            positive = results[comp]["sentiment_counts"]["positive"]
            negative = results[comp]["sentiment_counts"]["negative"]
            ratio = positive / negative if negative > 0 else positive if positive > 0 else 0
            comparative["sentiment_ratios"][comp] = ratio
        
        # Create visualization data
        viz_data = pd.DataFrame({
            'Competitor': list(comparative["sentiment_scores"].keys()),
            'Sentiment Score': list(comparative["sentiment_scores"].values()),
            'Positive Mentions': list(comparative["positive_mentions"].values()),
            'Negative Mentions': list(comparative["negative_mentions"].values()),
            'Positive-to-Negative Ratio': list(comparative["sentiment_ratios"].values())
        })
        
        # Create visualization and save it
        fig = self._create_competitor_sentiment_visualization(viz_data)
        viz_path = save_visualization(fig, "competitor_sentiment")
        
        return {
            "competitor_results": results,
            "comparative_analysis": comparative,
            "visualization_path": viz_path
        }
    
    def identify_sentiment_drivers(self, texts: List[Dict[str, Any]], 
                               sentiment_type: str = "all") -> Dict[str, Any]:
        """
        Identify key phrases and topics driving sentiment.
        
        Args:
            texts: List of dictionaries with text content and sentiment scores
            sentiment_type: Type of sentiment to analyze ("positive", "negative", "neutral", or "all")
            
        Returns:
            Dictionary containing sentiment drivers
        """
        if not texts:
            return {"error": "No texts provided for analysis"}
        
        # Filter texts by sentiment type if needed
        if sentiment_type != "all":
            filtered_texts = [t for t in texts if t.get('sentiment') == sentiment_type]
        else:
            filtered_texts = texts
        
        if not filtered_texts:
            return {"error": f"No texts with {sentiment_type} sentiment found"}
        
        # Extract entities and key phrases from texts
        all_entities = {'companies': [], 'locations': [], 'technologies': []}
        for text_item in filtered_texts:
            text = text_item.get('text', '')
            entities = self.nlp_processor.extract_entities(text)
            
            for entity_type in all_entities:
                all_entities[entity_type].extend(entities.get(entity_type, []))
        
        # Count entity occurrences
        entity_counts = self._count_entities(all_entities)
        
        # Sort entities by frequency and get top 10
        top_entities = {
            entity_type: sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]
            for entity_type, counts in entity_counts.items()
        }
        
        # Extract key phrases using NLP processor
        all_text = " ".join([t.get('text', '') for t in filtered_texts])
        key_phrases = self.nlp_processor.extract_key_insights(all_text, num_insights=5)
        
        return {
            "sentiment_type": sentiment_type,
            "text_count": len(filtered_texts),
            "top_entities": top_entities,
            "key_phrases": key_phrases
        }
    
    def _aggregate_by_source(self, sentiment_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Aggregate sentiment results by source."""
        source_sentiment = {}
        
        for result in sentiment_results:
            source = result.get('source', 'unknown')
            if source not in source_sentiment:
                source_sentiment[source] = {
                    "texts": 0,
                    "positive": 0,
                    "neutral": 0,
                    "negative": 0,
                    "average_score": 0
                }
            
            source_sentiment[source]["texts"] += 1
            source_sentiment[source][result['sentiment']] += 1
            source_sentiment[source]["average_score"] += result['score']
        
        # Calculate averages for sources
        for source in source_sentiment:
            if source_sentiment[source]["texts"] > 0:
                source_sentiment[source]["average_score"] /= source_sentiment[source]["texts"]
        
        return source_sentiment
    
    def _count_entities(self, entities_dict: Dict[str, List[str]]) -> Dict[str, Dict[str, int]]:
        """Count occurrences of entities."""
        entity_counts = {}
        
        for entity_type, entities in entities_dict.items():
            entity_counts[entity_type] = {}
            for entity in entities:
                if entity in entity_counts[entity_type]:
                    entity_counts[entity_type][entity] += 1
                else:
                    entity_counts[entity_type][entity] = 1
        
        return entity_counts
    
    def _create_competitor_sentiment_visualization(self, data: pd.DataFrame) -> go.Figure:
        """Create competitor sentiment visualization."""
        # Create two subplots: sentiment scores and positive/negative mentions
        from plotly.subplots import make_subplots
        
        fig = make_subplots(rows=1, cols=2, 
                          subplot_titles=("Sentiment Scores", "Positive vs Negative Mentions"),
                          specs=[[{"type": "bar"}, {"type": "bar"}]])
        
        # Add sentiment score bars
        fig.add_trace(
            go.Bar(
                x=data['Competitor'],
                y=data['Sentiment Score'],
                name="Sentiment Score",
                marker_color=data['Sentiment Score'].apply(
                    lambda x: 'green' if x > 0.25 else 'red' if x < -0.25 else 'gray'
                )
            ),
            row=1, col=1
        )
        
        # Add positive mentions bars
        fig.add_trace(
            go.Bar(
                x=data['Competitor'],
                y=data['Positive Mentions'],
                name="Positive Mentions",
                marker_color="green"
            ),
            row=1, col=2
        )
        
        # Add negative mentions bars
        fig.add_trace(
            go.Bar(
                x=data['Competitor'],
                y=data['Negative Mentions'],
                name="Negative Mentions",
                marker_color="red"
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Competitor Sentiment Analysis",
            height=500,
            barmode='group',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig