# Natural language processing
"""
Natural Language Processing module for text analysis and insight extraction.
"""

import re
import os
import json
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path

from marketsense import config
from marketsense.utils.cache import cached
from marketsense.utils.data_helpers import normalize_text

# Import optional dependencies for HuggingFace models
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

class NLPProcessor:
    """Natural Language Processing module for text analysis and insight extraction."""
    
    def __init__(self):
        """Initialize the NLP Processor."""
        self.batch_size = config.NLP_BATCH_SIZE
        self._sentiment_analyzer = None
        self._ner_model = None
    
    @property
    def sentiment_analyzer(self):
        """Lazy-loaded sentiment analysis model."""
        if not HF_AVAILABLE:
            return None
            
        if self._sentiment_analyzer is None:
            try:
                self._sentiment_analyzer = pipeline(
                    "sentiment-analysis", 
                    model=config.HF_MODEL_SENTIMENT,
                    tokenizer=config.HF_MODEL_SENTIMENT
                )
            except Exception as e:
                print(f"Error loading sentiment model: {str(e)}")
                self._sentiment_analyzer = None
                
        return self._sentiment_analyzer
    
    @property
    def ner_model(self):
        """Lazy-loaded named entity recognition model."""
        if not HF_AVAILABLE:
            return None
            
        if self._ner_model is None:
            try:
                self._ner_model = pipeline(
                    "ner", 
                    model=config.HF_MODEL_NER,
                    tokenizer=config.HF_MODEL_NER,
                    aggregation_strategy="simple"
                )
            except Exception as e:
                print(f"Error loading NER model: {str(e)}")
                self._ner_model = None
                
        return self._ner_model
    
    def extract_key_insights(self, text: str, num_insights: int = 5) -> List[str]:
        """
        Extract key insights from a text document.
        
        Args:
            text: Input text to analyze
            num_insights: Number of insights to extract
            
        Returns:
            List of key insights
        """
        # Split text into sentences
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        
        # Remove very short sentences
        sentences = [s for s in sentences if len(s.split()) > 5]
        
        if not sentences:
            return ["No significant insights found in the text."]
            
        # In a real implementation, we would use embeddings or other NLP techniques
        # to identify the most important sentences
        
        # For demo purposes, select sentences with important keywords
        important_keywords = ["significant", "key", "important", "substantial", "major", 
                             "growth", "opportunity", "challenge", "trend", "strategy",
                             "increase", "decrease", "market", "competitor", "advantage"]
        
        scored_sentences = []
        for sentence in sentences:
            score = sum(1 for keyword in important_keywords if keyword.lower() in sentence.lower())
            scored_sentences.append((sentence, score))
        
        # Sort by score and take top insights
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s[0] for s in scored_sentences[:num_insights]]
        
        # If we don't have enough high-scoring sentences, add some random ones
        if len(top_sentences) < num_insights:
            remaining = num_insights - len(top_sentences)
            remaining_sentences = [s for s, _ in scored_sentences[num_insights:]]
            if remaining_sentences:
                top_sentences.extend(np.random.choice(remaining_sentences, 
                                                   size=min(remaining, len(remaining_sentences)), 
                                                   replace=False))
        
        return top_sentences
    
    def analyze_sentiment(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment of multiple text snippets.
        
        Args:
            texts: List of text snippets to analyze
            
        Returns:
            List of sentiment analysis results
        """
        results = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            batch_results = self._analyze_sentiment_batch(batch)
            results.extend(batch_results)
        
        return results
    
    def _analyze_sentiment_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyze sentiment for a batch of texts."""
        results = []
        
        # Try using HuggingFace model if available
        if HF_AVAILABLE and self.sentiment_analyzer:
            try:
                # Process with HuggingFace pipeline
                hf_results = self.sentiment_analyzer(texts)
                
                for i, hf_result in enumerate(hf_results):
                    text = texts[i]
                    label = hf_result['label']
                    score = hf_result['score']
                    
                    # Map HF sentiment labels to our format
                    if label == "POSITIVE":
                        sentiment = "positive"
                        sentiment_score = score
                    elif label == "NEGATIVE":
                        sentiment = "negative"
                        sentiment_score = -score
                    else:
                        sentiment = "neutral"
                        sentiment_score = 0
                    
                    results.append({
                        "text": text[:100] + "..." if len(text) > 100 else text,
                        "sentiment": sentiment,
                        "score": sentiment_score,
                        "confidence": score
                    })
                
                return results
            except Exception as e:
                print(f"Error using HuggingFace for sentiment: {str(e)}")
                # Fall back to rule-based approach
        
        # Rule-based fallback approach
        positive_words = ["good", "great", "excellent", "positive", "impressive", "growth", 
                        "opportunity", "success", "beneficial", "advantage", "profit"]
        negative_words = ["bad", "poor", "negative", "challenging", "risk", "threat", 
                        "decline", "weak", "failure", "disadvantage", "loss"]
        
        for text in texts:
            text_lower = text.lower()
            
            # Count positive and negative words
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            # Calculate sentiment score (-1 to 1)
            total = positive_count + negative_count
            if total == 0:
                sentiment_score = 0  # Neutral
            else:
                sentiment_score = (positive_count - negative_count) / total
            
            # Determine sentiment category
            if sentiment_score > 0.25:
                sentiment = "positive"
            elif sentiment_score < -0.25:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            
            # Format the result
            results.append({
                "text": text[:100] + "..." if len(text) > 100 else text,
                "sentiment": sentiment,
                "score": sentiment_score,
                "confidence": 0.7 + 0.3 * abs(sentiment_score)  # Higher confidence for stronger sentiment
            })
        
        return results
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary of entity types and extracted entities
        """
        # Try using HuggingFace NER model if available
        if HF_AVAILABLE and self.ner_model:
            try:
                # Extract entities using the HuggingFace model
                entities = self.ner_model(text)
                
                # Process and group entities
                grouped_entities = {
                    "persons": [],
                    "organizations": [],
                    "locations": [],
                    "misc": [],
                    "technologies": []  # We'll populate this with rule-based extraction
                }
                
                # Group entities by type
                current_entity = ""
                current_type = ""
                
                for entity in entities:
                    entity_text = entity["word"]
                    entity_type = entity["entity_group"]
                    
                    # Map HuggingFace entity types to our categories
                    if entity_type in ["B-PER", "I-PER"]:
                        if entity_type == "B-PER":
                            if current_entity and current_type == "person":
                                grouped_entities["persons"].append(current_entity.strip())
                            current_entity = entity_text
                            current_type = "person"
                        else:
                            current_entity += " " + entity_text
                    
                    elif entity_type in ["B-ORG", "I-ORG"]:
                        if entity_type == "B-ORG":
                            if current_entity and current_type == "organization":
                                grouped_entities["organizations"].append(current_entity.strip())
                            current_entity = entity_text
                            current_type = "organization"
                        else:
                            current_entity += " " + entity_text
                    
                    elif entity_type in ["B-LOC", "I-LOC"]:
                        if entity_type == "B-LOC":
                            if current_entity and current_type == "location":
                                grouped_entities["locations"].append(current_entity.strip())
                            current_entity = entity_text
                            current_type = "location"
                        else:
                            current_entity += " " + entity_text
                    
                    elif entity_type in ["B-MISC", "I-MISC"]:
                        if entity_type == "B-MISC":
                            if current_entity and current_type == "misc":
                                grouped_entities["misc"].append(current_entity.strip())
                            current_entity = entity_text
                            current_type = "misc"
                        else:
                            current_entity += " " + entity_text
                
                # Add the last entity if any
                if current_entity:
                    if current_type == "person":
                        grouped_entities["persons"].append(current_entity.strip())
                    elif current_type == "organization":
                        grouped_entities["organizations"].append(current_entity.strip())
                    elif current_type == "location":
                        grouped_entities["locations"].append(current_entity.strip())
                    elif current_type == "misc":
                        grouped_entities["misc"].append(current_entity.strip())
                
                # For backward compatibility, map organizations to companies
                grouped_entities["companies"] = grouped_entities["organizations"]
                
                # Extract technology mentions using rule-based approach
                grouped_entities["technologies"] = self._extract_technologies(text)
                
                return grouped_entities
            
            except Exception as e:
                print(f"Error using HuggingFace for NER: {str(e)}")
                # Fall back to rule-based approach
        
        # Rule-based fallback approach
        companies = set()
        locations = set()
        technologies = set()
        
        # Simple pattern matching
        # Look for company patterns (e.g., "Company X", "X Corp", "X Inc")
        company_patterns = [
            r'([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)\s+(?:Corporation|Corp|Inc|Ltd|LLC|Group|Company)',
            r'([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)\s+(?:&|and)\s+([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)'
        ]
        
        for pattern in company_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    companies.update(match)
                else:
                    companies.add(match)
        
        # Look for location patterns
        location_patterns = [
            r'in\s+([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)',
            r'from\s+([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)',
            r'at\s+([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)'
        ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, text)
            locations.update(matches)
        
        # Extract technologies
        technologies = self._extract_technologies(text)
        
        # Clean up extracted entities
        companies = [c.strip() for c in companies if len(c.strip()) > 1]
        locations = [l.strip() for l in locations if len(l.strip()) > 1]
        
        return {
            "companies": companies,
            "locations": locations,
            "technologies": list(technologies)
        }
    
    def _extract_technologies(self, text: str) -> List[str]:
        """Extract technology mentions from text."""
        technology_keywords = [
            "AI", "Artificial Intelligence", "Machine Learning", "ML", "Cloud", "Blockchain",
            "IoT", "Internet of Things", "Big Data", "Analytics", "5G", "Automation",
            "Robotics", "VR", "AR", "Virtual Reality", "Augmented Reality", "API",
            "SaaS", "PaaS", "IaaS", "DevOps", "Quantum Computing", "Edge Computing"
        ]
        
        found_technologies = set()
        for tech in technology_keywords:
            if re.search(r'\b' + re.escape(tech) + r'\b', text, re.IGNORECASE):
                found_technologies.add(tech)
        
        return list(found_technologies)
    
    def summarize_text(self, text: str, max_sentences: int = 3) -> str:
        """
        Generate a concise summary of a text.
        
        Args:
            text: Input text to summarize
            max_sentences: Maximum number of sentences in the summary
            
        Returns:
            Summarized text
        """
        # In a real implementation, use proper text summarization techniques
        # For demonstration, we'll use the key insight extraction to summarize
        
        insights = self.extract_key_insights(text, num_insights=max_sentences)
        return " ".join(insights)