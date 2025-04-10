# Data analysis and visualization
"""
Data analysis module for processing and visualizing market data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from marketsense import config
from marketsense.utils.visualization import (
    save_visualization, create_bar_chart, create_line_chart, 
    create_pie_chart, create_scatter_chart, create_heatmap
)
from marketsense.utils.data_helpers import (
    find_date_column, find_numeric_columns, find_categorical_columns,
    ensure_datetime_column, extract_date_components, chunk_data, safe_division
)

class DataAnalyzer:
    """Data analysis module for processing and visualizing market data."""
    
    def __init__(self):
        """Initialize the Data Analyzer."""
        self.output_dir = config.OUTPUT_DIR
    
    def analyze_market_data(self, data: Union[pd.DataFrame, Dict[str, Any]], 
                         analysis_type: str = "trend") -> Dict[str, Any]:
        """
        Analyze market data based on the specified analysis type.
        
        Args:
            data: Input data as DataFrame or dictionary
            analysis_type: Type of analysis ("trend", "comparison", "segmentation")
            
        Returns:
            Dictionary containing analysis results
        """
        # Convert dictionary to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        
        # Perform the appropriate analysis based on type
        if analysis_type == "trend":
            return self._analyze_trends(data)
        elif analysis_type == "comparison":
            return self._analyze_comparison(data)
        elif analysis_type == "segmentation":
            return self._analyze_segmentation(data)
        else:
            raise ValueError(f"Unsupported analysis type: {analysis_type}")
    
    def _analyze_trends(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trends in time series data."""
        # Ensure data has a time/date column
        date_col = find_date_column(data)
        if not date_col:
            raise ValueError("No date or time column found in data")
        
        # Identify numeric columns that could be metrics
        metric_columns = find_numeric_columns(data, exclude_cols=[date_col])
        
        if not metric_columns:
            raise ValueError("No numeric metric columns found in data")
        
        # Calculate trends for each metric
        trends = {}
        
        for metric in metric_columns:
            # Sort by date
            sorted_data = data.sort_values(date_col)
            
            # Calculate basic trend metrics
            values = sorted_data[metric].values
            if len(values) < 2:
                continue
                
            start_value = values[0]
            end_value = values[-1]
            min_value = np.min(values)
            max_value = np.max(values)
            
            # Calculate growth rates
            absolute_change = end_value - start_value
            percentage_change = safe_division(absolute_change, start_value) * 100
            
            # Calculate trend direction and volatility
            if absolute_change > 0:
                direction = "upward"
            elif absolute_change < 0:
                direction = "downward"
            else:
                direction = "stable"
            
            # Calculate volatility
            mean_value = np.mean(values)
            volatility = safe_division(np.std(values), mean_value)
            
            # Create visualization
            fig = create_line_chart(
                data=sorted_data,
                x_col=date_col,
                y_cols=[metric],
                title=f"{metric} Trend Analysis"
            )
            
            # Save visualization
            viz_path = save_visualization(fig, f"{metric}_trend")
            
            # Store trend analysis
            trends[metric] = {
                "start_value": float(start_value),
                "end_value": float(end_value),
                "min_value": float(min_value),
                "max_value": float(max_value),
                "absolute_change": float(absolute_change),
                "percentage_change": float(percentage_change if not np.isnan(percentage_change) else 0),
                "direction": direction,
                "volatility": float(volatility if not np.isnan(volatility) else 0),
                "visualization_path": viz_path
            }
        
        return {
            "analysis_type": "trend",
            "metrics_analyzed": list(trends.keys()),
            "trends": trends
        }
    
    def _analyze_comparison(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze comparative data."""
        # Identify entity column (typically first non-numeric column)
        categorical_cols = find_categorical_columns(data)
        
        if not categorical_cols:
            raise ValueError("No entity column found in data")
        
        entity_col = categorical_cols[0]
        
        # Identify metric columns (numeric columns)
        metric_columns = find_numeric_columns(data, exclude_cols=[entity_col])
        
        if not metric_columns:
            raise ValueError("No numeric metric columns found in data")
        
        # Analyze each metric across entities
        comparisons = {}
        
        for metric in metric_columns:
            # Calculate summary statistics
            entities = data[entity_col].unique()
            values = data[metric].values
            
            highest_entity = data.loc[data[metric].idxmax(), entity_col]
            lowest_entity = data.loc[data[metric].idxmin(), entity_col]
            
            mean_value = np.mean(values)
            median_value = np.median(values)
            
            # Create visualization
            fig = create_bar_chart(
                data=data,
                x_col=entity_col,
                y_col=metric,
                title=f"{metric} Comparison by {entity_col}"
            )
            
            # Save visualization
            viz_path = save_visualization(fig, f"{metric}_comparison")
            
            # Store comparison analysis
            comparisons[metric] = {
                "highest_entity": highest_entity,
                "highest_value": float(data[metric].max()),
                "lowest_entity": lowest_entity,
                "lowest_value": float(data[metric].min()),
                "mean_value": float(mean_value),
                "median_value": float(median_value),
                "visualization_path": viz_path
            }
        
        return {
            "analysis_type": "comparison",
            "entity_column": entity_col,
            "entities": list(entities),
            "metrics_analyzed": list(comparisons.keys()),
            "comparisons": comparisons
        }
    
    def _analyze_segmentation(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data segmentation."""
        # Identify segment columns (categorical columns)
        segment_columns = find_categorical_columns(data, max_categories=10)
        
        if not segment_columns:
            raise ValueError("No segment columns found in data")
        
        # Identify metric columns (numeric columns to be analyzed)
        metric_columns = find_numeric_columns(data, exclude_cols=segment_columns)
        
        if not metric_columns:
            raise ValueError("No numeric metric columns found in data")
        
        # Analyze each segment's impact on metrics
        segmentation = {}
        
        # Limit to 2 segment columns for simplicity
        for segment_col in segment_columns[:2]:
            segment_values = data[segment_col].unique()
            
            segment_analysis = {}
            
            # Limit to 3 metrics for simplicity
            for metric in metric_columns[:3]:
                # Calculate metrics by segment
                segment_metrics = {}
                
                for segment in segment_values:
                    segment_data = data[data[segment_col] == segment]
                    if len(segment_data) > 0:
                        segment_metrics[segment] = {
                            "mean": float(segment_data[metric].mean()),
                            "median": float(segment_data[metric].median()),
                            "min": float(segment_data[metric].min()),
                            "max": float(segment_data[metric].max()),
                            "count": int(len(segment_data))
                        }
                
                # Create visualization for segment comparison
                fig = go.Figure()
                
                for segment in segment_values:
                    segment_data = data[data[segment_col] == segment]
                    if len(segment_data) > 0:
                        fig.add_trace(go.Box(
                            y=segment_data[metric],
                            name=str(segment),
                            boxmean=True
                        ))
                
                fig.update_layout(
                    title=f"{metric} Distribution by {segment_col}",
                    xaxis_title=segment_col,
                    yaxis_title=metric
                )
                
                # Save visualization
                viz_path = save_visualization(fig, f"{metric}_by_{segment_col}")
                
                # Calculate segment impact (variability explained by segmentation)
                total_variance = np.var(data[metric])
                weighted_within_variance = 0
                
                for segment in segment_values:
                    segment_data = data[data[segment_col] == segment]
                    if len(segment_data) > 0:
                        segment_variance = np.var(segment_data[metric])
                        segment_weight = len(segment_data) / len(data)
                        weighted_within_variance += segment_variance * segment_weight
                
                explained_variance = 1 - safe_division(weighted_within_variance, total_variance)
                
                segment_analysis[metric] = {
                    "segment_metrics": segment_metrics,
                    "explained_variance": float(explained_variance),
                    "visualization_path": viz_path
                }
            
            segmentation[segment_col] = segment_analysis
        
        return {
            "analysis_type": "segmentation",
            "segment_columns": list(segmentation.keys()),
            "metrics_analyzed": list(metric_columns[:3]),
            "segmentation": segmentation
        }
    
    def create_market_visualization(self, data: Union[pd.DataFrame, Dict[str, Any]], 
                                viz_type: str, title: str = "") -> str:
        """
        Create market data visualization.
        
        Args:
            data: Input data as DataFrame or dictionary
            viz_type: Visualization type ("bar", "line", "pie", "scatter", "heatmap")
            title: Visualization title
            
        Returns:
            Path to saved visualization
        """
        # Convert dictionary to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        
        # Create visualization based on type
        if viz_type == "bar":
            # Find appropriate columns for bar chart
            categorical_cols = find_categorical_columns(data)
            numeric_cols = find_numeric_columns(data, categorical_cols)
            
            if not categorical_cols or not numeric_cols:
                raise ValueError("Need at least one categorical and one numeric column for a bar chart")
            
            fig = create_bar_chart(data, categorical_cols[0], numeric_cols[0], title)
            
        elif viz_type == "line":
            # Find appropriate columns for line chart
            date_col = find_date_column(data)
            if not date_col:
                date_col = data.index.name or 'index'
                x_values = data.index
            else:
                x_values = data[date_col]
                
            numeric_cols = find_numeric_columns(data, [date_col] if date_col in data.columns else [])
            if not numeric_cols:
                raise ValueError("Need at least one numeric column for a line chart")
            
            # Limit to first 3 numeric columns for clarity
            y_cols = numeric_cols[:3]
            
            # Create a line chart
            fig = create_line_chart(data, date_col, y_cols, title)
            
        elif viz_type == "pie":
            # Find appropriate columns for pie chart
            categorical_cols = find_categorical_columns(data)
            numeric_cols = find_numeric_columns(data, categorical_cols)
            
            if not categorical_cols or not numeric_cols:
                raise ValueError("Need at least one categorical and one numeric column for a pie chart")
            
            fig = create_pie_chart(data, categorical_cols[0], numeric_cols[0], title)
            
        elif viz_type == "scatter":
            numeric_cols = find_numeric_columns(data)
            if len(numeric_cols) < 2:
                raise ValueError("Need at least two numeric columns for a scatter chart")
            
            categorical_cols = find_categorical_columns(data)
            color_col = categorical_cols[0] if categorical_cols else None
            
            fig = create_scatter_chart(data, numeric_cols[0], numeric_cols[1], title, color_col)
            
        elif viz_type == "heatmap":
            fig = create_heatmap(data, title)
            
        else:
            raise ValueError(f"Unsupported visualization type: {viz_type}")
        
        # Save visualization
        output_path = save_visualization(fig, f"{viz_type}_{title.replace(' ', '_').lower()}")
        
        return output_path