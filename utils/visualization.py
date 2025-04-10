# Visualization utilities
"""
Visualization utilities for MarketSense AI.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from marketsense import config

def save_visualization(fig: go.Figure, name: str, subdir: str = "") -> str:
    """
    Save a plotly figure to the output directory.
    
    Args:
        fig: Plotly figure to save
        name: Base name for the file
        subdir: Optional subdirectory within the output dir
    
    Returns:
        Path to the saved visualization file
    """
    # Create safe filename
    safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in name)
    safe_name = safe_name.replace(" ", "_").lower()
    
    # Create output directory if it doesn't exist
    output_dir = config.OUTPUT_DIR
    if subdir:
        output_dir = output_dir / subdir
        output_dir.mkdir(exist_ok=True)
    
    # Ensure filename is unique
    base_path = output_dir / f"{safe_name}.html"
    path = base_path
    counter = 1
    
    while path.exists():
        path = output_dir / f"{safe_name}_{counter}.html"
        counter += 1
    
    # Save the figure
    fig.write_html(path)
    return str(path)

def create_bar_chart(
    data: Union[pd.DataFrame, Dict[str, Any]], 
    x_col: str, 
    y_col: str, 
    title: str = "",
    color_col: Optional[str] = None
) -> go.Figure:
    """Create a bar chart."""
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    
    if color_col and color_col in data.columns:
        fig = px.bar(
            data,
            x=x_col,
            y=y_col,
            color=color_col,
            title=title or f"{y_col} by {x_col}"
        )
    else:
        fig = px.bar(
            data,
            x=x_col,
            y=y_col,
            title=title or f"{y_col} by {x_col}"
        )
    
    return fig

def create_line_chart(
    data: Union[pd.DataFrame, Dict[str, Any]],
    x_col: str,
    y_cols: List[str],
    title: str = ""
) -> go.Figure:
    """Create a line chart with multiple series."""
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    
    fig = go.Figure()
    
    for y_col in y_cols:
        if y_col in data.columns:
            fig.add_trace(go.Scatter(
                x=data[x_col],
                y=data[y_col],
                mode='lines+markers',
                name=y_col
            ))
    
    fig.update_layout(
        title=title or "Trend Analysis",
        xaxis_title=x_col,
        yaxis_title="Value"
    )
    
    return fig

def create_pie_chart(
    data: Union[pd.DataFrame, Dict[str, Any]],
    names_col: str,
    values_col: str,
    title: str = ""
) -> go.Figure:
    """Create a pie chart."""
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    
    fig = px.pie(
        data,
        names=names_col,
        values=values_col,
        title=title or f"{values_col} Distribution by {names_col}"
    )
    
    return fig

def create_scatter_chart(
    data: Union[pd.DataFrame, Dict[str, Any]],
    x_col: str,
    y_col: str,
    title: str = "",
    color_col: Optional[str] = None
) -> go.Figure:
    """Create a scatter chart."""
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    
    if color_col and color_col in data.columns:
        fig = px.scatter(
            data,
            x=x_col,
            y=y_col,
            color=color_col,
            title=title or f"{y_col} vs {x_col} by {color_col}"
        )
    else:
        fig = px.scatter(
            data,
            x=x_col,
            y=y_col,
            title=title or f"{y_col} vs {x_col}"
        )
    
    return fig

def create_heatmap(
    data: Union[pd.DataFrame, Dict[str, Any]],
    title: str = ""
) -> go.Figure:
    """Create a correlation heatmap."""
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    
    # Get numeric columns only
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        # Create an empty figure if not enough numeric columns
        fig = go.Figure()
        fig.update_layout(
            title="Not enough numeric data for correlation heatmap"
        )
        return fig
    
    # Calculate correlation matrix
    corr_matrix = data[numeric_cols].corr()
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        title=title or "Correlation Heatmap",
        color_continuous_scale="RdBu_r",
        origin="lower"
    )
    
    return fig

def create_swot_visualization(
    company_name: str,
    swot_data: Dict[str, List[str]]
) -> go.Figure:
    """Create SWOT analysis visualization."""
    # Convert SWOT data to format suitable for visualization
    strengths = swot_data.get("Strengths", ["No data"])
    weaknesses = swot_data.get("Weaknesses", ["No data"])
    opportunities = swot_data.get("Opportunities", ["No data"])
    threats = swot_data.get("Threats", ["No data"])
    
    # Create a figure with subplots
    fig = go.Figure()
    
    # Add each quadrant
    fig.add_trace(go.Scatter(
        x=[1], y=[1],
        mode="markers+text",
        marker=dict(size=100, color="green", opacity=0.3),
        text="Strengths",
        textposition="middle center",
        hoverinfo="text",
        hovertext="<br>".join(strengths),
        name="Strengths"
    ))
    
    fig.add_trace(go.Scatter(
        x=[-1], y=[1],
        mode="markers+text",
        marker=dict(size=100, color="red", opacity=0.3),
        text="Weaknesses",
        textposition="middle center",
        hoverinfo="text",
        hovertext="<br>".join(weaknesses),
        name="Weaknesses"
    ))
    
    fig.add_trace(go.Scatter(
        x=[1], y=[-1],
        mode="markers+text",
        marker=dict(size=100, color="blue", opacity=0.3),
        text="Opportunities",
        textposition="middle center",
        hoverinfo="text",
        hovertext="<br>".join(opportunities),
        name="Opportunities"
    ))
    
    fig.add_trace(go.Scatter(
        x=[-1], y=[-1],
        mode="markers+text",
        marker=dict(size=100, color="orange", opacity=0.3),
        text="Threats",
        textposition="middle center",
        hoverinfo="text",
        hovertext="<br>".join(threats),
        name="Threats"
    ))
    
    # Update layout
    fig.update_layout(
        title=f"SWOT Analysis for {company_name}",
        xaxis=dict(range=[-2, 2], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-2, 2], showgrid=False, zeroline=False, showticklabels=False),
        showlegend=True,
        width=800,
        height=800,
        annotations=[
            dict(x=0, y=0, showarrow=False, text="SWOT", font=dict(size=20))
        ]
    )
    
    return fig