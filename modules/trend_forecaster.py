# Trend forecasting
"""
Market trend forecasting module.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from sklearn.linear_model import LinearRegression

from marketsense import config
from marketsense.core.data_analyzer import DataAnalyzer
from marketsense.utils.visualization import save_visualization
from marketsense.utils.data_helpers import (
    find_date_column, find_numeric_columns, ensure_datetime_column, 
    extract_date_components, safe_division
)

class TrendForecaster:
    """Advanced market trend forecasting module."""
    
    def __init__(self):
        """Initialize the Trend Forecaster."""
        self.data_analyzer = DataAnalyzer()
    
    def forecast_market_trends(self, historical_data: pd.DataFrame, 
                            periods: int = 4, 
                            confidence_interval: float = 0.95) -> Dict[str, Any]:
        """
        Forecast future market trends based on historical data.
        
        Args:
            historical_data: DataFrame with time series data
            periods: Number of periods to forecast
            confidence_interval: Confidence interval for forecast (0-1)
            
        Returns:
            Dictionary containing forecast results
        """
        # Validate input data
        if historical_data.empty:
            return {"error": "Empty historical data provided"}
        
        # Identify time column
        date_col = find_date_column(historical_data)
        if not date_col:
            return {"error": "No date or time column found in historical data"}
        
        # Identify numeric columns to forecast
        metric_columns = find_numeric_columns(historical_data, exclude_cols=[date_col])
        if not metric_columns:
            return {"error": "No numeric columns found for forecasting"}
        
        # Sort data by date
        sorted_data = historical_data.sort_values(date_col).copy()
        
        # Convert date column to datetime if not already
        sorted_data = ensure_datetime_column(sorted_data, date_col)
        
        # Generate forecasts for each metric
        forecasts = {}
        
        for metric in metric_columns:
            # Apply a simple forecasting method
            forecast_result = self._generate_simple_forecast(
                sorted_data[[date_col, metric]], 
                date_col=date_col,
                value_col=metric,
                periods=periods,
                confidence_interval=confidence_interval
            )
            
            forecasts[metric] = forecast_result
        
        # Combine results
        return {
            "forecast_date": datetime.now().strftime("%Y-%m-%d"),
            "metrics_forecasted": metric_columns,
            "forecast_periods": periods,
            "confidence_interval": confidence_interval,
            "forecasts": forecasts
        }
    
    def _generate_simple_forecast(self, data: pd.DataFrame, 
                               date_col: str, 
                               value_col: str,
                               periods: int = 4,
                               confidence_interval: float = 0.95) -> Dict[str, Any]:
        """Generate a simple forecast using linear regression."""
        # Extract the time series
        y = data[value_col].values
        
        # Create a time index (0, 1, 2, ...)
        X = np.arange(len(y)).reshape(-1, 1)
        
        # Fit a simple linear regression
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict on historical data
        y_pred = model.predict(X)
        
        # Calculate forecast error
        errors = y - y_pred
        mse = np.mean(errors**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(errors))
        
        # Calculate confidence interval
        z_value = stats.norm.ppf((1 + confidence_interval) / 2)
        error_margin = z_value * rmse
        
        # Generate future dates
        last_date = data[date_col].iloc[-1]
        future_dates = self._generate_future_dates(data[date_col], last_date, periods)
        
        # Create future X values
        X_future = np.arange(len(y), len(y) + periods).reshape(-1, 1)
        
        # Generate forecasts
        y_forecast = model.predict(X_future)
        
        # Create upper and lower bounds
        lower_bound = y_forecast - error_margin
        upper_bound = y_forecast + error_margin
        
        # Create visualization
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(
            x=data[date_col],
            y=data[value_col],
            mode='lines+markers',
            name='Historical Data'
        ))
        
        # Add forecast
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=y_forecast,
            mode='lines+markers',
            name='Forecast',
            line=dict(dash='dot')
        ))
        
        # Add confidence intervals
        fig.add_trace(go.Scatter(
            x=future_dates + future_dates[::-1],
            y=np.concatenate([upper_bound, lower_bound[::-1]]),
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name=f'{confidence_interval:.0%} Confidence Interval'
        ))
        
        fig.update_layout(
            title=f"{value_col} Forecast",
            xaxis_title="Time Period",
            yaxis_title=value_col
        )
        
        # Save visualization
        viz_path = save_visualization(fig, f"{value_col}_forecast")
        
        # Format forecast data for return
        forecast_data = []
        for i in range(periods):
            forecast_data.append({
                "period": future_dates[i],
                "forecast": float(y_forecast[i]),
                "lower_bound": float(lower_bound[i]),
                "upper_bound": float(upper_bound[i])
            })
        
        return {
            "model_type": "linear_regression",
            "metrics": {
                "rmse": float(rmse),
                "mae": float(mae)
            },
            "forecast_data": forecast_data,
            "visualization_path": viz_path
        }
    
    def _generate_future_dates(self, date_series, last_date, periods):
        """Generate future dates based on the pattern in historical dates."""
        if pd.api.types.is_datetime64_dtype(date_series):
            # If date column is datetime type, try to determine frequency
            if len(date_series) >= 2:
                # Calculate most common time difference
                time_diffs = date_series.diff().dropna()
                if len(time_diffs) > 0:
                    most_common_diff = time_diffs.mode()[0]
                    return [last_date + (i + 1) * most_common_diff for i in range(periods)]
            
            # Fallback to daily frequency
            return [last_date + pd.Timedelta(days=i+1) for i in range(periods)]
        else:
            # If not datetime, use simple numeric increments
            return [last_date + i + 1 for i in range(periods)]
    
    def analyze_seasonality(self, time_series_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze seasonality patterns in time series data.
        
        Args:
            time_series_data: DataFrame with time series data
            
        Returns:
            Dictionary containing seasonality analysis
        """
        # Validate input data
        if time_series_data.empty:
            return {"error": "Empty time series data provided"}
        
        # Identify time column
        date_col = find_date_column(time_series_data)
        if not date_col:
            return {"error": "No date or time column found in time series data"}
        
        # Identify numeric columns to analyze
        metric_columns = find_numeric_columns(time_series_data, exclude_cols=[date_col])
        if not metric_columns:
            return {"error": "No numeric columns found for seasonality analysis"}
        
        # Extract date components
        df = extract_date_components(time_series_data.copy(), date_col)
        
        # Analyze seasonality for each metric
        seasonality_results = {}
        
        for metric in metric_columns:
            # Monthly seasonality
            monthly_analysis = self._analyze_monthly_seasonality(df, metric)
            
            # Quarterly seasonality
            quarterly_analysis = self._analyze_quarterly_seasonality(df, metric)
            
            # Day of week seasonality (if applicable)
            dow_analysis = None
            if len(df['day_of_week'].unique()) > 1:
                dow_analysis = self._analyze_dow_seasonality(df, metric)
            
            # Create visualization
            fig = self._create_seasonality_visualization(
                monthly_analysis, quarterly_analysis, dow_analysis, metric
            )
            
            # Save visualization
            viz_path = save_visualization(fig, f"{metric}_seasonality")
            
            # Store seasonality results
            seasonality_results[metric] = {
                "monthly_seasonality": monthly_analysis,
                "quarterly_seasonality": quarterly_analysis,
                "visualization_path": viz_path
            }
            
            if dow_analysis:
                seasonality_results[metric]["day_of_week_seasonality"] = dow_analysis
        
        return {
            "metrics_analyzed": metric_columns,
            "seasonality_results": seasonality_results
        }
    
    def _analyze_monthly_seasonality(self, df: pd.DataFrame, metric: str) -> Dict[str, Any]:
        """Analyze monthly seasonality pattern."""
        monthly_avg = df.groupby('month')[metric].mean()
        monthly_std = df.groupby('month')[metric].std()
        
        # Create monthly seasonality index
        overall_avg = df[metric].mean()
        monthly_seasonality_index = (monthly_avg / overall_avg * 100)
        
        # Detect seasonal patterns
        monthly_variation = monthly_seasonality_index.max() - monthly_seasonality_index.min()
        
        # Determine seasonality strength
        if monthly_variation > 30:
            monthly_strength = "Strong"
        elif monthly_variation > 15:
            monthly_strength = "Moderate"
        else:
            monthly_strength = "Weak"
        
        return {
            "index": monthly_seasonality_index.to_dict(),
            "variation": float(monthly_variation),
            "strength": monthly_strength,
            "peak_month": int(monthly_seasonality_index.idxmax()),
            "trough_month": int(monthly_seasonality_index.idxmin())
        }
    
    def _analyze_quarterly_seasonality(self, df: pd.DataFrame, metric: str) -> Dict[str, Any]:
        """Analyze quarterly seasonality pattern."""
        quarterly_avg = df.groupby('quarter')[metric].mean()
        quarterly_std = df.groupby('quarter')[metric].std()
        
        # Create quarterly seasonality index
        overall_avg = df[metric].mean()
        quarterly_seasonality_index = (quarterly_avg / overall_avg * 100)
        
        # Detect seasonal patterns
        quarterly_variation = quarterly_seasonality_index.max() - quarterly_seasonality_index.min()
        
        # Determine seasonality strength
        if quarterly_variation > 25:
            quarterly_strength = "Strong"
        elif quarterly_variation > 10:
            quarterly_strength = "Moderate"
        else:
            quarterly_strength = "Weak"
        
        return {
            "index": quarterly_seasonality_index.to_dict(),
            "variation": float(quarterly_variation),
            "strength": quarterly_strength,
            "peak_quarter": int(quarterly_seasonality_index.idxmax()),
            "trough_quarter": int(quarterly_seasonality_index.idxmin())
        }
    
    def _analyze_dow_seasonality(self, df: pd.DataFrame, metric: str) -> Dict[str, Any]:
        """Analyze day of week seasonality pattern."""
        dow_avg = df.groupby('day_of_week')[metric].mean()
        dow_std = df.groupby('day_of_week')[metric].std()
        
        # Create day of week seasonality index
        overall_avg = df[metric].mean()
        dow_seasonality_index = (dow_avg / overall_avg * 100)
        
        # Detect seasonal patterns
        dow_variation = dow_seasonality_index.max() - dow_seasonality_index.min()
        
        # Determine seasonality strength
        if dow_variation > 20:
            dow_strength = "Strong"
        elif dow_variation > 10:
            dow_strength = "Moderate"
        else:
            dow_strength = "Weak"
        
        return {
            "index": dow_seasonality_index.to_dict(),
            "variation": float(dow_variation),
            "strength": dow_strength,
            "peak_day": int(dow_seasonality_index.idxmax()),
            "trough_day": int(dow_seasonality_index.idxmin())
        }
    
    def _create_seasonality_visualization(
        self, monthly_analysis, quarterly_analysis, dow_analysis, metric_name
    ) -> go.Figure:
        """Create seasonality visualization."""
        # Create a subplot figure
        rows = 3 if dow_analysis else 2
        fig = make_subplots(rows=rows, cols=1, 
                          subplot_titles=("Monthly Seasonality", "Quarterly Seasonality") + 
                                        (("Day of Week Seasonality",) if dow_analysis else ()))
        
        # Monthly seasonality
        monthly_index = pd.Series(monthly_analysis["index"])
        fig.add_trace(
            go.Bar(x=monthly_index.index, 
                  y=monthly_index.values,
                  name="Monthly"),
            row=1, col=1
        )
        
        # Add 100% reference line
        fig.add_trace(
            go.Scatter(x=[1, 12], y=[100, 100], 
                      mode='lines', line=dict(dash='dash', color='red'),
                      name="Baseline"),
            row=1, col=1
        )
        
        # Quarterly seasonality
        quarterly_index = pd.Series(quarterly_analysis["index"])
        fig.add_trace(
            go.Bar(x=quarterly_index.index, 
                  y=quarterly_index.values,
                  name="Quarterly"),
            row=2, col=1
        )
        
        # Add 100% reference line
        fig.add_trace(
            go.Scatter(x=[1, 4], y=[100, 100], 
                      mode='lines', line=dict(dash='dash', color='red'),
                      showlegend=False),
            row=2, col=1
        )
        
        # Day of week seasonality (if available)
        if dow_analysis:
            dow_index = pd.Series(dow_analysis["index"])
            day_names = {
                0: "Monday", 1: "Tuesday", 2: "Wednesday", 
                3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"
            }
            x_values = [day_names.get(day, day) for day in dow_index.index]
            
            fig.add_trace(
                go.Bar(x=x_values, 
                      y=dow_index.values,
                      name="Day of Week"),
                row=3, col=1
            )
            
            # Add 100% reference line
            fig.add_trace(
                go.Scatter(x=[x_values[0], x_values[-1]], y=[100, 100], 
                          mode='lines', line=dict(dash='dash', color='red'),
                          showlegend=False),
                row=3, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=f"{metric_name} Seasonality Analysis",
            height=300 * rows
        )
        
        return fig