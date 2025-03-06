from typing import Dict, List, Tuple, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objects import Figure
import numpy as np

class ZeppVisualizer:
    """Visualization utilities for Zepp tennis data"""
    
    @staticmethod
    def create_shot_distribution_chart(shot_counts: Dict[str, int]) -> Figure:
        """
        Create a pie chart showing shot distribution
        
        Parameters:
        shot_counts (dict): Dictionary mapping shot types to counts
        
        Returns:
        plotly.graph_objects.Figure: Pie chart figure
        """
        # Filter out zero values
        filtered_counts = {k: v for k, v in shot_counts.items() if v > 0 and k != 'Total'}
        
        return px.pie(
            values=list(filtered_counts.values()),
            names=list(filtered_counts.keys()),
            title="Shot Distribution",
            color_discrete_sequence=px.colors.qualitative.Set1
        )
    
    @staticmethod
    def create_stacked_shot_distribution(shot_counts: Dict[str, int]) -> Figure:
        """
        Create a stacked horizontal bar chart showing shot distribution
        
        Parameters:
        shot_counts (dict): Dictionary mapping shot types to counts
        
        Returns:
        plotly.graph_objects.Figure: Stacked bar chart figure
        """
        # Filter out zero values and 'Total'
        filtered_counts = {k: v for k, v in shot_counts.items() 
                           if v > 0 and k != 'Total'}
        
        # Create mapping of shot types to colors
        colors = {
            'Serve': '#1f77b4',
            'Forehand': '#2ca02c',
            'Backhand': '#ff7f0e',
            'Volley': '#9467bd',
            'Smash': '#d62728'
        }
        
        # Calculate percentages
        total = sum(filtered_counts.values())
        percentages = {k: (v/total*100) for k, v in filtered_counts.items()}
        
        # Create figure
        fig = go.Figure()
        
        # Add a trace for each shot type
        for shot_type in filtered_counts.keys():
            fig.add_trace(go.Bar(
                x=[filtered_counts[shot_type]],
                y=['Breakdown'],
                orientation='h',
                name=shot_type,
                text=[f"{shot_type[:2]}: {percentages[shot_type]:.1f}%"],
                textposition='inside',
                marker_color=colors.get(shot_type, '#7f7f7f')
            ))
        
        fig.update_layout(
            barmode='stack',
            height=150,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showticklabels=False),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    
    @staticmethod
    def create_spin_analysis_chart(spin_data: List[Dict]) -> Tuple[Figure, pd.DataFrame]:
        """
        Create a bar chart for spin analysis
        
        Parameters:
        spin_data (list): List of dictionaries with spin data
        
        Returns:
        tuple: (plotly.graph_objects.Figure, pandas.DataFrame) - Bar chart and percentages
        """
        if not spin_data:
            return None, None
            
        spin_df = pd.DataFrame(spin_data)
        fig = px.bar(
            spin_df,
            x='motionType',
            y='count',
            color='spinType',
            barmode='group',
            title='Shot Distribution by Spin Type',
            labels={'motionType': 'Shot Type', 'count': 'Count', 'spinType': 'Spin Type'},
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        
        # Calculate percentages
        pivot = pd.pivot_table(
            spin_df,
            values='count',
            index='motionType',
            columns='spinType',
            aggfunc='sum'
        ).fillna(0)
        
        percentages = pivot.div(pivot.sum(axis=1), axis=0) * 100
        
        return fig, percentages
    
    @staticmethod
    def add_trend_analysis(
        fig: Figure,
        x: pd.Series,
        y: pd.Series,
        name: str,
        window_size: int = 5,
        remove_zeros: bool = False,
        y_min: Optional[float] = None,
        show_trendline: bool = True,
        show_rolling_avg: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Add trend analysis to an existing figure
        
        Parameters:
        fig (plotly.graph_objects.Figure): Figure to add trend line to
        x (pandas.Series): X-axis data (typically dates)
        y (pandas.Series): Y-axis data
        name (str): Name of the data series
        window_size (int): Window size for rolling average
        remove_zeros (bool): Whether to remove zero values
        y_min (float, optional): Minimum Y value to include
        show_trendline (bool): Whether to show trend line
        show_rolling_avg (bool): Whether to show rolling average
        
        Returns:
        pandas.DataFrame or None: Filtered data used for trend analysis
        """
        df = pd.DataFrame({'x': x, 'y': y})
        if remove_zeros:
            df = df[df['y'] > 0]
        if y_min is not None:
            df = df[df['y'] >= y_min]
        
        if df.empty:
            return None
            
        if show_rolling_avg:
            rolling_avg = df['y'].rolling(window=window_size, min_periods=1).mean()
            fig.add_trace(go.Scatter(
                x=df['x'],
                y=rolling_avg,
                name=f'{name} ({window_size}-session Rolling Avg)',
                line=dict(color='rgba(0,0,0,0.5)', width=2),
                opacity=0.7
            ))
        
        if show_trendline:
            x_numeric = pd.to_numeric(pd.to_datetime(df['x']))
            z = np.polyfit(x_numeric, df['y'], 1)
            p = np.poly1d(z)
            fig.add_trace(go.Scatter(
                x=df['x'],
                y=p(x_numeric),
                name=f'{name} Trend',
                line=dict(dash='dash'),
                opacity=0.5
            ))
        
        return df
    
    @staticmethod
    def create_radar_chart(metrics: Dict[str, float], max_value: float = 100) -> Figure:
        """
        Create a radar chart for displaying multiple metrics
        
        Parameters:
        metrics (dict): Dictionary mapping metric names to values
        max_value (float): Maximum value for scaling
        
        Returns:
        plotly.graph_objects.Figure: Radar chart figure
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=list(metrics.values()),
            theta=list(metrics.keys()),
            fill='toself',
            name='Session Metrics'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max_value]
                )
            ),
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_hit_points_chart(hit_points: List[List[float]], title: str) -> Figure:
        """
        Create a scatter plot for hit points
        
        Parameters:
        hit_points (list): List of [x, y] coordinate pairs
        title (str): Chart title
        
        Returns:
        plotly.graph_objects.Figure: Hit points scatter plot
        """
        if not hit_points:
            return go.Figure()
            
        # Convert to dataframe
        df_hits = pd.DataFrame(hit_points, columns=['x', 'y'])
        
        fig = px.scatter(
            df_hits,
            x='x',
            y='y',
            title=title,
            labels={'x': 'Horizontal Position', 'y': 'Vertical Position'}
        )
        
        # Add racket reference frame
        fig.update_layout(
            shapes=[
                # Racket outline
                dict(
                    type="rect",
                    x0=-1, y0=-1, x1=1, y1=1,
                    line=dict(color="rgba(0,0,0,0.3)"),
                    fillcolor="rgba(0,0,0,0)"
                ),
                # Horizontal center line
                dict(
                    type="line",
                    x0=-1, y0=0, x1=1, y1=0,
                    line=dict(color="rgba(0,0,0,0.3)", dash="dash")
                ),
                # Vertical center line
                dict(
                    type="line",
                    x0=0, y0=-1, x1=0, y1=1,
                    line=dict(color="rgba(0,0,0,0.3)", dash="dash")
                )
            ]
        )
        
        # Add sweet spot indication
        fig.add_shape(
            type="circle",
            x0=-0.3, y0=-0.3, x1=0.3, y1=0.3,
            line=dict(color="rgba(0,128,0,0.3)"),
            fillcolor="rgba(0,128,0,0.1)"
        )
        
        # Add annotations
        fig.add_annotation(
            x=0, y=1.1,
            text="Top",
            showarrow=False
        )
        fig.add_annotation(
            x=0, y=-1.1,
            text="Bottom",
            showarrow=False
        )
        
        return fig
