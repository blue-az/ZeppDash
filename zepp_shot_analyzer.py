import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import zepp_wrangle
from zepp_visualizer import ZeppVisualizer

class ShotAnalyzer:
    """Handles shot-by-shot analysis for Zepp tennis data"""
    
    def __init__(self, config):
        self.config = config
        self.visualizer = ZeppVisualizer()
    
    @staticmethod
    @st.cache_data
    def load_shot_data(_config, session_id: str, session_datetime: datetime) -> pd.DataFrame:
        """
        Load shot data for a specific session
        
        Parameters:
        _config: Configuration object
        session_id (str): ID of the session
        session_datetime (datetime): Datetime of the session
        
        Returns:
        pandas.DataFrame: Shot data for the session
        """
        # First try loading shot data directly
        df = zepp_wrangle.wrangle_shots(_config.DB_PATH)
        
        # Filter for the specific session if the ID is available in the data
        if 'session_id' in df.columns:
            df = df[df['session_id'] == session_id]
            
            if not df.empty:
                return df
        
        # If we don't have session_id in the data or no matching data was found,
        # try filtering by time window instead
        # Calculate session window (3 hours around the session datetime)
        session_start = session_datetime - timedelta(hours=1.5)
        session_end = session_datetime + timedelta(hours=1.5)
        
        # Filter for the session window
        df = df[(df['time'] >= session_start) & (df['time'] <= session_end)]
        
        return df
    
    def render_shot_analysis(self, session_id: str, session_datetime: datetime):
        """
        Render shot analysis
        
        Parameters:
        session_id (str): ID of the session
        session_datetime (datetime): Datetime of the session
        """
        # Load shot data
        df = self.load_shot_data(self.config, session_id, session_datetime)
        
        if df.empty:
            st.warning("No shot data found for this session.")
            return
        
        # Setup filters
        self._setup_filters(df)
        
        # Apply filters
        filtered_df = self._apply_filters(df)
        
        if filtered_df.empty:
            st.warning("No shots match the selected filters.")
            return
        
        # Render visualizations
        self._render_scatter_plot(filtered_df)
        self._render_histogram(filtered_df)
        self._render_line_plot(filtered_df)
        self._render_correlation_heatmap(filtered_df)
        self._render_summary_stats(filtered_df)
    
    def _setup_filters(self, df: pd.DataFrame):
        """
        Setup sidebar filters
        
        Parameters:
        df (pandas.DataFrame): Shot data
        """
        st.sidebar.header("Shot Analysis Controls")
        
        # Shot type selection
        self.selected_types = st.sidebar.multiselect(
            "Select Shot Types",
            df['stroke'].unique(),
            default=df['stroke'].unique()
        )
        
        # Stroke category selection
        self.stroke_categories = st.sidebar.multiselect(
            "Select Stroke Categories",
            ['Serve', 'Forehand', 'Backhand', 'Other'],
            default=['Serve', 'Forehand', 'Backhand', 'Other']
        )
        
        # Swing type selection
        self.swing_types = st.sidebar.multiselect(
            "Select Swing Types",
            df['swing_type'].unique(),
            default=df['swing_type'].unique()
        )
    
    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply selected filters to the dataframe
        
        Parameters:
        df (pandas.DataFrame): Shot data
        
        Returns:
        pandas.DataFrame: Filtered shot data
        """
        return df[
            (df['stroke'].isin(self.selected_types)) &
            (df['stroke_category'].isin(self.stroke_categories)) &
            (df['swing_type'].isin(self.swing_types))
        ]
    
    def _render_scatter_plot(self, df: pd.DataFrame):
        """
        Render scatter plot visualization
        
        Parameters:
        df (pandas.DataFrame): Filtered shot data
        """
        st.header("Shot Distribution")
        
        # Available metrics for axes
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Axis selection
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox("X-axis metric", numeric_cols, index=numeric_cols.index('power') if 'power' in numeric_cols else 0)
        with col2:
            y_axis = st.selectbox("Y-axis metric", numeric_cols, index=numeric_cols.index('racket_speed') if 'racket_speed' in numeric_cols else 0)
        
        # Jitter controls
        col3, col4 = st.columns(2)
        with col3:
            add_jitter = st.checkbox("Add Jitter", value=False)
        with col4:
            jitter_amount = st.slider(
                "Jitter Amount",
                0.0, 1.0, 0.5,
                disabled=not add_jitter
            )
        
        # Apply jitter if selected
        plot_df = df.copy()
        if add_jitter:
            for axis in [x_axis, y_axis]:
                if not isinstance(plot_df[axis], pd.Timestamp):
                    std = plot_df[axis].std() * jitter_amount * 0.1
                    plot_df[axis] += np.random.normal(0, std, len(plot_df))
        
        # Create scatter plot
        fig = px.scatter(
            plot_df,
            x=x_axis,
            y=y_axis,
            color="stroke_category",
            title=f"{x_axis} vs {y_axis}",
            hover_data=['stroke', 'swing_type', 'timestamp']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_histogram(self, df: pd.DataFrame):
        """
        Render histogram visualization
        
        Parameters:
        df (pandas.DataFrame): Filtered shot data
        """
        st.header("Shot Distribution Histogram")
        
        # Available metrics
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        col1, col2 = st.columns(2)
        with col1:
            metric = st.selectbox("Select metric", numeric_cols, index=numeric_cols.index('power') if 'power' in numeric_cols else 0)
        with col2:
            num_bins = st.slider("Number of bins", 5, 100, 20)
        
        fig = px.histogram(
            df,
            x=metric,
            nbins=num_bins,
            color="stroke_category",
            title=f"Distribution of {metric}"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_line_plot(self, df: pd.DataFrame):
        """
        Render line plot visualization
        
        Parameters:
        df (pandas.DataFrame): Filtered shot data
        """
        st.header("Shot Progression")
        
        # Available metrics
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        col1, col2 = st.columns(2)
        with col1:
            metric = st.selectbox(
                "Select metric for progression",
                numeric_cols,
                index=numeric_cols.index('power') if 'power' in numeric_cols else 0,
                key='line_metric'
            )
        with col2:
            add_average = st.checkbox("Show Moving Average")
        
        fig = go.Figure()
        
        # Add individual shots
        for category in df['stroke_category'].unique():
            category_data = df[df['stroke_category'] == category]
            fig.add_trace(go.Scatter(
                x=category_data['time'],
                y=category_data[metric],
                mode='markers',
                name=category,
                opacity=0.7
            ))
        
        # Add moving average if selected
        if add_average:
            window_size = st.slider("Moving Average Window", 5, 50, 20)
            df_sorted = df.sort_values('time')
            fig.add_trace(go.Scatter(
                x=df_sorted['time'],
                y=df_sorted[metric].rolling(window=window_size).mean(),
                mode='lines',
                name=f'{window_size}-shot Moving Average',
                line=dict(color='black', width=2)
            ))
        
        fig.update_layout(
            title=f"{metric} Progression",
            xaxis_title="Time",
            yaxis_title=metric
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_correlation_heatmap(self, df: pd.DataFrame):
        """
        Render correlation heatmap
        
        Parameters:
        df (pandas.DataFrame): Filtered shot data
        """
        st.header("Metric Correlations")
        
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=['number'])
        
        # Calculate correlation matrix
        corr = numeric_df.corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            ax=ax
        )
        st.pyplot(fig)
    
    def _render_summary_stats(self, df: pd.DataFrame):
        """
        Render summary statistics
        
        Parameters:
        df (pandas.DataFrame): Filtered shot data
        """
        st.header("Summary Statistics")
        
        # Calculate summary statistics
        summary = df.describe()
        
        # Display in a more readable format
        st.dataframe(summary.style.format("{:.2f}"), use_container_width=True)
        
        # Display statistics by stroke category
        st.subheader("Statistics by Stroke Category")
        
        # Create selectbox for metric
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_metric = st.selectbox(
            "Select metric for comparison",
            numeric_cols,
            index=numeric_cols.index('power') if 'power' in numeric_cols else 0,
            key='summary_metric'
        )
        
        # Calculate statistics by stroke category
        category_stats = df.groupby('stroke_category')[selected_metric].agg(['mean', 'std', 'min', 'max', 'count'])
        
        # Create bar chart
        fig = px.bar(
            category_stats,
            y=category_stats.index,
            x='mean',
            error_x='std',
            title=f"Average {selected_metric} by Stroke Category",
            labels={'y': 'Stroke Category', 'mean': f'Average {selected_metric}'},
            orientation='h'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display statistics by swing type
        st.subheader("Statistics by Swing Type")
        
        # Calculate statistics by swing type
        swing_stats = df.groupby('swing_type')[selected_metric].agg(['mean', 'std', 'min', 'max', 'count'])
        
        # Create bar chart
        fig = px.bar(
            swing_stats,
            y=swing_stats.index,
            x='mean',
            error_x='std',
            title=f"Average {selected_metric} by Swing Type",
            labels={'y': 'Swing Type', 'mean': f'Average {selected_metric}'},
            orientation='h'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def render_hit_points_analysis(self, session_id: str, session_datetime: datetime, session_data):
        """
        Render hit points analysis
        
        Parameters:
        session_id (str): ID of the session
        session_datetime (datetime): Datetime of the session
        session_data (dict): Session data with hit points
        """
        st.header("Hit Points Analysis")
        
        # Setup for hit points visualization
        shot_type = st.selectbox(
            "Select Shot Type",
            ["forehand", "backhand", "serve"]
        )
        
        if shot_type in ["forehand", "backhand"]:
            spin_type = st.selectbox(
                "Select Spin Type",
                ["flat", "slice", "topspin"]
            )
            hit_points = zepp_wrangle.extract_hit_points(session_data, shot_type, spin_type)
            title = f"Hit Points Distribution - {shot_type.capitalize()} {spin_type.capitalize()}"
        else:
            hit_points = zepp_wrangle.extract_hit_points(session_data, shot_type)
            title = f"Hit Points Distribution - {shot_type.capitalize()}"
        
        if not hit_points:
            st.info(f"No hit points data available for {shot_type}.")
            return
        
        # Create hit points chart
        hit_points_fig = self.visualizer.create_hit_points_chart(hit_points, title)
        st.plotly_chart(hit_points_fig, use_container_width=True)
        
        # Display hit points statistics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Hit Points", len(hit_points))
        
        with col2:
            avg_height = sum(point[1] for point in hit_points) / len(hit_points)
            st.metric("Average Height", f"{avg_height:.2f}")
        
        # Advanced analysis (centroid, dispersion)
        if len(hit_points) > 1:
            # Calculate centroid
            centroid_x = sum(point[0] for point in hit_points) / len(hit_points)
            centroid_y = sum(point[1] for point in hit_points) / len(hit_points)
            
            # Calculate dispersion (standard deviation from centroid)
            dispersion_x = np.std([point[0] for point in hit_points])
            dispersion_y = np.std([point[1] for point in hit_points])
            
            # Display advanced statistics
            st.subheader("Advanced Hit Point Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Centroid X", f"{centroid_x:.2f}")
                st.metric("Dispersion X", f"{dispersion_x:.2f}")
            
            with col2:
                st.metric("Centroid Y", f"{centroid_y:.2f}")
                st.metric("Dispersion Y", f"{dispersion_y:.2f}")
            
            # Calculate sweet spot percentage (hits within sweet spot)
            sweet_spot_hits = sum(1 for point in hit_points if (point[0]**2 + point[1]**2) <= 0.3**2)
            sweet_spot_percentage = (sweet_spot_hits / len(hit_points)) * 100
            
            st.metric("Sweet Spot Hits", f"{sweet_spot_hits} ({sweet_spot_percentage:.1f}%)")
