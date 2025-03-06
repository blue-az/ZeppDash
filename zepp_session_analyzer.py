import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional
from zepp_visualizer import ZeppVisualizer
import zepp_wrangle

class SessionAnalyzer:
    """Handles session-level analysis for Zepp tennis data"""
    
    def __init__(self, config):
        self.config = config
        self.visualizer = ZeppVisualizer()
    
    @staticmethod
    @st.cache_data
    def load_sessions(_config):
        """
        Load session data
        
        Parameters:
        _config: Configuration object
        
        Returns:
        pandas.DataFrame: Processed sessions data
        """
        # Load raw sessions data
        sessions_df = zepp_wrangle.load_sessions_data(_config.DB_PATH)
        
        # Process metrics
        metrics_df = zepp_wrangle.process_sessions_metrics(sessions_df)
        
        return metrics_df
    
    def render_session_analysis(self, session_id: str):
        """
        Render session analysis
        
        Parameters:
        session_id (str): ID of the session to analyze
        """
        # Load sessions data
        sessions_df = self.load_sessions(self.config)
        
        if sessions_df.empty:
            st.error("No sessions data available.")
            return
        
        # Get selected session
        if session_id not in sessions_df['session_id'].values:
            st.error(f"Session ID {session_id} not found.")
            return
            
        session = sessions_df[sessions_df['session_id'] == session_id].iloc[0]
        
        # Display session information
        st.header(f"Session Analysis: {session['formatted_datetime']}")
        
        # Display session metrics
        self.display_session_metrics(session)
        
        # Display shot distribution
        self.display_shot_distribution(session)
        
        # Display spin analysis
        self.display_spin_analysis(session)
        
        # Display performance radar chart
        self.display_performance_radar(session)
    
    def display_session_metrics(self, session):
        """
        Display session metrics
        
        Parameters:
        session (pandas.Series): Session data
        """
        # Create columns for metrics
        col1, col2, col3 = st.columns(3)
        
        # Column 1: Score metrics
        with col1:
            st.metric("Session Score", f"{session['session_score']:.1f}")
            st.metric("Consistency Score", f"{session['consistency_score']:.1f}")
        
        # Column 2: Speed metrics
        with col2:
            serve_speed_mph = session['serve_max_speed'] * self.config.SPEED_CONVERSION_FACTOR
            st.metric("Best Serve Speed", f"{serve_speed_mph:.1f} mph")
            
            forehand_speed_mph = session['forehand_max_speed'] * self.config.SPEED_CONVERSION_FACTOR
            st.metric("Best Forehand Speed", f"{forehand_speed_mph:.1f} mph")
        
        # Column 3: Activity metrics
        with col3:
            st.metric("Active Time", f"{session['active_time_min']:.1f} min")
            st.metric("Longest Rally", f"{session['longest_rally']}")
            st.metric("Total Shots", f"{session['total_shots']}")
    
    def display_shot_distribution(self, session):
        """
        Display shot distribution
        
        Parameters:
        session (pandas.Series): Session data
        """
        st.subheader("Shot Distribution")
        
        # Get shot counts
        shot_counts = {
            'Forehand': session['forehand_count'],
            'Backhand': session['backhand_count'],
            'Serve': session['serves_count'],
            'Volley': session['volley_count'],
            'Smash': session['smash_count']
        }
        
        # Create columns for charts
        col1, col2 = st.columns(2)
        
        # Column 1: Pie chart
        with col1:
            pie_fig = self.visualizer.create_shot_distribution_chart(shot_counts)
            st.plotly_chart(pie_fig, use_container_width=True)
        
        # Column 2: Bar chart
        with col2:
            # Create a dataframe for the bar chart
            shot_df = pd.DataFrame({
                'Shot Type': list(shot_counts.keys()),
                'Count': list(shot_counts.values())
            })
            shot_df = shot_df[shot_df['Count'] > 0]  # Filter out zero counts
            
            bar_fig = px.bar(
                shot_df,
                x='Shot Type',
                y='Count',
                color='Shot Type',
                title="Shot Counts",
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            st.plotly_chart(bar_fig, use_container_width=True)
    
    def display_spin_analysis(self, session):
        """
        Display spin analysis
        
        Parameters:
        session (pandas.Series): Session data
        """
        st.subheader("Spin Analysis")
        
        # Extract spin data
        session_json = session['session_json']
        spin_data = zepp_wrangle.extract_spin_analysis(session_json)
        
        if not spin_data:
            st.info("No spin data available for this session.")
            return
        
        # Create spin analysis chart
        spin_fig, percentages = self.visualizer.create_spin_analysis_chart(spin_data)
        st.plotly_chart(spin_fig, use_container_width=True)
        
        # Display percentages as a table
        if percentages is not None:
            st.markdown("### Spin Type Distribution (%)")
            st.dataframe(percentages.round(1), use_container_width=True)
    
    def display_performance_radar(self, session):
        """
        Display performance radar chart
        
        Parameters:
        session (pandas.Series): Session data
        """
        st.subheader("Performance Breakdown")
        
        # Create metrics for radar chart
        metrics = {
            'Consistency': session['consistency_score'],
            'Power': session['power_score'],
            'Intensity': session['intensity_score'],
            'Overall': session['session_score']
        }
        
        # Create radar chart
        radar_fig = self.visualizer.create_radar_chart(metrics)
        st.plotly_chart(radar_fig, use_container_width=True)
    
    def render_historical_analysis(self):
        """Render historical analysis of session data"""
        # Load sessions data
        sessions_df = self.load_sessions(self.config)
        
        if sessions_df.empty:
            st.error("No sessions data available for historical analysis.")
            return
        
        # Setup visualization options
        self.setup_historical_controls()
        
        # Display historical trends
        self.display_score_history(sessions_df)
        self.display_speed_history(sessions_df)
        self.display_activity_history(sessions_df)
        self.display_summary_statistics(sessions_df)
    
    def setup_historical_controls(self):
        """Setup controls for historical analysis"""
        st.sidebar.header("Visualization Options")
        
        self.viz_options = {
            'show_trendline': st.sidebar.checkbox("Show Trend Lines", value=True),
            'show_rolling_avg': st.sidebar.checkbox("Show Rolling Average", value=True),
            'rolling_window': st.sidebar.slider(
                "Rolling Average Window Size", 
                min_value=2, 
                max_value=20, 
                value=self.config.DEFAULT_ROLLING_WINDOW
            ),
            'min_date': st.sidebar.date_input("Start Date", value=None),
            'max_date': st.sidebar.date_input("End Date", value=None)
        }
    
    def filter_by_date_range(self, df):
        """
        Filter dataframe by date range
        
        Parameters:
        df (pandas.DataFrame): DataFrame to filter
        
        Returns:
        pandas.DataFrame: Filtered DataFrame
        """
        filtered_df = df.copy()
        
        # Apply date filters if provided
        if self.viz_options['min_date'] is not None:
            min_date = pd.Timestamp(self.viz_options['min_date'])
            filtered_df = filtered_df[filtered_df['start_time'].dt.date >= min_date.date()]
            
        if self.viz_options['max_date'] is not None:
            max_date = pd.Timestamp(self.viz_options['max_date'])
            filtered_df = filtered_df[filtered_df['start_time'].dt.date <= max_date.date()]
        
        return filtered_df
    
    def display_score_history(self, sessions_df):
        """
        Display session score history
        
        Parameters:
        sessions_df (pandas.DataFrame): Sessions data
        """
        st.subheader("Score History")
        
        # Filter by date range
        filtered_df = self.filter_by_date_range(sessions_df)
        
        if filtered_df.empty:
            st.info("No data available for the selected date range.")
            return
        
        # Sort by date
        filtered_df = filtered_df.sort_values('start_time')
        
        # Create figure
        fig = go.Figure()
        
        # Add session score trace
        fig.add_trace(go.Scatter(
            x=filtered_df['start_time'],
            y=filtered_df['session_score'],
            mode='markers+lines',
            name='Session Score',
            line=dict(color=self.config.PLOT_COLORS['overall'])
        ))
        
        # Add consistency score trace
        fig.add_trace(go.Scatter(
            x=filtered_df['start_time'],
            y=filtered_df['consistency_score'],
            mode='markers+lines',
            name='Consistency Score',
            line=dict(color=self.config.PLOT_COLORS['consistency'])
        ))
        
        # Add power score trace
        fig.add_trace(go.Scatter(
            x=filtered_df['start_time'],
            y=filtered_df['power_score'],
            mode='markers+lines',
            name='Power Score',
            line=dict(color=self.config.PLOT_COLORS['power'])
        ))
        
        # Add trend analysis
        for metric_name, y_data, color in [
            ('Session Score', filtered_df['session_score'], self.config.PLOT_COLORS['overall']),
            ('Consistency Score', filtered_df['consistency_score'], self.config.PLOT_COLORS['consistency']),
            ('Power Score', filtered_df['power_score'], self.config.PLOT_COLORS['power'])
        ]:
            self.visualizer.add_trend_analysis(
                fig,
                filtered_df['start_time'],
                y_data,
                metric_name,
                window_size=self.viz_options['rolling_window'],
                show_trendline=self.viz_options['show_trendline'],
                show_rolling_avg=self.viz_options['show_rolling_avg']
            )
        
        # Update layout
        fig.update_layout(
            title="Score History",
            xaxis_title="Date",
            yaxis_title="Score",
            hovermode='x unified',
            yaxis=dict(range=[0, 100])
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def display_speed_history(self, sessions_df):
        """
        Display speed history
        
        Parameters:
        sessions_df (pandas.DataFrame): Sessions data
        """
        st.subheader("Speed History")
        
        # Filter by date range
        filtered_df = self.filter_by_date_range(sessions_df)
        
        if filtered_df.empty:
            st.info("No data available for the selected date range.")
            return
        
        # Sort by date
        filtered_df = filtered_df.sort_values('start_time')
        
        # Convert speeds to mph
        filtered_df['serve_max_speed_mph'] = filtered_df['serve_max_speed'] * self.config.SPEED_CONVERSION_FACTOR
        filtered_df['forehand_max_speed_mph'] = filtered_df['forehand_max_speed'] * self.config.SPEED_CONVERSION_FACTOR
        filtered_df['backhand_max_speed_mph'] = filtered_df['backhand_max_speed'] * self.config.SPEED_CONVERSION_FACTOR
        
        # Create figure
        fig = go.Figure()
        
        # Add serve speed trace
        fig.add_trace(go.Scatter(
            x=filtered_df['start_time'],
            y=filtered_df['serve_max_speed_mph'],
            mode='markers+lines',
            name='Serve Max Speed',
            line=dict(color=self.config.PLOT_COLORS['serve'])
        ))
        
        # Add forehand speed trace
        fig.add_trace(go.Scatter(
            x=filtered_df['start_time'],
            y=filtered_df['forehand_max_speed_mph'],
            mode='markers+lines',
            name='Forehand Max Speed',
            line=dict(color=self.config.PLOT_COLORS['forehand'])
        ))
        
        # Add backhand speed trace
        fig.add_trace(go.Scatter(
            x=filtered_df['start_time'],
            y=filtered_df['backhand_max_speed_mph'],
            mode='markers+lines',
            name='Backhand Max Speed',
            line=dict(color=self.config.PLOT_COLORS['backhand'])
        ))
        
        # Add trend analysis
        for metric_name, y_data, color in [
            ('Serve Max Speed', filtered_df['serve_max_speed_mph'], self.config.PLOT_COLORS['serve']),
            ('Forehand Max Speed', filtered_df['forehand_max_speed_mph'], self.config.PLOT_COLORS['forehand']),
            ('Backhand Max Speed', filtered_df['backhand_max_speed_mph'], self.config.PLOT_COLORS['backhand'])
        ]:
            self.visualizer.add_trend_analysis(
                fig,
                filtered_df['start_time'],
                y_data,
                metric_name,
                window_size=self.viz_options['rolling_window'],
                show_trendline=self.viz_options['show_trendline'],
                show_rolling_avg=self.viz_options['show_rolling_avg'],
                remove_zeros=True,
                y_min=self.config.MIN_SPEED_THRESHOLD
            )
        
        # Update layout
        fig.update_layout(
            title="Speed History",
            xaxis_title="Date",
            yaxis_title="Speed (mph)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def display_activity_history(self, sessions_df):
        """
        Display activity history
        
        Parameters:
        sessions_df (pandas.DataFrame): Sessions data
        """
        st.subheader("Activity History")
        
        # Filter by date range
        filtered_df = self.filter_by_date_range(sessions_df)
        
        if filtered_df.empty:
            st.info("No data available for the selected date range.")
            return
        
        # Sort by date
        filtered_df = filtered_df.sort_values('start_time')
        
        # Create figure
        fig = go.Figure()
        
        # Add total shots trace
        fig.add_trace(go.Scatter(
            x=filtered_df['start_time'],
            y=filtered_df['total_shots'],
            mode='markers+lines',
            name='Total Shots',
            line=dict(color=self.config.PLOT_COLORS['overall'])
        ))
        
        # Add active time trace
        fig.add_trace(go.Scatter(
            x=filtered_df['start_time'],
            y=filtered_df['active_time_min'],
            mode='markers+lines',
            name='Active Time (min)',
            line=dict(color=self.config.PLOT_COLORS['intensity']),
            yaxis='y2'
        ))
        
        # Add trend analysis
        self.visualizer.add_trend_analysis(
            fig,
            filtered_df['start_time'],
            filtered_df['total_shots'],
            'Total Shots',
            window_size=self.viz_options['rolling_window'],
            show_trendline=self.viz_options['show_trendline'],
            show_rolling_avg=self.viz_options['show_rolling_avg']
        )
        
        self.visualizer.add_trend_analysis(
            fig,
            filtered_df['start_time'],
            filtered_df['active_time_min'],
            'Active Time (min)',
            window_size=self.viz_options['rolling_window'],
            show_trendline=self.viz_options['show_trendline'],
            show_rolling_avg=self.viz_options['show_rolling_avg']
        )
        
        # Update layout with dual y-axes - using updated property names
        fig.update_layout(
            title="Activity History",
            xaxis_title="Date",
            yaxis=dict(
                title="Total Shots",
                title_font=dict(color=self.config.PLOT_COLORS['overall'])
            ),
            yaxis2=dict(
                title="Active Time (min)",
                title_font=dict(color=self.config.PLOT_COLORS['intensity']),
                overlaying="y",
                side="right"
            ),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def display_summary_statistics(self, sessions_df):
        """
        Display summary statistics
        
        Parameters:
        sessions_df (pandas.DataFrame): Sessions data
        """
        st.subheader("Summary Statistics")
        
        # Filter by date range
        filtered_df = self.filter_by_date_range(sessions_df)
        
        if filtered_df.empty:
            st.info("No data available for the selected date range.")
            return
        
        # Create columns for metrics
        col1, col2, col3 = st.columns(3)
        
        # Column 1: Score metrics
        with col1:
            st.metric(
                "Average Session Score",
                f"{filtered_df['session_score'].mean():.1f}",
                f"Best: {filtered_df['session_score'].max():.1f}"
            )
            st.metric(
                "Average Consistency Score",
                f"{filtered_df['consistency_score'].mean():.1f}",
                f"Best: {filtered_df['consistency_score'].max():.1f}"
            )
        
        # Column 2: Speed metrics
        with col2:
            max_serve_speed = filtered_df['serve_max_speed'].max() * self.config.SPEED_CONVERSION_FACTOR
            max_forehand_speed = filtered_df['forehand_max_speed'].max() * self.config.SPEED_CONVERSION_FACTOR
            max_backhand_speed = filtered_df['backhand_max_speed'].max() * self.config.SPEED_CONVERSION_FACTOR
            
            st.metric(
                "Best Speeds (mph)",
                f"Serve: {max_serve_speed:.1f}",
                f"FH: {max_forehand_speed:.1f} / BH: {max_backhand_speed:.1f}"
            )
        
        # Column 3: Activity metrics
        with col3:
            st.metric(
                "Average Total Shots",
                f"{filtered_df['total_shots'].mean():.1f}",
                f"Max: {filtered_df['total_shots'].max()}"
            )
            st.metric(
                "Average Active Time",
                f"{filtered_df['active_time_min'].mean():.1f} min",
                f"Max: {filtered_df['active_time_min'].max():.1f} min"
            )
        
        # Display monthly averages - with error handling
        st.markdown("### Monthly Averages")
        
        try:
            # Make sure we have numeric data for these columns
            numeric_columns = ['session_score', 'consistency_score', 'power_score']
            
            # Ensure all columns are numeric
            for col in numeric_columns:
                filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce')
            
            # Set datetime index for resampling
            monthly_df = filtered_df.copy()
            monthly_df['month_date'] = pd.to_datetime(monthly_df['start_time']).dt.to_period('M').dt.to_timestamp()
            
            # Group by month and calculate mean
            monthly_data = []
            for month, group in monthly_df.groupby(monthly_df['month_date']):
                month_str = month.strftime('%b %Y')
                monthly_data.append({
                    'month': month_str,
                    'session_score': group['session_score'].mean(),
                    'consistency_score': group['consistency_score'].mean(),
                    'power_score': group['power_score'].mean()
                })
            
            # Convert to DataFrame
            monthly_avg = pd.DataFrame(monthly_data)
            
            if not monthly_avg.empty:
                # Create a more flexible chart
                fig = px.bar(
                    monthly_avg,
                    x='month',
                    y=['session_score', 'consistency_score', 'power_score'],
                    title="Monthly Average Scores",
                    labels={'value': 'Score', 'variable': 'Metric', 'month': 'Month'},
                    barmode='group'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No monthly data available for the selected range.")
        except Exception:
            # Try a simpler approach - just group by month string
            try:
                filtered_df['month'] = filtered_df['start_time'].dt.strftime('%b %Y')
                simple_monthly = filtered_df.groupby('month')['session_score'].mean().reset_index()
                
                if not simple_monthly.empty:
                    fig = px.bar(
                        simple_monthly,
                        x='month',
                        y='session_score',
                        title="Monthly Average Session Score",
                        labels={'session_score': 'Score', 'month': 'Month'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.info("No monthly data available for the selected range.")
