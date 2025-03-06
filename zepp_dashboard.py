import streamlit as st
import pandas as pd
from config import Config
from zepp_session_analyzer import SessionAnalyzer
from zepp_shot_analyzer import ShotAnalyzer
import zepp_wrangle

class ZeppDashboard:
    """Main dashboard class for Zepp tennis data analysis"""
    
    def __init__(self):
        """Initialize the dashboard"""
        # Configure Streamlit page
        st.set_page_config(layout="wide", page_title="Zepp Tennis Dashboard")
        
        # Initialize configuration
        self.config = Config()
        
        # Initialize analyzers
        self.session_analyzer = SessionAnalyzer(self.config)
        self.shot_analyzer = ShotAnalyzer(self.config)
        
        # Initialize view
        self.initialize_view()
    
    def initialize_view(self):
        """Initialize the dashboard view"""
        st.title("Zepp Tennis Dashboard")
        
        # Main view selection
        self.view_mode = st.sidebar.radio(
            "Select View",
            ["Session Analysis", "Historical Analysis", "Shot Analysis", "Hit Points Analysis"],
            key="view_mode_radio"
        )
        
        # Load sessions data
        self.sessions_df = self.session_analyzer.load_sessions(self.config)
        
        if self.sessions_df.empty:
            st.error("No sessions data found. Please check your database configuration.")
            return
        
        # Render selected view
        if self.view_mode == "Session Analysis":
            self.render_session_analysis()
        elif self.view_mode == "Historical Analysis":
            self.render_historical_analysis()
        elif self.view_mode == "Shot Analysis":
            self.render_shot_analysis()
        else:  # Hit Points Analysis
            self.render_hit_points_analysis()
    
    def get_session_selector(self) -> str:
        """
        Get selected session ID
        
        Returns:
        str: Selected session ID
        """
        return st.sidebar.selectbox(
            "Select Session",
            self.sessions_df['session_id'].unique(),
            format_func=lambda x: f"ID: {x} - {self.sessions_df[self.sessions_df['session_id'] == x]['formatted_datetime'].iloc[0]}",
            key="session_selector"
        )
    
    def render_session_analysis(self):
        """Render session analysis view"""
        session_id = self.get_session_selector()
        if session_id:
            self.session_analyzer.render_session_analysis(session_id)
    
    def render_historical_analysis(self):
        """Render historical analysis view"""
        self.session_analyzer.render_historical_analysis()
    
    def render_shot_analysis(self):
        """Render shot analysis view"""
        session_id = self.get_session_selector()
        if session_id:
            session = self.sessions_df[self.sessions_df['session_id'] == session_id].iloc[0]
            session_datetime = session['start_time']
            
            # Display session info
            st.subheader(f"Session: {session['formatted_datetime']}")
            st.markdown(f"Total Shots: {session['total_shots']}")
            
            # Render shot analysis - use the same DB path as sessions since shots are extracted from session data
            self.shot_analyzer.render_shot_analysis(session_id, session_datetime)
    
    def render_hit_points_analysis(self):
        """Render hit points analysis view"""
        session_id = self.get_session_selector()
        if session_id:
            session = self.sessions_df[self.sessions_df['session_id'] == session_id].iloc[0]
            session_datetime = session['start_time']
            session_data = session['session_json']
            
            # Display session info
            st.subheader(f"Session: {session['formatted_datetime']}")
            
            # Render hit points analysis
            self.shot_analyzer.render_hit_points_analysis(session_id, session_datetime, session_data)

def main():
    """Main entry point"""
    ZeppDashboard()

if __name__ == "__main__":
    main()
