# Zepp Tennis Dashboard

A comprehensive Streamlit dashboard for analyzing Zepp tennis sensor data, combining session-level analytics and shot-by-shot analysis.

## Features

### 1. Session Analysis
- Session metrics display (score, speeds, activity levels)
- Shot distribution visualization
- Spin analysis with detailed breakdowns
- Performance radar chart showing consistency, power, and intensity

### 2. Historical Analysis
- Score progression over time
- Speed development tracking
- Activity level trends
- Customizable visualizations with:
  - Configurable rolling averages
  - Optional trend lines
  - Date range filtering
- Summary statistics and monthly averages

### 3. Shot Analysis
- Detailed shot-by-shot analysis for each session
- Interactive scatter plots
- Shot distribution histograms
- Shot progression analysis
- Correlation heatmaps
- Customizable filters for shot types and swing types

### 4. Hit Points Analysis
- Visualization of racket impact points
- Sweet spot percentage calculation
- Centroid and dispersion analysis
- Comparison across different shot and spin types

## Project Structure
```
zepp_dashboard/
├── config.py               # Configuration settings
├── zepp_dashboard.py       # Main dashboard implementation
├── zepp_wrangle.py         # Data loading and preprocessing
├── zepp_visualizer.py      # Visualization utilities
├── zepp_session_analyzer.py # Session-level analysis
├── zepp_shot_analyzer.py   # Shot-by-shot analysis
├── main.py                 # Application entry point
├── streamlit_app.py        # Streamlit-specific entry point
└── requirements.txt        # Package dependencies
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd zepp_dashboard
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Configuration

1. Database Settings:
   - Update database paths in `config.py`:
     - `DB_PATH`: Path to session database (ztennis.db)
     - The application uses a single database for both session and shot data

2. Timezone Configuration:
   - Default timezone is 'America/Phoenix'
   - Modify in `config.py` if needed

3. Analysis Settings:
   - Speed conversion factor (m/s to mph): 2.25
   - Default rolling window size: 5
   - Minimum speed threshold: 50.0
   - Customizable plot colors

## Usage

1. Start the dashboard:
```bash
streamlit run streamlit_app.py
```

2. Access the dashboard in your web browser (default: http://localhost:8501)

3. Navigate between views:
   - Session Analysis: View individual session details
   - Historical Analysis: Track progress over time 
   - Shot Analysis: Analyze shot-by-shot data for specific sessions
   - Hit Points Analysis: Visualize racket impact points

4. Using Historical Analysis:
   - Toggle trend lines and rolling averages
   - Adjust rolling average window size
   - Filter by date range
   - View summary statistics and monthly trends

5. Using Shot Analysis:
   - Select a specific session
   - Filter by shot types and swing types
   - Analyze shot distributions and metrics
   - View shot progression within the session
   - Explore correlations between metrics

## Data Structure and Processing

### Session Data
The dashboard extracts session data from the Zepp database using the `session_report` table. Each session contains:
- Timestamp information
- Overall session score
- Shot counts by type (forehand, backhand, serve)
- Speed data
- Activity metrics
- JSON data with detailed analysis

### Shot Data
Shot data is extracted from session reports and organized into a structured format including:
- Shot timestamp
- Shot type and spin
- Speeds and metrics
- Impact points

## Key Features

### Robust Data Handling
- Automatically detects available tables in the database
- Adapts to different database structures
- Handles missing or incomplete data gracefully
- Silent error handling for smooth user experience

### Customizable Visualizations
- All charts include filter options
- Historical analysis supports date range selection
- Multiple chart types for different data views
- Color-coded visualizations for easy interpretation

### Performance Optimizations
- Uses Streamlit caching for efficient data loading
- Optimized data processing to minimize memory use
- Selective data loading for specific sessions

## Troubleshooting

1. Database Connection Issues:
   - Verify database path in config.py
   - Check file permissions
   - Ensure SQLite is properly installed

2. Visualization Issues:
   - Check for missing data in required columns
   - Verify data types match expected formats
   - Confirm timezone settings if time-based issues occur

3. Shot Analysis Issues:
   - Ensure session timestamps align with shot data
   - Check filter combinations for empty result sets

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Credits

Developed by blue-az
