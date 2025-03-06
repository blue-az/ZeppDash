import sqlite3
import pandas as pd
import pytz
import json
from typing import Dict, List, Optional, Tuple
import streamlit as st

def wrangle_shots(db_path, start_date=None, end_date=None):
    """
    Load and process shot-by-shot data from Zepp database
    
    Parameters:
    db_path (str): Path to the Zepp shot database
    start_date (datetime, optional): Start date for filtering
    end_date (datetime, optional): End date for filtering
    
    Returns:
    pandas.DataFrame: Processed shot data
    """
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        
        # Find all tables in the database
        tables_query = "SELECT name FROM sqlite_master WHERE type='table'"
        tables = pd.read_sql(tables_query, conn)['name'].tolist()
        
        # First, check if this is a database with direct session_report access
        # This approach is used in ZeppHist.py
        if 'session_report' in tables:
            # Get the session report data
            query = "SELECT _id, report FROM session_report"
            session_df = pd.read_sql_query(query, conn)
            
            # Close the database connection
            conn.close()
            
            # Extract shots from the report JSON
            all_shots = []
            timestamps = []
            
            for _, row in session_df.iterrows():
                try:
                    if pd.isna(row['report']) or not row['report']:
                        continue
                        
                    session_json = json.loads(row['report'])
                    if not all(key in session_json for key in ['session']):
                        continue
                    
                    # Get session information
                    session_data = session_json['session']
                    session_id = row['_id']
                    
                    # Get session timestamp
                    session_time = pd.Timestamp.now()  # Fallback
                    if 'start_time' in session_data:
                        session_time = pd.to_datetime(session_data['start_time'], unit='ms')
                    
                    # Process forehand data
                    for spin_type in ['flat', 'slice', 'topspin']:
                        swings_count = session_data['swings']['forehand'].get(f'{spin_type}_swings', 0)
                        avg_speed = session_data['swings']['forehand'].get(f'{spin_type}_average_speed', 0)
                        max_speed = session_data['swings']['forehand'].get(f'{spin_type}_max_speed', 0)
                        
                        for i in range(swings_count):
                            shot = {
                                'session_id': session_id,
                                'time': session_time,
                                'swing_type': spin_type.upper(),
                                'stroke': f"{spin_type.upper()}FH",
                                'racket_speed': avg_speed,  # Using average as approximation
                                'max_speed': max_speed
                            }
                            all_shots.append(shot)
                            timestamps.append(session_time)
                    
                    # Process backhand data
                    for spin_type in ['flat', 'slice', 'topspin']:
                        swings_count = session_data['swings']['backhand'].get(f'{spin_type}_swings', 0)
                        avg_speed = session_data['swings']['backhand'].get(f'{spin_type}_average_speed', 0)
                        max_speed = session_data['swings']['backhand'].get(f'{spin_type}_max_speed', 0)
                        
                        for i in range(swings_count):
                            shot = {
                                'session_id': session_id,
                                'time': session_time,
                                'swing_type': spin_type.upper(),
                                'stroke': f"{spin_type.upper()}BH",
                                'racket_speed': avg_speed,  # Using average as approximation
                                'max_speed': max_speed
                            }
                            all_shots.append(shot)
                            timestamps.append(session_time)
                    
                    # Process serve data
                    swings_count = session_data['swings']['serve'].get('serve_swings', 0)
                    avg_speed = session_data['swings']['serve'].get('serve_average_speed', 0)
                    max_speed = session_data['swings']['serve'].get('serve_max_speed', 0)
                    
                    for i in range(swings_count):
                        shot = {
                            'session_id': session_id,
                            'time': session_time,
                            'swing_type': 'SERVE',
                            'stroke': 'SERVEFH',  # Default to forehand
                            'racket_speed': avg_speed,
                            'max_speed': max_speed
                        }
                        all_shots.append(shot)
                        timestamps.append(session_time)
                    
                except Exception:
                    continue
            
            # Create DataFrame from extracted shots
            if all_shots:
                df = pd.DataFrame(all_shots)
                
                # Create a proper time index
                if timestamps:
                    time_index = pd.DatetimeIndex(timestamps)
                    df['time'] = time_index
                
                # Add stroke_category
                def categorize_stroke(row):
                    stroke_lower = str(row['stroke']).lower()
                    if 'serve' in stroke_lower:
                        return 'Serve'
                    elif 'fh' in stroke_lower:
                        return 'Forehand'
                    elif 'bh' in stroke_lower:
                        return 'Backhand'
                    else:
                        return 'Other'
                
                df['stroke_category'] = df.apply(categorize_stroke, axis=1)
                
                # Add timestamp column
                df['timestamp'] = df['time'].dt.strftime('%m-%d-%Y %I:%M:%S %p')
                
                # Filter by date range if provided
                if start_date and end_date:
                    df = df[(df['time'] >= start_date) & (df['time'] <= end_date)]
                
                return df
            else:
                return pd.DataFrame(columns=['time', 'stroke', 'stroke_category', 'swing_type'])
        
        # If we get here, we need to try other approaches
        # Try to detect the right table - check common shot tables
        candidate_tables = [
            table for table in tables if any(
                keyword in table.lower() for keyword in 
                ['swing', 'motion', 'shot', 'stroke']
            )
        ]
        
        if not candidate_tables:
            # If no obvious shot tables, use session data tables
            candidate_tables = [
                table for table in tables if any(
                    keyword in table.lower() for keyword in 
                    ['session', 'activity']
                )
            ]
        
        # If still no candidate tables, try all tables
        if not candidate_tables:
            candidate_tables = tables
        
        # Try each candidate table
        for shot_table in candidate_tables:
            try:
                # Get column information for this table
                cols_query = f"PRAGMA table_info({shot_table})"
                cols_df = pd.read_sql(cols_query, conn)
                columns = cols_df['name'].tolist()
                
                # Common timestamp column names
                time_columns = ['time', 'l_id', 'timestamp', 'date', 'datetime']
                timestamp_column = next((col for col in time_columns if col in columns), None)
                
                if timestamp_column:
                    # Read sample data from the table
                    query = f"SELECT * FROM {shot_table} LIMIT 10"
                    sample_df = pd.read_sql(query, conn)
                    
                    # If we have at least one row, this might be a usable table
                    if not sample_df.empty:
                        # Read full data
                        query = f"SELECT * FROM {shot_table}"
                        df = pd.read_sql(query, conn)
                        
                        # Process timestamps based on the detected column
                        if timestamp_column == 'l_id':
                            # Zepp classic format
                            df['time'] = pd.to_datetime(df['l_id'], unit='ms')
                            time_column = 'time'
                        elif timestamp_column == 'time':
                            # Check time format (could be seconds or milliseconds)
                            if df['time'].max() > 1e12:  # Likely milliseconds
                                df['time'] = pd.to_datetime(df['time'], unit='ms')
                            else:  # Likely seconds or other format
                                df['time'] = pd.to_datetime(df['time']/10000, unit='s')
                            time_column = 'time'
                        else:
                            # Try to convert whatever timestamp we found
                            try:
                                df['time'] = pd.to_datetime(df[timestamp_column])
                                time_column = 'time'
                            except:
                                time_column = timestamp_column  # Use as-is
                        
                        # Convert to Arizona timezone if possible
                        try:
                            az_timezone = pytz.timezone('America/Phoenix')
                            df['time'] = df['time'].dt.tz_localize('UTC').dt.tz_convert(az_timezone)
                        except:
                            pass
                        
                        # Try to identify shot type information
                        shot_type_columns = ['type', 'stroke', 'stroke_type', 'swing_type']
                        shot_type_column = next((col for col in shot_type_columns if col in columns), None)
                        
                        # Add stroke field based on available information
                        if 'stroke' not in df.columns:
                            if shot_type_column:
                                df['stroke'] = df[shot_type_column]
                            else:
                                # Create a default stroke column
                                df['stroke'] = 'UNKNOWN'
                        
                        # Add stroke_category field
                        if 'stroke_category' not in df.columns:
                            def categorize_stroke(row):
                                if shot_type_column:
                                    value = str(row[shot_type_column]).lower()
                                    if any(serve in value for serve in ['serve', 'service']):
                                        return 'Serve'
                                    elif any(fh in value for fh in ['forehand', 'fh']):
                                        return 'Forehand'
                                    elif any(bh in value for bh in ['backhand', 'bh']):
                                        return 'Backhand'
                                    else:
                                        return 'Other'
                                else:
                                    return 'Unknown'
                            
                            df['stroke_category'] = df.apply(categorize_stroke, axis=1)
                        
                        # Filter by date range if provided
                        if start_date and end_date and time_column in df.columns:
                            df = df[(df[time_column] >= start_date) & (df[time_column] <= end_date)]
                        
                        # Add timestamp column for display
                        if 'timestamp' not in df.columns and time_column in df.columns:
                            try:
                                df['timestamp'] = df[time_column].dt.strftime('%m-%d-%Y %I:%M:%S.%f %p')
                            except:
                                df['timestamp'] = df[time_column]
                        
                        # Add swing_type if not present
                        if 'swing_type' not in df.columns:
                            df['swing_type'] = df['stroke'] if 'stroke' in df.columns else 'UNKNOWN'
                        
                        conn.close()
                        return df
            except Exception:
                continue
        
        # If we got here, no suitable table was found
        raise ValueError("Could not find a valid shots table in the database")
        
    except Exception:
        if 'conn' in locals():
            conn.close()
        # Return empty DataFrame with necessary columns
        return pd.DataFrame(columns=['time', 'stroke', 'stroke_category', 'swing_type'])

@st.cache_data
def load_sessions_data(db_path):
    """
    Load session data from Zepp database
    
    Parameters:
    db_path (str): Path to the Zepp session database
    
    Returns:
    pandas.DataFrame: Processed session data
    """
    try:
        conn = sqlite3.connect(db_path)
        
        # Find all tables in the database
        tables_query = "SELECT name FROM sqlite_master WHERE type='table'"
        tables = pd.read_sql(tables_query, conn)['name'].tolist()
        
        # Try to find the session table
        session_table = None
        
        # Check for session_report table first (from ZeppHist)
        if 'session_report' in tables:
            session_table = 'session_report'
            cols_query = f"PRAGMA table_info({session_table})"
            cols_df = pd.read_sql(cols_query, conn)
            columns = cols_df['name'].tolist()
            
            # Query session data
            query = "SELECT _id, session_id, start_time, end_time, session_score, report FROM session_report"
            df = pd.read_sql_query(query, conn)
        
        # Check for tb_activities table (from BabDash)
        elif 'tb_activities' in tables:
            session_table = 'tb_activities'
            cols_query = f"PRAGMA table_info({session_table})"
            cols_df = pd.read_sql(cols_query, conn)
            columns = cols_df['name'].tolist()
            
            # Query session data
            query = "SELECT * FROM tb_activities"
            df = pd.read_sql_query(query, conn)
            
            # Map column names if needed
            if '_id' not in df.columns and 'id' in df.columns:
                df = df.rename(columns={'id': '_id'})
                
            # Add session_score if it doesn't exist
            if 'session_score' not in df.columns and 'piq_score' in df.columns:
                df['session_score'] = df['piq_score']
                
            # Add report column placeholder if not available
            if 'report' not in df.columns:
                df['report'] = None
        
        # If no suitable table found
        else:
            # Use the first table
            if tables:
                session_table = tables[0]
                query = f"SELECT * FROM {session_table}"
                df = pd.read_sql_query(query, conn)
            else:
                raise ValueError("No tables found in the database")
        
        # Process timestamps - try to handle different timestamp formats
        if 'start_time' in df.columns:
            # Check if it's a unix timestamp (milliseconds)
            if df['start_time'].dtype in ['int64', 'float64']:
                df['start_datetime'] = pd.to_datetime(df['start_time'], unit='ms')
            else:
                # Try different time formats
                try:
                    df['start_datetime'] = pd.to_datetime(df['start_time'])
                except:
                    df['start_datetime'] = pd.to_datetime('now')  # Fallback
        
        if 'end_time' in df.columns:
            if df['end_time'].dtype in ['int64', 'float64']:
                df['end_datetime'] = pd.to_datetime(df['end_time'], unit='ms')
            else:
                try:
                    df['end_datetime'] = pd.to_datetime(df['end_time'])
                except:
                    df['end_datetime'] = df['start_datetime'] + pd.Timedelta(hours=1)  # Fallback
        
        # If we don't have start_datetime but have datetime column
        if 'start_datetime' not in df.columns and 'datetime' in df.columns:
            df['start_datetime'] = pd.to_datetime(df['datetime'])
        
        # Convert to Arizona timezone
        az_timezone = pytz.timezone('America/Phoenix')
        
        if 'start_datetime' in df.columns:
            try:
                df['start_datetime'] = df['start_datetime'].dt.tz_localize('UTC').dt.tz_convert(az_timezone)
            except:
                try:
                    # Already localized
                    df['start_datetime'] = df['start_datetime'].dt.tz_convert(az_timezone)
                except:
                    pass
        
        if 'end_datetime' in df.columns:
            try:
                df['end_datetime'] = df['end_datetime'].dt.tz_localize('UTC').dt.tz_convert(az_timezone)
            except:
                try:
                    # Already localized
                    df['end_datetime'] = df['end_datetime'].dt.tz_convert(az_timezone)
                except:
                    pass
        
        # Add formatted date and time columns
        if 'start_datetime' in df.columns:
            df['formatted_date'] = df['start_datetime'].dt.strftime('%m-%d-%Y')
            df['formatted_time'] = df['start_datetime'].dt.strftime('%I:%M %p')
            df['formatted_datetime'] = df['start_datetime'].dt.strftime('%m-%d-%Y %I:%M %p')
        
        # Calculate duration in minutes
        if 'start_time' in df.columns and 'end_time' in df.columns:
            if df['start_time'].dtype in ['int64', 'float64'] and df['end_time'].dtype in ['int64', 'float64']:
                df['duration_min'] = (df['end_time'] - df['start_time']) / 60000
        
        # Make sure we have session_id column
        if 'session_id' not in df.columns and '_id' in df.columns:
            df['session_id'] = df['_id']
        
        conn.close()
        return df
    
    except Exception:
        if 'conn' in locals():
            conn.close()
        # Return empty DataFrame with essential columns
        return pd.DataFrame(columns=['_id', 'session_id', 'session_score', 'start_datetime', 'formatted_datetime'])

def safe_parse_json(json_str):
    """
    Safely parse JSON string to dict
    
    Parameters:
    json_str (str): JSON string to parse
    
    Returns:
    dict: Parsed JSON data or empty dict if parsing fails
    """
    try:
        return json.loads(json_str) if pd.notna(json_str) and json_str else {}
    except json.JSONDecodeError:
        return {}

@st.cache_data
def process_sessions_metrics(sessions_df):
    """
    Extract metrics from session data
    
    Parameters:
    sessions_df (pandas.DataFrame): DataFrame of session data
    
    Returns:
    pandas.DataFrame: DataFrame with extracted metrics
    """
    metrics_list = []
    
    for _, row in sessions_df.iterrows():
        try:
            # Initialize default values
            metrics = {
                'session_id': row.get('_id', row.get('session_id', '')),
                'start_time': row.get('start_datetime', pd.Timestamp('now')),
                'formatted_datetime': row.get('formatted_datetime', ''),
                'duration_min': row.get('duration_min', 0),
                'session_score': row.get('session_score', 0),
                'total_shots': 0,
                'forehand_count': 0,
                'backhand_count': 0,
                'serves_count': 0,
                'volley_count': 0,
                'smash_count': 0,
                'consistency_score': 0,
                'power_score': 0,
                'intensity_score': 0,
                'serve_avg_speed': 0,
                'serve_max_speed': 0,
                'forehand_avg_speed': 0,
                'forehand_max_speed': 0,
                'backhand_avg_speed': 0,
                'backhand_max_speed': 0,
                'active_time_min': 0,
                'longest_rally': 0,
            }
            
            # Check if this is a Zepp session with report data
            if 'report' in row and pd.notna(row['report']):
                session_json = safe_parse_json(row['report'])
                
                if session_json and 'session' in session_json:
                    session_data = session_json['session']
                    profile_data = session_json.get('profile_snapshot', {})
                    
                    # Extract shot counts if available
                    shot_breakdown = calculate_shot_breakdown(session_json)
                    if shot_breakdown:
                        metrics.update({
                            'total_shots': shot_breakdown['Total'],
                            'forehand_count': shot_breakdown['Forehand'],
                            'backhand_count': shot_breakdown['Backhand'],
                            'serves_count': shot_breakdown['Serve'],
                            'volley_count': shot_breakdown['Volley'],
                            'smash_count': shot_breakdown['Smash'],
                        })
                    
                    # Get scores
                    if 'swings' in session_data and 'scores' in session_data['swings']:
                        scores = session_data['swings']['scores']
                        metrics.update({
                            'consistency_score': scores.get('consistency_score', 0),
                            'power_score': scores.get('power_score', 0),
                            'intensity_score': scores.get('intensity_score', 0),
                        })
                    
                    # Get serve speeds
                    if 'swings' in session_data and 'serve' in session_data['swings']:
                        serve_data = session_data['swings']['serve']
                        metrics.update({
                            'serve_avg_speed': serve_data.get('serve_average_speed', 0),
                            'serve_max_speed': serve_data.get('serve_max_speed', 0),
                        })
                    
                    # Get forehand/backhand speeds
                    metrics.update({
                        'forehand_avg_speed': calculate_avg_speed(session_data, 'forehand'),
                        'forehand_max_speed': calculate_max_speed(session_data, 'forehand'),
                        'backhand_avg_speed': calculate_avg_speed(session_data, 'backhand'),
                        'backhand_max_speed': calculate_max_speed(session_data, 'backhand'),
                    })
                    
                    # Get activity metrics
                    metrics.update({
                        'active_time_min': session_data.get('active_time', 0) / 60,
                        'longest_rally': session_data.get('longest_rally_swings', 0),
                    })
                    
                    # Store the original JSON
                    metrics['session_json'] = session_json
            
            # Check if this is a Babolat session
            elif any(col in row for col in ['total_shot_count', 'piq_score', 'forehand_count']):
                # Apply Babolat specific processing
                for field in ['total_shot_count', 'forehand_count', 'backhand_count', 
                              'serves_count', 'volley_count', 'smash_count']:
                    if field in row:
                        # Map to the standard field name
                        metrics_field = field if field != 'total_shot_count' else 'total_shots'
                        metrics[metrics_field] = row[field]
                
                # Add scores if available
                score_mapping = [
                    ('piq_score', 'session_score'),
                    ('max_piq_score', 'max_piq_score'),
                    ('consistency_score', 'consistency_score'),
                    ('power_score', 'power_score'),
                    ('activity_level', 'intensity_score')
                ]
                
                for field, target in score_mapping:
                    if field in row:
                        metrics[target] = row[field]
                
                # Add speed metrics if available
                speed_mapping = [
                    ('max_serve_speed', 'serve_max_speed'),
                    ('max_forehand_speed', 'forehand_max_speed'),
                    ('max_backhand_speed', 'backhand_max_speed')
                ]
                
                for field, target in speed_mapping:
                    if field in row:
                        # Babolat speeds might need conversion
                        metrics[target] = row[field]
                
                # Add activity metrics
                if 'active_time' in row:
                    metrics['active_time_min'] = row['active_time']
                if 'best_rally' in row:
                    metrics['longest_rally'] = row['best_rally']
                
                # Create a basic session JSON structure for compatibility
                metrics['session_json'] = {
                    'session': {
                        'scores': {
                            'piq_score': metrics['session_score'],
                            'consistency_score': metrics['consistency_score'],
                            'power_score': metrics['power_score']
                        },
                        'shots': {
                            'total': metrics['total_shots'],
                            'forehand': metrics['forehand_count'],
                            'backhand': metrics['backhand_count'],
                            'serve': metrics['serves_count']
                        },
                        'speeds': {
                            'serve_max': metrics['serve_max_speed'],
                            'forehand_max': metrics['forehand_max_speed'],
                            'backhand_max': metrics['backhand_max_speed']
                        }
                    }
                }
            
            # Only add sessions that have at least some data
            if metrics['total_shots'] > 0 or metrics['session_score'] > 0:
                metrics_list.append(metrics)
            
        except Exception:
            continue
            
    return pd.DataFrame(metrics_list) if metrics_list else pd.DataFrame()

def calculate_shot_breakdown(session_json):
    """
    Calculate shot breakdown from session data
    
    Parameters:
    session_json (dict): Parsed session JSON data
    
    Returns:
    dict: Shot breakdown or None if data is invalid
    """
    try:
        # For Zepp data structure
        if 'session' in session_json and 'swings' in session_json['session']:
            swings = session_json['session']['swings']
            shots = {
                'serve': swings.get('serve', {}).get('serve_swings', 0),
                'forehand': sum(swings.get('forehand', {}).get(f'{type}_swings', 0) 
                              for type in ['flat', 'slice', 'topspin']),
                'backhand': sum(swings.get('backhand', {}).get(f'{type}_swings', 0) 
                              for type in ['flat', 'slice', 'topspin'])
            }
            
            # Add volley and smash counts if available (set to 0 if not)
            shots['volley'] = 0  # Zepp doesn't separate volleys in the same way
            shots['smash'] = 0   # Zepp doesn't separate smashes in the same way
            
            shots['total'] = sum(shots.values())
            
            if shots['total'] == 0:
                return None
                
            return {
                'Serve': shots['serve'],
                'Forehand': shots['forehand'],
                'Backhand': shots['backhand'],
                'Volley': shots['volley'],
                'Smash': shots['smash'],
                'Total': shots['total'],
                'Serve_pct': (shots['serve'] / shots['total'] * 100) if shots['total'] > 0 else 0,
                'Forehand_pct': (shots['forehand'] / shots['total'] * 100) if shots['total'] > 0 else 0,
                'Backhand_pct': (shots['backhand'] / shots['total'] * 100) if shots['total'] > 0 else 0
            }
        
        # For simpler JSON structures (or fallback)
        elif 'shots' in session_json:
            shots = session_json['shots']
            total = sum(shots.values())
            
            return {
                'Serve': shots.get('serve', 0),
                'Forehand': shots.get('forehand', 0),
                'Backhand': shots.get('backhand', 0),
                'Volley': shots.get('volley', 0),
                'Smash': shots.get('smash', 0),
                'Total': total,
                'Serve_pct': (shots.get('serve', 0) / total * 100) if total > 0 else 0,
                'Forehand_pct': (shots.get('forehand', 0) / total * 100) if total > 0 else 0,
                'Backhand_pct': (shots.get('backhand', 0) / total * 100) if total > 0 else 0
            }
    except Exception:
        return None

def calculate_avg_speed(session_data, stroke_type):
    """
    Calculate average speed for a stroke type
    
    Parameters:
    session_data (dict): Session data
    stroke_type (str): Stroke type ('forehand' or 'backhand')
    
    Returns:
    float: Average speed
    """
    if stroke_type not in ['forehand', 'backhand']:
        return 0.0
        
    try:
        # Check if we have the Zepp data structure
        if 'swings' in session_data and stroke_type in session_data['swings']:
            stroke_data = session_data['swings'][stroke_type]
            
            # Calculate weighted average
            speeds = [
                (stroke_data.get(f'{spin}_average_speed', 0), stroke_data.get(f'{spin}_swings', 0))
                for spin in ['flat', 'slice', 'topspin']
            ]
            
            total_swings = sum(count for _, count in speeds)
            if total_swings == 0:
                return 0.0
                
            weighted_avg = sum(speed * count for speed, count in speeds) / total_swings
            return weighted_avg
        
        # Check for simpler structure (for compatibility)
        elif 'speeds' in session_data and f'{stroke_type}_avg' in session_data['speeds']:
            return session_data['speeds'][f'{stroke_type}_avg']
        
        return 0.0
    except Exception:
        return 0.0

def calculate_max_speed(session_data, stroke_type):
    """
    Find maximum speed for a stroke type
    
    Parameters:
    session_data (dict): Session data
    stroke_type (str): Stroke type ('forehand' or 'backhand')
    
    Returns:
    float: Maximum speed
    """
    if stroke_type not in ['forehand', 'backhand']:
        return 0.0
        
    try:
        # Check if we have the Zepp data structure
        if 'swings' in session_data and stroke_type in session_data['swings']:
            stroke_data = session_data['swings'][stroke_type]
            
            # Find maximum speed across spin types
            max_speeds = [
                stroke_data.get(f'{spin}_max_speed', 0)
                for spin in ['flat', 'slice', 'topspin']
            ]
            
            return max(max_speeds) if max_speeds else 0.0
        
        # Check for simpler structure (for compatibility)
        elif 'speeds' in session_data and f'{stroke_type}_max' in session_data['speeds']:
            return session_data['speeds'][f'{stroke_type}_max']
            
        return 0.0
    except Exception:
        return 0.0

def extract_spin_analysis(session_json):
    """
    Extract spin analysis data from session
    
    Parameters:
    session_json (dict): Session JSON data
    
    Returns:
    list: List of spin data dictionaries
    """
    try:
        # Check if this is a Zepp data structure
        if 'session' in session_json and 'swings' in session_json['session']:
            spin_data = []
            swings = session_json['session']['swings']
            
            # Process forehand data
            for spin_type in ['flat', 'slice', 'topspin']:
                swings_count = swings.get('forehand', {}).get(f'{spin_type}_swings', 0)
                if swings_count > 0:
                    spin_data.append({
                        'motionType': 'Forehand',
                        'spinType': spin_type.capitalize(),
                        'count': swings_count
                    })
            
            # Process backhand data
            for spin_type in ['flat', 'slice', 'topspin']:
                swings_count = swings.get('backhand', {}).get(f'{spin_type}_swings', 0)
                if swings_count > 0:
                    spin_data.append({
                        'motionType': 'Backhand',
                        'spinType': spin_type.capitalize(),
                        'count': swings_count
                    })
            
            # Process serve data (Zepp doesn't categorize serves by spin)
            serve_count = swings.get('serve', {}).get('serve_swings', 0)
            if serve_count > 0:
                spin_data.append({
                    'motionType': 'Serve',
                    'spinType': 'All',
                    'count': serve_count
                })
            
            return spin_data
        
        # For Babolat or simpler data structures (try to create from json if available)
        elif 'activity_statistics_spin_json' in session_json and session_json['activity_statistics_spin_json']:
            # Parse the spin JSON if it's a string
            if isinstance(session_json['activity_statistics_spin_json'], str):
                return safe_parse_json(session_json['activity_statistics_spin_json'])
            else:
                return session_json['activity_statistics_spin_json']
        
        # If we have any spin stats in the main JSON
        elif 'spin_stats' in session_json:
            return session_json['spin_stats']
            
        return []
    except Exception:
        return []

def extract_hit_points(session_json, shot_type, spin_type=None):
    """
    Extract hit points data from session
    
    Parameters:
    session_json (dict): Session JSON data
    shot_type (str): Shot type ('forehand', 'backhand', or 'serve')
    spin_type (str, optional): Spin type for forehand/backhand ('flat', 'slice', 'topspin')
    
    Returns:
    list: List of hit point coordinates
    """
    try:
        # For Zepp data structure
        if 'session' in session_json and 'swings' in session_json['session']:
            swings = session_json['session']['swings']
            
            if shot_type == 'serve':
                return swings.get('serve', {}).get('serve_hit_points', [])
            elif shot_type in ['forehand', 'backhand'] and spin_type in ['flat', 'slice', 'topspin']:
                return swings.get(shot_type, {}).get(f'{spin_type}_hit_points', [])
            else:
                return []
        
        # For simpler data structures or fallback
        elif 'hit_points' in session_json:
            hit_points = session_json['hit_points']
            if shot_type in hit_points:
                if spin_type is None or spin_type not in hit_points[shot_type]:
                    return hit_points[shot_type].get('all', [])
                else:
                    return hit_points[shot_type].get(spin_type, [])
            
        return []
    except Exception:
        return []
