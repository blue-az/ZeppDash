from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    """Configuration settings for the Zepp Tennis Dashboard"""
    
    # Database paths
    DB_PATH: Path = Path('./ztennis.db')  # Path to session database
    SHOT_DB_PATH: Path = Path('./ztennis.db')  # For Zepp, shots are extracted from session data
    
    # Time and timezone settings
    TIMEZONE: str = 'America/Phoenix'
    
    # Conversion factors and thresholds
    SPEED_CONVERSION_FACTOR: float = 2.25  # m/s to mph
    DEFAULT_ROLLING_WINDOW: int = 5
    MIN_SPEED_THRESHOLD: float = 50.0
    
    # Plot colors
    PLOT_COLORS: dict = None
    
    def __post_init__(self):
        self.PLOT_COLORS = {
            'best_piq': '#2ecc71',
            'avg_piq': '#3498db',
            'forehand': '#e67e22',
            'backhand': '#9b59b6',
            'serve': '#e74c3c',
            'consistency': '#1f77b4',
            'power': '#ff7f0e',
            'intensity': '#2ca02c',
            'overall': '#d62728'
        }
