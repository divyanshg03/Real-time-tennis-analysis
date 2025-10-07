"""
Tennis Analyzer Module
Main analysis engine that coordinates all tracking and metrics calculation.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trackers.ball_tracker import BallTracker
from trackers.player_tracker import PlayerTracker
from utils.court_detector import CourtDetector
from utils.speed_calculator import SpeedCalculator


class TennisAnalyzer:
    """
    Main tennis analysis system that coordinates tracking and metrics.
    """
    
    def __init__(self, fps: float = 30.0, pixels_per_meter: float = 100.0):
        """
        Initialize the tennis analyzer.
        
        Args:
            fps: Video frame rate
            pixels_per_meter: Scale factor for distance calculations
        """
        self.ball_tracker = BallTracker()
        self.player_tracker = PlayerTracker()
        self.court_detector = CourtDetector()
        self.speed_calculator = SpeedCalculator(fps, pixels_per_meter)
        
        self.frame_count = 0
        self.metrics_history = []
        
    def analyze_frame(self, frame: np.ndarray) -> Dict:
        """
        Analyze a single frame and extract all metrics.
        
        Args:
            frame: Input video frame (BGR format)
            
        Returns:
            Dictionary containing all analysis metrics
        """
        self.frame_count += 1
        
        # Track ball
        ball = self.ball_tracker.track(frame)
        ball_trajectory = self.ball_tracker.get_trajectory()
        
        # Calculate ball speed
        ball_speed_kmh = 0.0
        ball_speed_mph = 0.0
        if len(ball_trajectory) >= 2:
            ball_speed_kmh = self.speed_calculator.calculate_speed_kmh(
                ball_trajectory, time_window=5
            )
            ball_speed_mph = self.speed_calculator.calculate_speed_mph(
                ball_trajectory, time_window=5
            )
        
        # Track players
        players = self.player_tracker.track(frame)
        
        # Calculate player speeds
        player_metrics = []
        for player in players:
            player_id = player['id']
            player_history = self.player_tracker.get_player_history(player_id, frames=5)
            
            if len(player_history) >= 2:
                speed_kmh = self.speed_calculator.calculate_speed_kmh(
                    player_history, time_window=5
                )
                speed_mph = self.speed_calculator.calculate_speed_mph(
                    player_history, time_window=5
                )
            else:
                speed_kmh = 0.0
                speed_mph = 0.0
            
            player_metrics.append({
                'id': player_id,
                'position': player['position'],
                'bbox': player['bbox'],
                'speed_kmh': speed_kmh,
                'speed_mph': speed_mph
            })
        
        # Compile frame metrics
        frame_metrics = {
            'frame': self.frame_count,
            'ball': {
                'detected': ball is not None,
                'position': (ball[0], ball[1]) if ball else None,
                'radius': ball[2] if ball else None,
                'speed_kmh': ball_speed_kmh,
                'speed_mph': ball_speed_mph,
                'trajectory': ball_trajectory
            },
            'players': player_metrics,
            'num_players': len(players)
        }
        
        self.metrics_history.append(frame_metrics)
        
        # Keep only last 100 frames of history
        if len(self.metrics_history) > 100:
            self.metrics_history.pop(0)
        
        return frame_metrics
    
    def get_game_statistics(self) -> Dict:
        """
        Calculate aggregate statistics over the analyzed frames.
        
        Returns:
            Dictionary containing game statistics
        """
        if not self.metrics_history:
            return {}
        
        # Ball statistics
        ball_speeds = [m['ball']['speed_kmh'] for m in self.metrics_history 
                      if m['ball']['speed_kmh'] > 0]
        
        # Player statistics
        player_speeds = []
        for metrics in self.metrics_history:
            for player in metrics['players']:
                if player['speed_kmh'] > 0:
                    player_speeds.append(player['speed_kmh'])
        
        stats = {
            'total_frames_analyzed': len(self.metrics_history),
            'ball_stats': {
                'max_speed_kmh': max(ball_speeds) if ball_speeds else 0,
                'max_speed_mph': max(ball_speeds) * 0.621371 if ball_speeds else 0,
                'avg_speed_kmh': np.mean(ball_speeds) if ball_speeds else 0,
                'avg_speed_mph': np.mean(ball_speeds) * 0.621371 if ball_speeds else 0,
                'detection_rate': sum(1 for m in self.metrics_history 
                                    if m['ball']['detected']) / len(self.metrics_history) * 100
            },
            'player_stats': {
                'max_speed_kmh': max(player_speeds) if player_speeds else 0,
                'max_speed_mph': max(player_speeds) * 0.621371 if player_speeds else 0,
                'avg_speed_kmh': np.mean(player_speeds) if player_speeds else 0,
                'avg_speed_mph': np.mean(player_speeds) * 0.621371 if player_speeds else 0
            }
        }
        
        return stats
    
    def reset(self):
        """Reset all tracking and metrics."""
        self.ball_tracker.reset()
        self.player_tracker.reset()
        self.frame_count = 0
        self.metrics_history = []
