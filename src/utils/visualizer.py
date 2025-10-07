"""
Visualization Module
Provides visualization utilities for displaying analysis results on video frames.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple


class TennisVisualizer:
    """
    Visualizes tennis analysis results on video frames.
    """
    
    def __init__(self):
        """Initialize the visualizer."""
        self.ball_color = (0, 255, 255)  # Yellow
        self.player_color = (255, 0, 0)  # Blue
        self.trajectory_color = (0, 255, 0)  # Green
        self.text_color = (255, 255, 255)  # White
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.thickness = 2
        
    def draw_ball(self, frame: np.ndarray, ball_info: Dict) -> np.ndarray:
        """
        Draw ball position and information on frame.
        
        Args:
            frame: Input video frame
            ball_info: Ball information dictionary
            
        Returns:
            Frame with ball visualization
        """
        if not ball_info['detected']:
            return frame
        
        x, y = ball_info['position']
        radius = ball_info['radius']
        
        # Draw ball circle
        cv2.circle(frame, (x, y), radius, self.ball_color, 2)
        cv2.circle(frame, (x, y), 2, self.ball_color, -1)
        
        # Draw ball speed
        speed_text = f"Ball: {ball_info['speed_kmh']:.1f} km/h ({ball_info['speed_mph']:.1f} mph)"
        cv2.putText(frame, speed_text, (x + radius + 5, y), 
                   self.font, self.font_scale, self.text_color, self.thickness)
        
        return frame
    
    def draw_trajectory(self, frame: np.ndarray, trajectory: List[Tuple[int, int]], 
                       color: Tuple[int, int, int] = None) -> np.ndarray:
        """
        Draw ball or player trajectory on frame.
        
        Args:
            frame: Input video frame
            trajectory: List of (x, y) positions
            color: Color for trajectory line
            
        Returns:
            Frame with trajectory visualization
        """
        if len(trajectory) < 2:
            return frame
        
        color = color or self.trajectory_color
        
        # Draw trajectory lines
        for i in range(1, len(trajectory)):
            pt1 = trajectory[i-1]
            pt2 = trajectory[i]
            cv2.line(frame, pt1, pt2, color, 2)
        
        return frame
    
    def draw_players(self, frame: np.ndarray, players: List[Dict]) -> np.ndarray:
        """
        Draw player bounding boxes and information on frame.
        
        Args:
            frame: Input video frame
            players: List of player information dictionaries
            
        Returns:
            Frame with player visualization
        """
        for player in players:
            x, y, w, h = player['bbox']
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), self.player_color, 2)
            
            # Draw player ID
            id_text = f"Player {player['id']}"
            cv2.putText(frame, id_text, (x, y - 30), 
                       self.font, self.font_scale, self.text_color, self.thickness)
            
            # Draw player speed
            speed_text = f"{player['speed_kmh']:.1f} km/h"
            cv2.putText(frame, speed_text, (x, y - 10), 
                       self.font, self.font_scale, self.text_color, self.thickness)
        
        return frame
    
    def draw_statistics(self, frame: np.ndarray, stats: Dict, 
                       position: Tuple[int, int] = (10, 30)) -> np.ndarray:
        """
        Draw game statistics on frame.
        
        Args:
            frame: Input video frame
            stats: Statistics dictionary
            position: Top-left position for stats display
            
        Returns:
            Frame with statistics visualization
        """
        x, y = position
        line_height = 25
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 5, y - 20), (x + 400, y + line_height * 6), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Draw statistics text
        if 'ball_stats' in stats:
            ball_stats = stats['ball_stats']
            cv2.putText(frame, f"Ball Max Speed: {ball_stats['max_speed_kmh']:.1f} km/h", 
                       (x, y), self.font, self.font_scale, self.text_color, self.thickness)
            y += line_height
            cv2.putText(frame, f"Ball Avg Speed: {ball_stats['avg_speed_kmh']:.1f} km/h", 
                       (x, y), self.font, self.font_scale, self.text_color, self.thickness)
            y += line_height
        
        if 'player_stats' in stats:
            player_stats = stats['player_stats']
            cv2.putText(frame, f"Player Max Speed: {player_stats['max_speed_kmh']:.1f} km/h", 
                       (x, y), self.font, self.font_scale, self.text_color, self.thickness)
            y += line_height
            cv2.putText(frame, f"Player Avg Speed: {player_stats['avg_speed_kmh']:.1f} km/h", 
                       (x, y), self.font, self.font_scale, self.text_color, self.thickness)
        
        return frame
    
    def visualize_frame(self, frame: np.ndarray, metrics: Dict, 
                       show_trajectory: bool = True,
                       show_stats: bool = False,
                       game_stats: Dict = None) -> np.ndarray:
        """
        Apply all visualizations to a frame.
        
        Args:
            frame: Input video frame
            metrics: Frame metrics dictionary
            show_trajectory: Whether to show ball trajectory
            show_stats: Whether to show game statistics
            game_stats: Game statistics dictionary (required if show_stats=True)
            
        Returns:
            Fully annotated frame
        """
        annotated = frame.copy()
        
        # Draw ball
        annotated = self.draw_ball(annotated, metrics['ball'])
        
        # Draw trajectory
        if show_trajectory and len(metrics['ball']['trajectory']) > 0:
            annotated = self.draw_trajectory(annotated, metrics['ball']['trajectory'])
        
        # Draw players
        annotated = self.draw_players(annotated, metrics['players'])
        
        # Draw statistics overlay
        if show_stats and game_stats:
            annotated = self.draw_statistics(annotated, game_stats)
        
        # Draw frame number
        cv2.putText(annotated, f"Frame: {metrics['frame']}", (10, frame.shape[0] - 10),
                   self.font, self.font_scale, self.text_color, self.thickness)
        
        return annotated
