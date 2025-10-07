"""
Speed Calculator Module
Calculates speed of ball and players based on position tracking.
"""

import numpy as np
from typing import List, Tuple, Optional


class SpeedCalculator:
    """
    Calculates speed metrics for tennis analysis.
    """
    
    def __init__(self, fps: float = 30.0, pixels_per_meter: float = 100.0):
        """
        Initialize the speed calculator.
        
        Args:
            fps: Video frame rate (frames per second)
            pixels_per_meter: Conversion factor from pixels to meters
        """
        self.fps = fps
        self.pixels_per_meter = pixels_per_meter
        self.meters_per_pixel = 1.0 / pixels_per_meter if pixels_per_meter > 0 else 0.01
        
    def calculate_distance(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """
        Calculate Euclidean distance between two points.
        
        Args:
            point1: First point (x, y)
            point2: Second point (x, y)
            
        Returns:
            Distance in pixels
        """
        return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
    
    def calculate_speed_kmh(self, positions: List[Tuple[int, int]], 
                           time_window: Optional[int] = None) -> float:
        """
        Calculate speed in km/h from a list of positions.
        
        Args:
            positions: List of (x, y) positions
            time_window: Number of frames to consider (None = all frames)
            
        Returns:
            Speed in km/h
        """
        if len(positions) < 2:
            return 0.0
        
        # Use specified time window or all positions
        if time_window and len(positions) > time_window:
            positions = positions[-time_window:]
        
        # Calculate total distance traveled
        total_distance_pixels = 0
        for i in range(1, len(positions)):
            total_distance_pixels += self.calculate_distance(positions[i-1], positions[i])
        
        # Convert to meters
        total_distance_meters = total_distance_pixels * self.meters_per_pixel
        
        # Calculate time elapsed
        time_seconds = (len(positions) - 1) / self.fps
        
        if time_seconds <= 0:
            return 0.0
        
        # Calculate speed in m/s then convert to km/h
        speed_ms = total_distance_meters / time_seconds
        speed_kmh = speed_ms * 3.6
        
        return speed_kmh
    
    def calculate_speed_mph(self, positions: List[Tuple[int, int]], 
                           time_window: Optional[int] = None) -> float:
        """
        Calculate speed in mph from a list of positions.
        
        Args:
            positions: List of (x, y) positions
            time_window: Number of frames to consider (None = all frames)
            
        Returns:
            Speed in mph
        """
        speed_kmh = self.calculate_speed_kmh(positions, time_window)
        return speed_kmh * 0.621371  # Convert km/h to mph
    
    def calculate_instantaneous_speed_kmh(self, pos1: Tuple[int, int], 
                                         pos2: Tuple[int, int]) -> float:
        """
        Calculate instantaneous speed between two consecutive frames.
        
        Args:
            pos1: Position at frame t (x, y)
            pos2: Position at frame t+1 (x, y)
            
        Returns:
            Speed in km/h
        """
        distance_pixels = self.calculate_distance(pos1, pos2)
        distance_meters = distance_pixels * self.meters_per_pixel
        time_seconds = 1.0 / self.fps
        speed_ms = distance_meters / time_seconds
        return speed_ms * 3.6
    
    def calculate_average_speed_kmh(self, positions: List[Tuple[int, int]]) -> float:
        """
        Calculate average speed over all positions.
        
        Args:
            positions: List of (x, y) positions
            
        Returns:
            Average speed in km/h
        """
        if len(positions) < 2:
            return 0.0
        
        # Calculate distance from start to end
        start_pos = positions[0]
        end_pos = positions[-1]
        
        distance_pixels = self.calculate_distance(start_pos, end_pos)
        distance_meters = distance_pixels * self.meters_per_pixel
        
        # Calculate time elapsed
        time_seconds = (len(positions) - 1) / self.fps
        
        if time_seconds <= 0:
            return 0.0
        
        # Calculate average speed
        speed_ms = distance_meters / time_seconds
        return speed_ms * 3.6
    
    def set_scale(self, pixels_per_meter: float):
        """
        Update the pixel-to-meter conversion scale.
        
        Args:
            pixels_per_meter: New conversion factor
        """
        self.pixels_per_meter = pixels_per_meter
        self.meters_per_pixel = 1.0 / pixels_per_meter if pixels_per_meter > 0 else 0.01
