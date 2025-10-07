"""
Ball Tracker Module
Tracks tennis ball position and trajectory in real-time.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


class BallTracker:
    """
    Tracks tennis ball movement in video frames.
    Uses color detection and contour analysis to identify and track the ball.
    """
    
    def __init__(self, color_range: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        """
        Initialize the ball tracker.
        
        Args:
            color_range: Tuple of (lower_bound, upper_bound) HSV color range for ball detection.
                        Default is yellow-green tennis ball color.
        """
        if color_range is None:
            # Default HSV range for tennis ball (yellow-green)
            self.lower_bound = np.array([20, 100, 100])
            self.upper_bound = np.array([40, 255, 255])
        else:
            self.lower_bound, self.upper_bound = color_range
        
        self.ball_positions = []
        self.min_radius = 5
        self.max_radius = 50
        
    def detect_ball(self, frame: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """
        Detect ball in a single frame.
        
        Args:
            frame: Input video frame (BGR format)
            
        Returns:
            Tuple of (x, y, radius) if ball is detected, None otherwise
        """
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for ball color
        mask = cv2.inRange(hsv, self.lower_bound, self.upper_bound)
        
        # Apply morphological operations to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find the largest circular contour
        best_circle = None
        max_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 20:  # Too small
                continue
            
            # Get minimum enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(contour)
            
            if self.min_radius <= radius <= self.max_radius:
                # Calculate circularity score
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    score = circularity * area  # Favor larger, more circular objects
                    
                    if score > max_score:
                        max_score = score
                        best_circle = (int(x), int(y), int(radius))
        
        return best_circle
    
    def track(self, frame: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """
        Track ball across frames and maintain position history.
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple of (x, y, radius) if ball is detected, None otherwise
        """
        ball = self.detect_ball(frame)
        
        if ball is not None:
            x, y, radius = ball
            self.ball_positions.append((x, y))
            
            # Keep only last 30 positions for trajectory
            if len(self.ball_positions) > 30:
                self.ball_positions.pop(0)
        
        return ball
    
    def get_trajectory(self) -> List[Tuple[int, int]]:
        """
        Get recent ball trajectory.
        
        Returns:
            List of (x, y) positions
        """
        return self.ball_positions.copy()
    
    def reset(self):
        """Reset tracking history."""
        self.ball_positions = []
