"""
Court Detector Module
Detects tennis court boundaries and establishes coordinate system.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


class CourtDetector:
    """
    Detects tennis court lines and establishes a coordinate system.
    """
    
    def __init__(self):
        """Initialize the court detector."""
        self.court_lines = []
        self.court_corners = None
        # Standard tennis court dimensions in feet
        self.court_length_ft = 78
        self.court_width_ft = 36
        
    def detect_lines(self, frame: np.ndarray) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Detect court lines in a frame.
        
        Args:
            frame: Input video frame (BGR format)
            
        Returns:
            List of line segments [(start_point, end_point), ...]
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        
        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(
            edges, 
            rho=1, 
            theta=np.pi/180, 
            threshold=100,
            minLineLength=100,
            maxLineGap=10
        )
        
        detected_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                detected_lines.append(((x1, y1), (x2, y2)))
        
        self.court_lines = detected_lines
        return detected_lines
    
    def find_court_corners(self, frame: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        """
        Find the four corners of the tennis court.
        
        Args:
            frame: Input video frame
            
        Returns:
            List of four corner points or None if not detected
        """
        lines = self.detect_lines(frame)
        
        if len(lines) < 4:
            return None
        
        # For simplified implementation, use frame dimensions
        # In a real implementation, this would use more sophisticated line intersection
        height, width = frame.shape[:2]
        
        # Estimate court corners based on frame dimensions
        # This is a simplified approach
        margin_x = width // 10
        margin_y = height // 10
        
        corners = [
            (margin_x, margin_y),  # Top-left
            (width - margin_x, margin_y),  # Top-right
            (width - margin_x, height - margin_y),  # Bottom-right
            (margin_x, height - margin_y)  # Bottom-left
        ]
        
        self.court_corners = corners
        return corners
    
    def get_court_dimensions(self) -> Tuple[float, float]:
        """
        Get tennis court dimensions.
        
        Returns:
            Tuple of (length_ft, width_ft)
        """
        return (self.court_length_ft, self.court_width_ft)
    
    def pixel_to_feet(self, pixel_distance: float, reference_axis: str = 'length') -> float:
        """
        Convert pixel distance to feet based on court dimensions.
        
        Args:
            pixel_distance: Distance in pixels
            reference_axis: 'length' or 'width' for reference
            
        Returns:
            Distance in feet
        """
        if self.court_corners is None:
            return pixel_distance * 0.1  # Default rough estimate
        
        if reference_axis == 'length':
            # Calculate pixel length of court
            top_left, top_right, bottom_right, bottom_left = self.court_corners
            pixel_length = np.sqrt(
                (bottom_left[0] - top_left[0])**2 + 
                (bottom_left[1] - top_left[1])**2
            )
            scale = self.court_length_ft / pixel_length if pixel_length > 0 else 0.1
        else:  # width
            top_left, top_right, bottom_right, bottom_left = self.court_corners
            pixel_width = np.sqrt(
                (top_right[0] - top_left[0])**2 + 
                (top_right[1] - top_left[1])**2
            )
            scale = self.court_width_ft / pixel_width if pixel_width > 0 else 0.1
        
        return pixel_distance * scale
