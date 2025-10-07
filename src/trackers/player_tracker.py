"""
Player Tracker Module
Tracks player positions and movements on the tennis court.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional


class PlayerTracker:
    """
    Tracks player positions on the tennis court.
    Uses background subtraction and contour detection to identify players.
    """
    
    def __init__(self):
        """Initialize the player tracker."""
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )
        self.player_positions = []
        self.min_contour_area = 1000
        self.max_players = 4  # For doubles
        
    def detect_players(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect players in a single frame.
        
        Args:
            frame: Input video frame (BGR format)
            
        Returns:
            List of bounding boxes [(x, y, width, height), ...]
        """
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Remove shadows (value 127 in mask)
        fg_mask[fg_mask == 127] = 0
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        players = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < self.min_contour_area:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by aspect ratio (players should be taller than wide)
            aspect_ratio = h / w if w > 0 else 0
            if 1.0 < aspect_ratio < 4.0:  # Reasonable range for standing players
                players.append((x, y, w, h))
        
        # Sort by area and keep only the largest detections (actual players)
        players.sort(key=lambda p: p[2] * p[3], reverse=True)
        players = players[:self.max_players]
        
        return players
    
    def track(self, frame: np.ndarray) -> List[Dict[str, any]]:
        """
        Track players and calculate their positions.
        
        Args:
            frame: Input video frame
            
        Returns:
            List of player dictionaries with position and centroid information
        """
        bboxes = self.detect_players(frame)
        
        tracked_players = []
        for idx, (x, y, w, h) in enumerate(bboxes):
            centroid_x = x + w // 2
            centroid_y = y + h // 2
            
            player_info = {
                'id': idx,
                'bbox': (x, y, w, h),
                'centroid': (centroid_x, centroid_y),
                'position': (centroid_x, centroid_y + h // 2)  # Bottom center
            }
            tracked_players.append(player_info)
        
        self.player_positions.append(tracked_players)
        
        # Keep only last 30 frames of position history
        if len(self.player_positions) > 30:
            self.player_positions.pop(0)
        
        return tracked_players
    
    def get_player_history(self, player_id: int, frames: int = 10) -> List[Tuple[int, int]]:
        """
        Get position history for a specific player.
        
        Args:
            player_id: Player identifier
            frames: Number of past frames to retrieve
            
        Returns:
            List of positions [(x, y), ...]
        """
        history = []
        for frame_players in self.player_positions[-frames:]:
            for player in frame_players:
                if player['id'] == player_id:
                    history.append(player['position'])
                    break
        return history
    
    def reset(self):
        """Reset tracking history."""
        self.player_positions = []
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )
