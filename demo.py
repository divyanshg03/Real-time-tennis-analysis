"""
Example Demo Script
Demonstrates the basic usage of the tennis analysis system without video input.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from analysis.tennis_analyzer import TennisAnalyzer
from utils.speed_calculator import SpeedCalculator
import numpy as np


def demo_speed_calculation():
    """Demonstrate speed calculation with sample positions."""
    print("=" * 50)
    print("Tennis Analysis System - Speed Calculator Demo")
    print("=" * 50)
    print()
    
    # Initialize speed calculator
    calculator = SpeedCalculator(fps=30.0, pixels_per_meter=100.0)
    
    # Simulate ball positions (moving from (100, 100) to (500, 300))
    positions = []
    for i in range(10):
        x = 100 + i * 40
        y = 100 + i * 20
        positions.append((x, y))
    
    print("Sample ball trajectory positions:")
    for i, pos in enumerate(positions):
        print(f"  Frame {i}: {pos}")
    print()
    
    # Calculate speeds
    speed_kmh = calculator.calculate_speed_kmh(positions)
    speed_mph = calculator.calculate_speed_mph(positions)
    avg_speed = calculator.calculate_average_speed_kmh(positions)
    
    print("Speed Analysis Results:")
    print(f"  Speed (km/h): {speed_kmh:.2f}")
    print(f"  Speed (mph): {speed_mph:.2f}")
    print(f"  Average Speed (km/h): {avg_speed:.2f}")
    print()
    
    # Calculate instantaneous speed
    if len(positions) >= 2:
        inst_speed = calculator.calculate_instantaneous_speed_kmh(
            positions[0], positions[1]
        )
        print(f"  Instantaneous Speed (frame 0 to 1): {inst_speed:.2f} km/h")
    print()


def demo_analyzer_metrics():
    """Demonstrate analyzer metrics structure."""
    print("=" * 50)
    print("Tennis Analysis System - Metrics Structure Demo")
    print("=" * 50)
    print()
    
    analyzer = TennisAnalyzer(fps=30.0, pixels_per_meter=100.0)
    
    # Create a dummy frame
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    print("Analyzing sample frame...")
    metrics = analyzer.analyze_frame(dummy_frame)
    
    print("\nFrame Metrics Structure:")
    print(f"  Frame Number: {metrics['frame']}")
    print(f"  Ball Detected: {metrics['ball']['detected']}")
    print(f"  Ball Position: {metrics['ball']['position']}")
    print(f"  Ball Speed (km/h): {metrics['ball']['speed_kmh']:.2f}")
    print(f"  Ball Speed (mph): {metrics['ball']['speed_mph']:.2f}")
    print(f"  Number of Players: {metrics['num_players']}")
    print()
    
    # Analyze a few more frames
    for _ in range(5):
        analyzer.analyze_frame(dummy_frame)
    
    # Get game statistics
    stats = analyzer.get_game_statistics()
    
    print("Game Statistics:")
    print(f"  Total Frames Analyzed: {stats['total_frames_analyzed']}")
    print(f"  Ball Detection Rate: {stats['ball_stats']['detection_rate']:.1f}%")
    print(f"  Max Ball Speed: {stats['ball_stats']['max_speed_kmh']:.2f} km/h")
    print(f"  Avg Ball Speed: {stats['ball_stats']['avg_speed_kmh']:.2f} km/h")
    print()


def main():
    """Run all demos."""
    print("\n")
    print("╔════════════════════════════════════════════════╗")
    print("║  Real-time Tennis Analysis System - Demo      ║")
    print("╚════════════════════════════════════════════════╝")
    print()
    
    demo_speed_calculation()
    print()
    demo_analyzer_metrics()
    
    print("=" * 50)
    print("Demo Complete!")
    print("=" * 50)
    print()
    print("To analyze actual video files, use:")
    print("  python main.py <video_path>")
    print()
    print("For webcam analysis, use:")
    print("  python main.py webcam")
    print()


if __name__ == "__main__":
    main()
