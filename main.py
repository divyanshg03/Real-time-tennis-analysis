"""
Main Tennis Analysis Script
Example usage of the real-time tennis analysis system.
"""

import cv2
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from analysis.tennis_analyzer import TennisAnalyzer
from utils.visualizer import TennisVisualizer


def analyze_video(video_path: str, output_path: str = None, show_display: bool = True):
    """
    Analyze a tennis video file.
    
    Args:
        video_path: Path to input video file
        output_path: Path to save output video (optional)
        show_display: Whether to display the analysis in real-time
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print()
    
    # Initialize analyzer and visualizer
    analyzer = TennisAnalyzer(fps=fps, pixels_per_meter=100.0)
    visualizer = TennisVisualizer()
    
    # Setup video writer if output path is provided
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    print("Starting analysis...")
    print("Press 'q' to quit, 's' to show statistics")
    print()
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        
        # Analyze frame
        metrics = analyzer.analyze_frame(frame)
        
        # Get game statistics
        game_stats = analyzer.get_game_statistics()
        
        # Visualize results
        annotated_frame = visualizer.visualize_frame(
            frame, metrics, 
            show_trajectory=True,
            show_stats=True,
            game_stats=game_stats
        )
        
        # Write to output file
        if writer:
            writer.write(annotated_frame)
        
        # Display frame
        if show_display:
            cv2.imshow('Tennis Analysis', annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                print("\n=== Current Game Statistics ===")
                if game_stats:
                    print(f"Frames analyzed: {game_stats['total_frames_analyzed']}")
                    print("\nBall Statistics:")
                    print(f"  Max Speed: {game_stats['ball_stats']['max_speed_kmh']:.1f} km/h "
                          f"({game_stats['ball_stats']['max_speed_mph']:.1f} mph)")
                    print(f"  Avg Speed: {game_stats['ball_stats']['avg_speed_kmh']:.1f} km/h "
                          f"({game_stats['ball_stats']['avg_speed_mph']:.1f} mph)")
                    print(f"  Detection Rate: {game_stats['ball_stats']['detection_rate']:.1f}%")
                    print("\nPlayer Statistics:")
                    print(f"  Max Speed: {game_stats['player_stats']['max_speed_kmh']:.1f} km/h "
                          f"({game_stats['player_stats']['max_speed_mph']:.1f} mph)")
                    print(f"  Avg Speed: {game_stats['player_stats']['avg_speed_kmh']:.1f} km/h "
                          f"({game_stats['player_stats']['avg_speed_mph']:.1f} mph)")
                print("==============================\n")
        
        # Progress update
        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames} frames...")
    
    # Cleanup
    cap.release()
    if writer:
        writer.release()
    if show_display:
        cv2.destroyAllWindows()
    
    # Print final statistics
    print("\n=== Final Game Statistics ===")
    game_stats = analyzer.get_game_statistics()
    if game_stats:
        print(f"Total frames analyzed: {game_stats['total_frames_analyzed']}")
        print("\nBall Statistics:")
        print(f"  Max Speed: {game_stats['ball_stats']['max_speed_kmh']:.1f} km/h "
              f"({game_stats['ball_stats']['max_speed_mph']:.1f} mph)")
        print(f"  Avg Speed: {game_stats['ball_stats']['avg_speed_kmh']:.1f} km/h "
              f"({game_stats['ball_stats']['avg_speed_mph']:.1f} mph)")
        print(f"  Detection Rate: {game_stats['ball_stats']['detection_rate']:.1f}%")
        print("\nPlayer Statistics:")
        print(f"  Max Speed: {game_stats['player_stats']['max_speed_kmh']:.1f} km/h "
              f"({game_stats['player_stats']['max_speed_mph']:.1f} mph)")
        print(f"  Avg Speed: {game_stats['player_stats']['avg_speed_kmh']:.1f} km/h "
              f"({game_stats['player_stats']['avg_speed_mph']:.1f} mph)")
    print("============================")
    
    if output_path:
        print(f"\nOutput saved to: {output_path}")


def analyze_webcam():
    """
    Analyze tennis game from webcam in real-time.
    """
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Get webcam properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    # Initialize analyzer and visualizer
    analyzer = TennisAnalyzer(fps=fps, pixels_per_meter=100.0)
    visualizer = TennisVisualizer()
    
    print("Starting real-time analysis from webcam...")
    print("Press 'q' to quit, 's' to show statistics, 'r' to reset")
    print()
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Analyze frame
        metrics = analyzer.analyze_frame(frame)
        
        # Get game statistics
        game_stats = analyzer.get_game_statistics()
        
        # Visualize results
        annotated_frame = visualizer.visualize_frame(
            frame, metrics, 
            show_trajectory=True,
            show_stats=True,
            game_stats=game_stats
        )
        
        # Display frame
        cv2.imshow('Real-time Tennis Analysis', annotated_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            print("\n=== Current Game Statistics ===")
            if game_stats:
                print(f"Frames analyzed: {game_stats['total_frames_analyzed']}")
                print("\nBall Statistics:")
                print(f"  Max Speed: {game_stats['ball_stats']['max_speed_kmh']:.1f} km/h "
                      f"({game_stats['ball_stats']['max_speed_mph']:.1f} mph)")
                print(f"  Avg Speed: {game_stats['ball_stats']['avg_speed_kmh']:.1f} km/h "
                      f"({game_stats['ball_stats']['avg_speed_mph']:.1f} mph)")
                print("\nPlayer Statistics:")
                print(f"  Max Speed: {game_stats['player_stats']['max_speed_kmh']:.1f} km/h "
                      f"({game_stats['player_stats']['max_speed_mph']:.1f} mph)")
                print(f"  Avg Speed: {game_stats['player_stats']['avg_speed_kmh']:.1f} km/h "
                      f"({game_stats['player_stats']['avg_speed_mph']:.1f} mph)")
            print("==============================\n")
        elif key == ord('r'):
            analyzer.reset()
            print("Analysis reset.")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Real-time Tennis Analysis System")
        print("=================================")
        print()
        print("Usage:")
        print("  python main.py <video_path> [output_path]  - Analyze video file")
        print("  python main.py webcam                       - Analyze from webcam")
        print()
        print("Examples:")
        print("  python main.py tennis_match.mp4")
        print("  python main.py tennis_match.mp4 output_analyzed.mp4")
        print("  python main.py webcam")
        sys.exit(1)
    
    if sys.argv[1].lower() == 'webcam':
        analyze_webcam()
    else:
        video_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        analyze_video(video_path, output_path)
