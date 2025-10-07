# Real-time Tennis Analysis

A real-time tennis game analysis system that tracks and analyzes key factors in tennis matches, including ball speed, player speed, and other performance metrics.

## Features

- **Ball Tracking**: Real-time detection and tracking of tennis ball trajectory
- **Player Tracking**: Identification and tracking of player positions and movements
- **Speed Analysis**: 
  - Ball speed calculation in km/h and mph
  - Player movement speed tracking
  - Instantaneous and average speed metrics
- **Court Detection**: Automatic tennis court boundary detection
- **Real-time Visualization**: Overlays tracking data and metrics on video frames
- **Statistics**: Comprehensive game statistics including:
  - Maximum and average ball speeds
  - Maximum and average player speeds
  - Ball detection rates
  - Frame-by-frame metrics

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/divyanshg03/Real-time-tennis-analysis.git
cd Real-time-tennis-analysis
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Analyze a Video File

To analyze a tennis match video:

```bash
python main.py <video_path>
```

To save the analyzed output:

```bash
python main.py <video_path> <output_path>
```

Example:
```bash
python main.py tennis_match.mp4 analyzed_output.mp4
```

### Real-time Webcam Analysis

To analyze tennis in real-time from a webcam:

```bash
python main.py webcam
```

### Interactive Controls

While the analysis is running:
- Press `q` to quit
- Press `s` to show current statistics in the console
- Press `r` to reset the analysis (webcam mode only)

## Project Structure

```
Real-time-tennis-analysis/
├── src/
│   ├── analysis/
│   │   ├── __init__.py
│   │   └── tennis_analyzer.py    # Main analysis engine
│   ├── trackers/
│   │   ├── __init__.py
│   │   ├── ball_tracker.py       # Ball detection and tracking
│   │   └── player_tracker.py     # Player detection and tracking
│   └── utils/
│       ├── __init__.py
│       ├── court_detector.py     # Court line detection
│       ├── speed_calculator.py   # Speed calculation utilities
│       └── visualizer.py         # Visualization functions
├── main.py                        # Main application script
├── config.ini                     # Configuration file
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── LICENSE                        # Apache 2.0 License
```

## How It Works

### 1. Ball Tracking
The system uses color detection and contour analysis to identify the tennis ball in each frame. It tracks the ball's position over time to calculate trajectory and speed.

### 2. Player Tracking
Background subtraction and contour detection are used to identify and track player positions on the court. The system can handle multiple players (singles or doubles matches).

### 3. Speed Calculation
Speed is calculated by:
- Tracking position changes over time
- Converting pixel distances to real-world measurements
- Calculating instantaneous and average speeds

### 4. Visualization
All tracking data and metrics are overlaid on the video frames in real-time, including:
- Ball position with speed display
- Ball trajectory path
- Player bounding boxes with speed
- Game statistics panel

## Configuration

Edit `config.ini` to customize:
- Video processing parameters
- Tracking sensitivity
- Court dimensions
- Visualization colors and styles
- Display options

## Key Metrics Analyzed

### Ball Metrics
- Position (x, y coordinates)
- Speed (km/h and mph)
- Trajectory
- Detection rate

### Player Metrics
- Position on court
- Movement speed
- Bounding box dimensions

### Game Statistics
- Maximum ball speed
- Average ball speed
- Maximum player speed
- Average player speed
- Total frames analyzed

## Technical Details

### Dependencies
- **OpenCV (cv2)**: Video processing, image manipulation, and computer vision
- **NumPy**: Numerical computations and array operations

### Algorithms Used
- **Color-based Detection**: HSV color space filtering for ball detection
- **Background Subtraction**: MOG2 algorithm for player detection
- **Contour Analysis**: For object identification and tracking
- **Morphological Operations**: For noise reduction and feature enhancement
- **Hough Transform**: For court line detection

## Limitations and Future Improvements

### Current Limitations
- Requires good lighting conditions
- Ball color detection may need calibration for different ball colors
- Court detection is simplified and may need enhancement
- Speed calculations are estimates based on pixel-to-meter conversion

### Future Enhancements
- Machine learning-based object detection (YOLO, SSD)
- Shot classification (forehand, backhand, serve, volley)
- Rally counting and analysis
- Player pose estimation
- Improved court perspective correction
- Heat maps for player positioning
- Advanced statistics (spin rate, bounce angle, etc.)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Author

Divyansh Gupta

## Acknowledgments

- OpenCV community for excellent computer vision tools
- Tennis analysis research community for insights on sports analytics