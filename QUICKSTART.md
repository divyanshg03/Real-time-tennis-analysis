# Quick Start Guide

## Installation
```bash
pip install -r requirements.txt
```

## Basic Usage

### 1. Analyze a video file
```bash
python main.py video.mp4
```

### 2. Analyze and save output
```bash
python main.py video.mp4 output.mp4
```

### 3. Real-time webcam analysis
```bash
python main.py webcam
```

### 4. Run demo (no video required)
```bash
python demo.py
```

## Key Features

- **Ball Tracking**: Detects and tracks tennis ball in real-time
- **Player Tracking**: Tracks player positions and movements
- **Speed Calculation**: Calculates ball and player speeds in km/h and mph
- **Visualization**: Overlays tracking data on video frames
- **Statistics**: Provides comprehensive game statistics

## Interactive Controls (during playback)

- `q` - Quit the analysis
- `s` - Show current statistics in console
- `r` - Reset analysis (webcam mode only)

## Configuration

Edit `config.ini` to customize:
- Tracking parameters
- Speed calculation settings
- Visualization colors
- Display options

## Troubleshooting

**Issue**: ModuleNotFoundError: No module named 'cv2'
**Solution**: Install dependencies with `pip install -r requirements.txt`

**Issue**: Video file not opening
**Solution**: Ensure the video file path is correct and the format is supported (mp4, avi, mov, etc.)

**Issue**: Webcam not detected
**Solution**: Ensure your webcam is connected and not in use by another application

## System Requirements

- Python 3.7+
- OpenCV 4.5+
- NumPy 1.19+
- Webcam (optional, for real-time analysis)

## Performance Tips

- Use lower resolution videos for faster processing
- Close other applications to free up system resources
- For recorded analysis, disable real-time display with `show_display=False` in code

## Support

For issues and questions, please open an issue on GitHub.
