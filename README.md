# Pose Art Studio

A real-time, AI-powered pose estimation and artistic rendering application. Transform your camera feed into various artistic styles using MediaPipe and OpenCV.

## Features

- **Real-time Pose Tracking**: Accurate body tracking using MediaPipe Pose Landmarker.
- **Artistic Styles**:
    - **Aura**: A shimmering energy field with smoke trails.
    - **Ultimate 3D Wireframe**: Volumetric line art representing the body.
    - **Magic Button**: Interactive physics-based button that reacts to your movement.
    - **Hell Fire / Shadow Void**: Energetic particle effects.
    - **Minimalist Line Art**: Clean, ghosting trail effects.
- **Gesture Control**: Change styles or trigger effects using hand gestures.
- **Background Customization**: Choose between room background, solid colors, or photo backgrounds.

## Prerequisites

- Python 3.9 or higher
- Webcam

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd final_pose
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r sanjay/requirements.txt
   ```

## Usage

Run the application:
```bash
python sanjay/app.py
```

- **Toggle Camera**: Start the live feed.
- **Style Menu**: Select different artistic renderings.
- **Gesture Support**: Enable gesture control to change styles with a "Hands Up" gesture or trigger Aura surges by crossing your wrists.

## Assets

The application automatically downloads the necessary MediaPipe task files on first run:
- `pose_landmarker_lite.task`
- `hand_landmarker.task`
- `face_landmarker.task`

## License

MIT
