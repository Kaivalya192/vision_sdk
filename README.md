# Vision SDK

## Overview
Vision SDK is a Python-based application designed for vision-guided sorting using advanced computer vision techniques. It provides a graphical user interface (GUI) for real-time video processing, object detection, and template matching. The application is built using PyQt5 and OpenCV, making it highly interactive and efficient for industrial use cases.

## Features
- **Real-Time Video Processing**: Capture and process video frames from a webcam.
- **Template Matching**: Detect and manage up to 5 object templates with adjustable parameters.
- **Graphical User Interface**: Intuitive GUI for controlling video processing, object detection, and publishing results.
- **UDP Publisher**: Publish detection results in JSON format over UDP.
- **Customizable Processing**: Options for frame skipping, resolution adjustment, and preprocessing (e.g., CLAHE and flipping).

## Installation

### Prerequisites
- Python 3.8+
- Pip
- A webcam for video input

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/Kaivalya192/vision_sdk.git
   cd vision_sdk
   ```
2. Install the SDK in editable mode:
   ```bash
   pip install -e .
   ```

## Usage

### Running the Application
To start the application, run the following command:
```bash
python app/main.py [camera_index]
```
- Replace `[camera_index]` with the index of your webcam (default is `0`).

### Key Functionalities
- **Capture ROI**: Select a region of interest (ROI) for template matching.
- **Clear Active**: Clear the currently active template slot.
- **Rotate View**: Rotate the video feed by 90° increments.
- **Publish JSON**: Enable or disable publishing detection results over UDP.

### GUI Components
- **Video Feed**: Displays the live video feed with overlays for detected objects.
- **Publisher Settings**: Configure IP and port for UDP publishing.
- **Processing Settings**: Adjust processing width, frame skipping, and preprocessing options.
- **Template Manager**: Manage object templates, including enabling/disabling slots and setting maximum instances.
- **Detections Table**: View detailed information about detected objects.

## Project Structure
```
vision_sdk/
├── app/
│   └── main.py          # Main application entry point
├── dexsdk/
│   ├── detection_multi.py
│   ├── detection.py
│   ├── utils.py
│   ├── video.py
│   └── camera/
│       └── webcam.py
│   └── net/
│       └── publisher.py
│   └── ui/
│       └── video_label.py
├── requirements.txt      # Python dependencies
├── setup.py              # Package setup
├── pyproject.toml        # Project metadata
└── README.md             # Project documentation
```

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
- Built with PyQt5 and OpenCV.
- Inspired by industrial vision-guided sorting systems.

### JSON Payload Format

The application publishes detection results in the following JSON format:

```json
{
  "version": "1.0",
  "sdk": {
    "name": "dexsdk",
    "module": "SIFT",
    "soft_color_gate": true
  },
  "session": {
    "session_id": "ab12cd34",
    "frame_id": 1234
  },
  "timestamp_ms": 1724730000000,
  "camera": {
    "source_index": 0,
    "proc_width": 640,
    "proc_height": 360,
    "coordinate_space": "processed",
    "units": "pixels"
  },
  "result": {
    "counts": {
      "objects": 2,
      "detections": 3
    },
    "objects": [
      {
        "object_id": 0,
        "name": "Obj1",
        "template_size": [180, 120],
        "detections": [
          {
            "instance_id": 0,
            "score": 0.73,
            "inliers": 42,
            "pose": {
              "x": 245.1,
              "y": 132.4,
              "theta_deg": -12.5,
              "x_scale": 0.98,
              "y_scale": 1.01
            },
            "center": [252.0, 140.5],
            "quad": [
              [165.2, 80.1],
              [330.1, 76.3],
              [334.4, 196.0],
              [168.8, 199.2]
            ],
            "color": {
              "bhattacharyya": 0.28,
              "correlation": 0.42,
              "deltaE": 22.3
            }
          }
        ]
      }
    ]
  },
  "timing_ms": {
    "detect_ms": 7.9,
    "draw_ms": 1.2,
    "total_ms": 9.1
  }
}
```

#### Key Fields:
- **`version`**: Payload version.
- **`sdk`**: Information about the SDK, including the algorithm used and color gating settings.
- **`session`**: Identifies the session and frame.
- **`timestamp_ms`**: Timestamp in Unix epoch milliseconds.
- **`camera`**: Camera and processed frame details.
- **`result`**: Detection results, including object counts and detailed information for each detected object.
- **`timing_ms`**: Timings for various processing stages.

