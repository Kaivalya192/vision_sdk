# Dexsent Vision SDK

A minimal Python SDK for Dexsent’s wrist‑mounted pick‑and‑place vision module (RealSense‑based).

## Features

- **Camera Connection & Streaming**  
  Connect to Intel RealSense D435i, configure RGB + Depth streams, and grab synchronized frames.

- **Calibration**  
  Intrinsic (lens) and extrinsic (camera↔gripper) calibration using chessboard or AprilTag patterns.

- **Object Teaching & Storage**  
  Teach parts via template, depth‑map profile, or YOLO model; assets stored under `objects/<name>/`.

- **Model Training Helper**  
  Kick off a YOLO training run (`train_model`) and export ONNX for inference.

- **Real‑time Detection & 6‑D Pose**  
  `visual_pick(name)` returns `{x,y,z,yaw}` in mm/degrees.

- **Conveyor‑Belt Speed Estimation**  
  “Vision-in-Motion” speed estimate (mm/s) without external encoder.

- **Presence Queries**  
  `object_visible(name)` → Boolean for PLC/robot logic.

- **Logging & Snapshots**  
  Save raw or annotated frames plus pose metadata.
