#!/usr/bin/env python3
"""
Thin entrypoint for the modular VisionServer.

Run:
  pip install websockets==12.* opencv-python numpy
  python app/vision_server.py
"""

from app.server.server import main


if __name__ == "__main__":
    main()

