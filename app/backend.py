#!/usr/bin/env python3
"""
Thin entrypoint for VisionServer or the Calibrate+Measure UI.

Usage:
  python app/backend.py              # start VisionServer (default)
  python app/backend.py measure      # start unified Calibrate+Measure UI
"""

import sys

def _run_server():
    from app.server.server import main as server_main
    server_main()

def _run_measure_ui():
    from app.measure_ui import main as ui_main
    ui_main()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() in {"measure", "measure-ui"}:
        _run_measure_ui()
    else:
        _run_server()
