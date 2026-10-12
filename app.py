"""
MoodMentor — Unified Enterprise AI Platform Entrypoint
Production Cloud & Hugging Face Spaces Runner
"""

import os
import sys

# Guarantee repository root, milestone3, and milestone4 are in sys.path
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
M3_DIR = os.path.join(REPO_ROOT, "milestone3")
M4_DIR = os.path.join(REPO_ROOT, "milestone4")
DASHBOARD_DIR = os.path.join(M4_DIR, "dashboard")

for path in [REPO_ROOT, M4_DIR, DASHBOARD_DIR, M3_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

import runpy

target_app = os.path.join(DASHBOARD_DIR, "app.py")
runpy.run_path(target_app, run_name="__main__")
