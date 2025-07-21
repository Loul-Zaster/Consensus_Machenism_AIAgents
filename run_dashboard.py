#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Wrapper script to run the Consensus Mechanism AI Agents dashboard
with correct Python path configuration.
"""

import os
import sys
import subprocess

# Get the absolute path to the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Ensure the current directory is in the Python path
sys.path.insert(0, current_dir)

# Check if agent.py exists in the current directory
if not os.path.exists(os.path.join(current_dir, "agent.py")):
    print("Error: agent.py not found in the current directory.")
    print("The dashboard requires agent.py for the evaluation_ragas module.")
    sys.exit(1)

print("Starting dashboard with correct Python path...")

# Run Streamlit with the dashboard script
try:
    result = subprocess.run(["streamlit", "run", os.path.join(current_dir, "dashboard.py")], 
                           check=True)
    sys.exit(result.returncode)
except subprocess.CalledProcessError as e:
    print(f"Error running dashboard: {e}")
    sys.exit(e.returncode)
except FileNotFoundError:
    print("Error: Could not find the streamlit executable.")
    print("Make sure Streamlit is installed: pip install streamlit")
    sys.exit(1) 