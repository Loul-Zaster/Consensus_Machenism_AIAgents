#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to run the Consensus Mechanism AI Agents application
Runs the unified dashboard for both the cancer analysis system and the RAG evaluation tool.
"""

import os
import sys
import subprocess
import time
import webbrowser
import threading
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def run_streamlit(script_path, port=None):
    """Run Streamlit application with optional port"""
    cmd = ["streamlit", "run", script_path]
    if port:
        cmd.extend(["--server.port", str(port)])
    
    # Display application launch message
    app_name = "Consensus Mechanism AI Agents Dashboard"
    url = f"http://localhost:{port}" if port else "http://localhost:8501"
    print(f"\n⭐ Launching {app_name} at {url}")

    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def check_api_keys():
    """Check API keys and display status messages"""
    io_api_key = os.getenv("IOINTELLIGENCE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    serper_api_key = os.getenv("SERPER_API_KEY")
    
    if io_api_key:
        print("✅ IOINTELLIGENCE_API_KEY: Configured")
        print("🔹 Using IO.net Intelligence API as primary API")
    elif openai_api_key:
        print("✅ OPENAI_API_KEY: Configured")
        print("🔹 Using OpenAI API as fallback API")
    else:
        print("❌ LLM API KEY: Not found! Please configure IOINTELLIGENCE_API_KEY or OPENAI_API_KEY in .env file")
        
    if serper_api_key:
        print("✅ SERPER_API_KEY: Configured (for web search feature)")
    else:
        print("⚠️ SERPER_API_KEY: Not found (web search will use sample data)")
    
    if io_api_key:
        print("✅ Translation Agent: Ready (using IO Intelligence API)")
    else:
        print("⚠️ Translation Agent: Not available (requires IO Intelligence API key)")

def open_browser(port, delay=1.5):
    """Open browser to localhost address with specified port after a delay"""
    time.sleep(delay)  # Wait for Streamlit application to start
    webbrowser.open(f"http://localhost:{port}")

def main():
    # Check environment variables
    print("\n🔍 Checking configuration...")
    check_api_keys()
    
    # Determine absolute path for dashboard file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dashboard_script = os.path.join(current_dir, "dashboard.py")
    
    # Ensure dashboard file exists
    if not os.path.exists(dashboard_script):
        print(f"\n❌ Error: File not found {dashboard_script}")
        return
    
    try:
        # Run integrated dashboard
        dashboard_process = run_streamlit(dashboard_script, 8501)
        
        # Open browser to dashboard
        threading.Thread(target=open_browser, args=(8501,)).start()
        
        # Display instructions
        print("\n🚀 Dashboard launched successfully!")
        print("📊 Consensus Mechanism AI Agents: http://localhost:8501")
        print("\nPress Ctrl+C to stop...")
        
        # Wait until user presses Ctrl+C
        dashboard_process.wait()
            
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping application...")
        dashboard_process.terminate()
        print("✅ Stopped successfully!")

if __name__ == "__main__":
    main() 