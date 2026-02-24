"""
Launch Full Real-Time Dark Web Intelligence Platform
Starts both the detector and dashboard
"""

import subprocess
import time
import sys
from pathlib import Path

def start_detector():
    """Start the real-time detector in background"""
    return subprocess.Popen(
        [sys.executable, 'src/streaming/real_time_pipeline_v3.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

def start_dashboard():
    """Start the web dashboard"""
    return subprocess.Popen(
        [sys.executable, 'src/streaming/intelligence_dashboard.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
        universal_newlines=True
    )

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🕵️  DARK WEB INTELLIGENCE PLATFORM - FULL DEPLOYMENT")
    print("="*70 + "\n")
    
    print("📡 Starting threat detection engine...")
    detector_process = start_detector()
    time.sleep(2)
    
    print("🌐 Starting real-time intelligence dashboard...")
    dashboard_process = start_dashboard()
    time.sleep(3)
    
    print("\n" + "="*70)
    print("✅ PLATFORM OPERATIONAL")
    print("="*70)
    print("\n📊 Dashboard: http://localhost:5000")
    print("🔴 Real-time monitoring: ACTIVE")
    print("⚡ WebSocket updates: ENABLED")
    print("\nOpen your browser to http://localhost:5000")
    print("\nPress Ctrl+C to shutdown...\n")
    
    try:
        # Stream dashboard output to console
        for line in dashboard_process.stdout:
            print(line, end='')
    except KeyboardInterrupt:
        print("\n\n⏸️  Shutting down platform...")
        detector_process.terminate()
        dashboard_process.terminate()
        print("✅ Shutdown complete\n")
