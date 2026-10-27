"""
run_demo.py
===========
One-Click Launcher for PediaMorph AI Local Demonstration.
Starts the FastAPI backend and automatically opens the interactive triage UI in your default browser.

Usage:
    python run_demo.py
"""

import sys
import time
import webbrowser
import threading
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent

def open_browser():
    time.sleep(1.5)
    url = "http://127.0.0.1:8000/docs"
    print(f"\n[API ENGINE] Opening interactive Swagger API tester at: {url}")
    print("[API ENGINE] Mobile developers can test endpoints directly with already-clicked photos.\n")
    webbrowser.open(url)

if __name__ == "__main__":
    try:
        import uvicorn
    except ImportError:
        print("[ERROR] uvicorn is not installed. Run: pip install uvicorn")
        sys.exit(1)

    print("=" * 75)
    print("  PEDIAMORPH AI - FRONTLINE ARM MORPHOMETRICS & DUAL-TIER CLASSIFIER")
    print("=" * 75)
    print("  * Phase 1: Natural Child Left-Arm Photo (No Checkerboard Needed)")
    print("  * Phase 2: Tier 1 Mobile Ensemble (40% LDA + 60% XGBoost, 3-Tier Traffic Light)")
    print("  * Phase 2: Tier 2 Cloud Deep Learning (17-Node EGAT Attention Heatmaps)")
    print("  * Escalation: Multimodal Secondary Resolver (>89.7% Accuracy)")
    print("  * Frontline WhatsApp Bot: GET/POST /api/v1/whatsapp-webhook")
    print("=" * 75)

    threading.Thread(target=open_browser, daemon=True).start()

    # Run Uvicorn server with auto-reload enabled
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        app_dir=str(CURRENT_DIR),
        log_level="info",
        reload=True
    )
