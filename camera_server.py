"""
Deprecated compatibility entrypoint.

The main Django app no longer uses this HTTP camera hop. The live detection
pipeline starts a single DirectShow camera manager from surveillance.apps and
streams frames directly from shared memory.
"""

if __name__ == "__main__":
    print("camera_server.py is deprecated and unused by the main Django app.")
    print("Run start.ps1 or python run_server.py instead.")

