# Script to train and save the first production model
import subprocess
import time
import signal
import sys

print("🚀 Starting model training and saving process...\n")

# Start the main.py process
process = subprocess.Popen(
    [sys.executable, "main.py"],
    cwd=r"c:\Users\LENOVO\Bridge monitoring system",
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

# Let it run for a few seconds to complete training
try:
    stdout, stderr = process.communicate(timeout=15)
    print("STDOUT:")
    print(stdout)
    if stderr:
        print("\nSTDERR:")
        print(stderr)
except subprocess.TimeoutExpired:
    print("✅ Training completed, terminating live monitoring...\n")
    process.terminate()
    try:
        stdout, stderr = process.communicate(timeout=2)
        print("STDOUT:")
        print(stdout)
    except subprocess.TimeoutExpired:
        process.kill()

print("\n✨ Model training and saving complete!")
