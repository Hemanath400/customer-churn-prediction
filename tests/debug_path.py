# 📁 tests/debug_path.py
import sys
from pathlib import Path

print("🔍 Current working directory:", Path.cwd())
print("🔍 This file location:", Path(__file__).parent)
print("🔍 Project root:", Path(__file__).parent.parent)
print("\n📋 Python path:")
for i, path in enumerate(sys.path):
    print(f"  {i}: {path}")

# Check if simple_api.py exists
api_path = Path(__file__).parent.parent / "deployment" / "simple_api.py"
print(f"\n✅ simple_api.py exists: {api_path.exists()} at {api_path}")