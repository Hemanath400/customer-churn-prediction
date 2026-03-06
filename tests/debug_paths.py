# 📁 tests/debug_paths.py
from pathlib import Path

print("🔍 PATH DEBUGGING")
print("=" * 50)

# Current file location
current_file = Path(__file__)
print(f"Current file: {current_file}")

# Tests folder
tests_folder = current_file.parent
print(f"Tests folder: {tests_folder}")

# Project root (parent of tests)
project_root = tests_folder.parent
print(f"Project root: {project_root}")

# Notebooks folder
notebooks_folder = project_root / "notebooks"
print(f"Notebooks folder: {notebooks_folder}")

# Check if notebooks exists
print(f"\n📁 Notebooks folder exists: {notebooks_folder.exists()}")

if notebooks_folder.exists():
    print("\n📄 Files in notebooks:")
    for f in notebooks_folder.glob("*"):
        print(f"  - {f.name} ({f.stat().st_size} bytes)")
else:
    print(f"❌ Notebooks folder not found at {notebooks_folder}")
    
    # Try alternative locations
    alt_locations = [
        project_root,
        project_root.parent / "notebooks",
        Path("../notebooks").resolve(),
    ]
    
    print("\n🔍 Checking alternative locations:")
    for alt in alt_locations:
        print(f"  {alt} exists: {alt.exists()}")
        if alt.exists():
            print(f"    Files: {[f.name for f in alt.glob('*.pkl')]}")