# 📁 tests/find_models.py
from pathlib import Path

print("🔍 SEARCHING FOR MODEL FILES")
print("=" * 60)

root = Path.cwd()
print(f"Searching in: {root}\n")

# Find all .pkl files
pkl_files = list(root.rglob("*.pkl"))

if pkl_files:
    print(f"Found {len(pkl_files)} .pkl files:")
    for f in pkl_files:
        size = f.stat().st_size / (1024*1024)  # Size in MB
        print(f"  ✅ {f.relative_to(root)} ({size:.2f} MB)")
else:
    print("❌ No .pkl files found")
    
    # Check specific locations
    print("\nChecking specific locations:")
    locations = [
        root / "models",
        root / "notebooks",
        root / "deployment",
        root.parent / "churn_new" / "notebooks",
        Path("C:/Users/HEMANATH/Desktop/churn_new/notebooks")
    ]
    
    for loc in locations:
        if loc.exists():
            print(f"\n📁 {loc}:")
            for f in loc.glob("*.pkl"):
                print(f"  ✅ {f.name}")