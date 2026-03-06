# 📁 tests/test_data.py
import pytest
import pandas as pd
from pathlib import Path

# Get the correct path - notebooks folder is in the same directory as tests parent
NOTEBOOKS_PATH = Path(__file__).parent.parent / "notebooks"
print(f"🔍 Looking for data in: {NOTEBOOKS_PATH}")

def test_data_exists():
    """Test data file exists"""
    data_path = NOTEBOOKS_PATH / "telco_churn_cleaned.csv"
    print(f"Checking: {data_path}")
    assert data_path.exists(), f"Data not found at {data_path}"

def test_data_loads():
    """Test data loads"""
    data_path = NOTEBOOKS_PATH / "telco_churn_cleaned.csv"
    df = pd.read_csv(data_path)
    assert len(df) > 0
    print(f"✅ Data loaded: {len(df)} rows")

def test_churn_column():
    """Test churn column has correct values"""
    data_path = NOTEBOOKS_PATH / "telco_churn_cleaned.csv"
    df = pd.read_csv(data_path)
    assert 'Churn' in df.columns
    assert set(df['Churn'].unique()).issubset({'Yes', 'No'})

if __name__ == "__main__":
    pytest.main([__file__, "-v"])