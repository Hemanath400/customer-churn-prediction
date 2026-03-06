# 📁 tests/conftest.py - Add this file
import pytest
from pathlib import Path

@pytest.fixture
def project_root():
    """Return the project root directory"""
    return Path(__file__).parent.parent

@pytest.fixture
def notebooks_path(project_root):
    """Return the notebooks path"""
    return project_root / "notebooks"

@pytest.fixture
def models_path(project_root):
    """Return the models path"""
    # Try both locations
    models_dir = project_root / "models"
    if models_dir.exists():
        return models_dir
    return project_root / "notebooks"