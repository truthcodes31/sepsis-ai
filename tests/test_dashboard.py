# test_dashboard.py
import pytest
import subprocess

def test_dashboard():
    process = subprocess.Popen(["streamlit", "run", "src/dashboard.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process is not None