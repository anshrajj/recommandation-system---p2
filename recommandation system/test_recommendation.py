#!/usr/bin/env python3
"""
Test script for the recommendation system
"""

import sys
import subprocess

def test_content_based():
    """Test content-based recommendation system"""
    print("Testing Content-Based Recommendation System...")
    
    # Test with different genre inputs
    test_cases = [
        "Action, Sci-Fi",
        "Drama, Romance",
        "Comedy",
        "Horror, Thriller"
    ]
    
    for genres in test_cases:
        print(f"\nTesting genres: {genres}")
        try:
            result = subprocess.run(
                [sys.executable, "app.py"],
                input=genres + "\nno\n",
                text=True,
                capture_output=True,
                timeout=30
            )
            print("Output:", result.stdout[-500:])  # Show last 500 chars
            if result.stderr:
                print("Errors:", result.stderr)
        except subprocess.TimeoutExpired:
            print("Test timed out")
        except Exception as e:
            print(f"Error: {e}")

def test_imports():
    """Test if all required packages are installed"""
    print("Testing package imports...")
    packages = ['pandas', 'sklearn', 'numpy']
    
    for package in packages:
        try:
            __import__(package)
            print(f"✓ {package} imported successfully")
        except ImportError as e:
            print(f"✗ {package} import failed: {e}")

if __name__ == "__main__":
    test_imports()
    test_content_based()
