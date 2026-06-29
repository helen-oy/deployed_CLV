#!/usr/bin/env python3
"""
Local test script for CLV API endpoints
Run after starting the API with: python scripts/run_api.py
"""

import requests
import json
import sys
from typing import Dict, Any

BASE_URL = "http://localhost:8000"

def print_response(title: str, response: requests.Response):
    """Pretty print API response."""
    print(f"\n{'='*60}")
    print(f"✅ {title}")
    print(f"{'='*60}")
    print(f"Status Code: {response.status_code}")
    try:
        print(json.dumps(response.json(), indent=2))
    except:
        print(response.text)

def test_health_check():
    """Test health check endpoint."""
    try:
        response = requests.get(f"{BASE_URL}/health")
        print_response("Health Check", response)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_root():
    """Test root endpoint."""
    try:
        response = requests.get(f"{BASE_URL}/")
        print_response("Root Endpoint", response)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Root endpoint failed: {e}")
        return False

def test_predict_clv():
    """Test CLV prediction endpoint."""
    try:
        payload = {
            "customers": [
                {
                    "customer_id": "17850",
                    "recency": 30,
                    "frequency": 50,
                    "monetary": 4000.0
                },
                {
                    "customer_id": "13802",
                    "recency": 15,
                    "frequency": 100,
                    "monetary": 8000.0
                }
            ]
        }
        response = requests.post(
            f"{BASE_URL}/predict/clv",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        print_response("CLV Prediction", response)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ CLV prediction failed: {e}")
        return False

def test_get_recommendations():
    """Test recommendations endpoint."""
    try:
        payload = {
            "customer_id": "17850",
            "clv_segment": "Loyal",
            "churn_risk": "Low"
        }
        response = requests.post(
            f"{BASE_URL}/recommend",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        print_response("Recommendations", response)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Recommendations failed: {e}")
        return False

def test_segment_summary():
    """Test segment summary endpoint."""
    try:
        response = requests.get(f"{BASE_URL}/segments/summary")
        print_response("Segment Summary", response)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Segment summary failed: {e}")
        return False

def test_top_customers():
    """Test top customers endpoint."""
    try:
        response = requests.get(f"{BASE_URL}/customers/top/5")
        print_response("Top 5 Priority Customers", response)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Top customers failed: {e}")
        return False

def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("CLV Prediction API - Local Test Suite")
    print("="*60)
    print(f"Testing API at: {BASE_URL}\n")

    # Check if API is running
    try:
        requests.get(f"{BASE_URL}/health", timeout=2)
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Cannot connect to API at", BASE_URL)
        print("Make sure the API is running: python scripts/run_api.py")
        sys.exit(1)

    tests = [
        ("Health Check", test_health_check),
        ("Root Endpoint", test_root),
        ("CLV Prediction", test_predict_clv),
        ("Recommendations", test_get_recommendations),
        ("Segment Summary", test_segment_summary),
        ("Top Customers", test_top_customers),
    ]

    results = []
    for name, test_func in tests:
        print(f"\nRunning: {name}...")
        result = test_func()
        results.append((name, result))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")

    print(f"\nTotal: {passed}/{total} tests passed")
    print("="*60)

    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
