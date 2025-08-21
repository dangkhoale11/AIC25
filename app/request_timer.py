# test_health.py
import requests

try:
    resp = requests.get("http://localhost:8000/health", timeout=180)
    print(resp.json())
except Exception as e:
    print(f"Failed to connect: {e}")
