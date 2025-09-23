#!/usr/bin/env python3
"""
Test migration script - run this to test the migration locally
"""

import requests
import json

def test_migration():
    """Test the migration endpoint"""
    try:
        # First, let's check if the endpoint exists
        print("🔍 Testing migration endpoint...")
        
        # Try to access the endpoint
        response = requests.post(
            "http://localhost:5000/admin/migrate_flashcards_to_spaced_repetition",
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 302:
            print("🔄 Redirect detected - authentication required")
            print("Please login first at http://localhost:5000")
        elif response.status_code == 200:
            try:
                data = response.json()
                print("✅ Migration Response:")
                print(json.dumps(data, indent=2))
            except:
                print("📄 HTML Response (likely login page):")
                print(response.text[:500] + "..." if len(response.text) > 500 else response.text)
        else:
            print(f"❌ Unexpected status code: {response.status_code}")
            print("Response:", response.text[:500])
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to localhost:5000")
        print("Make sure your Flask app is running")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_migration()
