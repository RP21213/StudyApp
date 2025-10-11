#!/usr/bin/env python3
"""
Test script to verify spaced repetition card updates are working correctly.
This script simulates a card review and verifies the update persists.
"""

import os
import sys
from datetime import datetime, timezone
from google.cloud import firestore

# Initialize Firestore
db = firestore.Client()

def test_card_update():
    """Test updating a spaced repetition card"""
    print("🧪 Testing Spaced Repetition Card Update\n")
    
    # Find a test card
    print("Looking for a test card...")
    cards_ref = db.collection('spaced_repetition_cards').limit(1)
    cards = list(cards_ref.stream())
    
    if not cards:
        print("❌ No cards found in database")
        return False
    
    test_card_doc = cards[0]
    test_card_id = test_card_doc.id
    test_card_data = test_card_doc.to_dict()
    
    print(f"Found card: {test_card_id}")
    print(f"Current state:")
    print(f"  - Repetitions: {test_card_data.get('repetitions', 0)}")
    print(f"  - Interval: {test_card_data.get('interval_days', 1)} days")
    print(f"  - Ease Factor: {test_card_data.get('ease_factor', 2.5)}")
    print(f"  - Last Reviewed: {test_card_data.get('last_reviewed', 'Never')}")
    print()
    
    # Prepare update data
    new_data = {
        'repetitions': int(test_card_data.get('repetitions', 0)) + 1,
        'interval_days': 6,
        'ease_factor': 2.5,
        'last_reviewed': datetime.now(timezone.utc),
        'next_review': datetime.now(timezone.utc),
    }
    
    print(f"Updating card with:")
    print(f"  - Repetitions: {new_data['repetitions']}")
    print(f"  - Interval: {new_data['interval_days']} days")
    print()
    
    # Perform update
    try:
        card_ref = db.collection('spaced_repetition_cards').document(test_card_id)
        card_ref.update(new_data)
        print("✅ Update call succeeded\n")
    except Exception as e:
        print(f"❌ Update failed: {e}")
        return False
    
    # Verify update persisted
    print("Verifying update persisted...")
    updated_doc = card_ref.get()
    if not updated_doc.exists:
        print("❌ Card no longer exists!")
        return False
    
    updated_data = updated_doc.to_dict()
    print(f"Card state after update:")
    print(f"  - Repetitions: {updated_data.get('repetitions', 0)}")
    print(f"  - Interval: {updated_data.get('interval_days', 1)} days")
    print(f"  - Ease Factor: {updated_data.get('ease_factor', 2.5)}")
    print()
    
    # Check if values match
    if updated_data.get('repetitions') == new_data['repetitions']:
        print("✅ Repetitions updated correctly")
    else:
        print(f"❌ Repetitions mismatch! Expected {new_data['repetitions']}, got {updated_data.get('repetitions')}")
        return False
    
    if updated_data.get('interval_days') == new_data['interval_days']:
        print("✅ Interval updated correctly")
    else:
        print(f"❌ Interval mismatch! Expected {new_data['interval_days']}, got {updated_data.get('interval_days')}")
        return False
    
    print("\n🎉 All tests passed! Spaced repetition updates are working correctly.")
    return True

if __name__ == "__main__":
    try:
        success = test_card_update()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

