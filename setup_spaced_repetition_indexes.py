#!/usr/bin/env python3
"""
Database index setup script for Spaced Repetition system.
Run this script to create the necessary Firestore indexes for optimal performance.

Required indexes for spaced repetition queries:
1. spaced_repetition_cards: activity_id (for getting cards by activity)
2. spaced_repetition_cards: next_review (for getting due cards)
3. spaced_repetition_cards: activity_id + next_review (compound for due cards by activity)
4. review_sessions: user_id + hub_id (for getting user sessions)
5. review_sessions: user_id + started_at (for getting recent sessions)
6. user_spaced_repetition_settings: user_id (for getting user settings)
"""

import firebase_admin
from firebase_admin import credentials, firestore
import json
import os

def setup_indexes():
    """Create Firestore indexes for spaced repetition system"""
    
    # Initialize Firebase Admin SDK
    if not firebase_admin._apps:
        # Try to get credentials from environment or use default
        try:
            cred = credentials.Certificate('path/to/serviceAccountKey.json')
            firebase_admin.initialize_app(cred)
        except:
            # Use default credentials (for Google Cloud environments)
            firebase_admin.initialize_app()
    
    db = firestore.client()
    
    print("Setting up Firestore indexes for Spaced Repetition system...")
    
    # Note: Firestore indexes are typically created through the Firebase Console
    # or using the Firebase CLI. This script provides the index specifications
    # that need to be created.
    
    indexes_to_create = [
        {
            "collection": "spaced_repetition_cards",
            "fields": [
                {"field": "activity_id", "order": "ASCENDING"}
            ],
            "description": "Index for querying cards by activity"
        },
        {
            "collection": "spaced_repetition_cards", 
            "fields": [
                {"field": "next_review", "order": "ASCENDING"}
            ],
            "description": "Index for querying due cards"
        },
        {
            "collection": "spaced_repetition_cards",
            "fields": [
                {"field": "activity_id", "order": "ASCENDING"},
                {"field": "next_review", "order": "ASCENDING"}
            ],
            "description": "Compound index for due cards by activity"
        },
        {
            "collection": "review_sessions",
            "fields": [
                {"field": "user_id", "order": "ASCENDING"},
                {"field": "hub_id", "order": "ASCENDING"}
            ],
            "description": "Index for user sessions by hub"
        },
        {
            "collection": "review_sessions",
            "fields": [
                {"field": "user_id", "order": "ASCENDING"},
                {"field": "started_at", "order": "DESCENDING"}
            ],
            "description": "Index for recent user sessions"
        },
        {
            "collection": "user_spaced_repetition_settings",
            "fields": [
                {"field": "user_id", "order": "ASCENDING"}
            ],
            "description": "Index for user settings"
        }
    ]
    
    print("\nRequired Firestore Indexes:")
    print("=" * 50)
    
    for i, index in enumerate(indexes_to_create, 1):
        print(f"\n{i}. Collection: {index['collection']}")
        print(f"   Description: {index['description']}")
        print("   Fields:")
        for field in index['fields']:
            print(f"     - {field['field']} ({field['order']})")
    
    print("\n" + "=" * 50)
    print("IMPORTANT: These indexes need to be created in the Firebase Console")
    print("or using the Firebase CLI. This script only shows the specifications.")
    print("\nTo create indexes using Firebase CLI:")
    print("1. Install Firebase CLI: npm install -g firebase-tools")
    print("2. Login: firebase login")
    print("3. Initialize: firebase init firestore")
    print("4. Create firestore.indexes.json with the above specifications")
    print("5. Deploy: firebase deploy --only firestore:indexes")
    
    # Create a sample firestore.indexes.json file
    firestore_indexes = {
        "indexes": []
    }
    
    for index in indexes_to_create:
        firestore_index = {
            "collectionGroup": index['collection'],
            "queryScope": "COLLECTION",
            "fields": index['fields']
        }
        firestore_indexes["indexes"].append(firestore_index)
    
    # Write the indexes file
    with open('firestore.indexes.json', 'w') as f:
        json.dump(firestore_indexes, f, indent=2)
    
    print(f"\n✅ Created firestore.indexes.json file")
    print("You can now deploy these indexes using: firebase deploy --only firestore:indexes")
    
    return True

if __name__ == "__main__":
    setup_indexes()
