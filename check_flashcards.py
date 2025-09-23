#!/usr/bin/env python3
"""
Check existing flashcards before migration
"""

import firebase_admin
from firebase_admin import credentials, firestore
import os

def check_existing_flashcards():
    """Check what flashcards exist in the database"""
    try:
        # Initialize Firebase
        cred = credentials.Certificate('cors.json')
        firebase_admin.initialize_app(cred)
        db = firestore.client()

        # Get all flashcard activities
        activities_query = db.collection('activities').where('type', '==', 'Flashcards')
        activities = list(activities_query.stream())

        print('📚 Existing Flashcard Activities:')
        print('=' * 50)

        total_cards = 0
        for activity_doc in activities:
            activity_data = activity_doc.to_dict()
            
            # Debug: print the structure
            print(f"Activity ID: {activity_doc.id}")
            print(f"Activity data keys: {list(activity_data.keys())}")
            
            # Check if data exists and is a dict
            if 'data' in activity_data and isinstance(activity_data['data'], dict):
                cards = activity_data['data'].get('cards', [])
                total_cards += len(cards)
                
                print(f'Title: {activity_data.get("title", "Untitled")}')
                print(f'Hub ID: {activity_data.get("hub_id", "Unknown")}')
                print(f'Cards: {len(cards)}')
                print(f'Created: {activity_data.get("created_at", "Unknown")}')
            else:
                print(f'Title: {activity_data.get("title", "Untitled")}')
                print(f'Hub ID: {activity_data.get("hub_id", "Unknown")}')
                print(f'Cards: 0 (no data field or invalid structure)')
                print(f'Created: {activity_data.get("created_at", "Unknown")}')
            
            print('-' * 30)

        print(f'\nTotal flashcard activities: {len(activities)}')
        print(f'Total cards: {total_cards}')
        
        # Check if any cards are already migrated
        sr_cards_query = db.collection('spaced_repetition_cards').limit(1)
        sr_cards = list(sr_cards_query.stream())
        
        if sr_cards:
            print(f'\n⚠️  WARNING: {len(sr_cards)} spaced repetition cards already exist!')
            print('Migration may skip existing cards.')
        else:
            print('\n✅ No spaced repetition cards found. Ready for migration.')
            
        return len(activities), total_cards
        
    except Exception as e:
        print(f'Error checking flashcards: {e}')
        import traceback
        traceback.print_exc()
        return 0, 0

if __name__ == "__main__":
    check_existing_flashcards()