#!/usr/bin/env python3
"""
Migration script for converting existing flashcards to spaced repetition system
This script uses the same Firebase initialization as the main app
"""

import os
import json
import base64
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime, timezone
from models import SpacedRepetitionCard

def initialize_firebase():
    """Initialize Firebase using the same method as the main app"""
    try:
        # Check if Firebase is already initialized
        if firebase_admin._apps:
            return firebase_admin.get_app()
        
        # Get Firebase credentials from environment variable
        firebase_key_b64 = os.getenv("FIREBASE_KEY_B64")
        if firebase_key_b64:
            # Decode the base64 encoded Firebase key
            firebase_key_json = base64.b64decode(firebase_key_b64).decode('utf-8')
            cred_dict = json.loads(firebase_key_json)
            cred = credentials.Certificate(cred_dict)
        else:
            # Try to load from file
            key_path = os.path.join(os.path.dirname(__file__), "firebase_key.json")
            if os.path.exists(key_path):
                cred = credentials.Certificate(key_path)
            else:
                print("❌ No Firebase credentials found!")
                print("Please set FIREBASE_KEY_B64 environment variable or create firebase_key.json")
                return None
        
        # Initialize Firebase
        bucket_name = os.getenv("FIREBASE_BUCKET_NAME", "ai-study-hub-f3040.firebasestorage.app")
        app = firebase_admin.initialize_app(cred, {'storageBucket': bucket_name})
        print("✅ Firebase initialized successfully!")
        return app
        
    except Exception as e:
        print(f"❌ Firebase initialization failed: {e}")
        return None

def migrate_flashcards():
    """Migrate existing flashcards to spaced repetition system"""
    try:
        # Initialize Firebase
        app = initialize_firebase()
        if not app:
            return False
        
        db = firestore.client()
        
        # Get all flashcard activities
        print("🔍 Searching for flashcard activities...")
        activities_query = db.collection('activities').where('type', '==', 'Flashcards')
        activities = list(activities_query.stream())
        
        if not activities:
            print("ℹ️  No flashcard activities found.")
            return True
        
        print(f"📚 Found {len(activities)} flashcard activities")
        
        migrated_count = 0
        skipped_count = 0
        total_cards = 0
        
        for activity_doc in activities:
            activity_data = activity_doc.to_dict()
            activity_id = activity_doc.id
            
            print(f"\n📖 Processing: {activity_data.get('title', 'Untitled')} (ID: {activity_id})")
            
            # Check if already migrated
            existing_sr_cards = list(db.collection('spaced_repetition_cards')
                                   .where('activity_id', '==', activity_id)
                                   .limit(1)
                                   .stream())
            
            if existing_sr_cards:
                print(f"⏭️  Already migrated, skipping...")
                skipped_count += 1
                continue
            
            # Get cards from activity
            cards = activity_data.get('data', {}).get('cards', [])
            if not cards:
                print(f"⚠️  No cards found in activity, skipping...")
                skipped_count += 1
                continue
            
            print(f"🃏 Found {len(cards)} cards to migrate")
            
            # Create SpacedRepetitionCard for each card
            batch = db.batch()
            for card_index, card in enumerate(cards):
                sr_card_ref = db.collection('spaced_repetition_cards').document()
                sr_card = SpacedRepetitionCard(
                    id=sr_card_ref.id,
                    activity_id=activity_id,
                    card_index=card_index,
                    front=card.get('front', ''),
                    back=card.get('back', ''),
                    ease_factor=2.5,  # Default ease factor
                    interval_days=1,   # Start with 1 day interval
                    repetitions=0,
                    difficulty='medium'
                )
                batch.set(sr_card_ref, sr_card.to_dict())
            
            batch.commit()
            migrated_count += 1
            total_cards += len(cards)
            print(f"✅ Migrated {len(cards)} cards")
        
        print(f"\n🎉 Migration completed!")
        print(f"📊 Summary:")
        print(f"   • Activities migrated: {migrated_count}")
        print(f"   • Activities skipped: {skipped_count}")
        print(f"   • Total cards migrated: {total_cards}")
        
        return True
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_migration_status():
    """Check the current status of spaced repetition cards"""
    try:
        app = initialize_firebase()
        if not app:
            return
        
        db = firestore.client()
        
        # Count spaced repetition cards
        sr_cards = list(db.collection('spaced_repetition_cards').stream())
        print(f"📊 Current spaced repetition cards: {len(sr_cards)}")
        
        if sr_cards:
            # Group by activity
            activities = {}
            for card in sr_cards:
                card_data = card.to_dict()
                activity_id = card_data.get('activity_id')
                if activity_id not in activities:
                    activities[activity_id] = 0
                activities[activity_id] += 1
            
            print(f"📚 Cards by activity:")
            for activity_id, count in activities.items():
                print(f"   • {activity_id}: {count} cards")
        
    except Exception as e:
        print(f"❌ Error checking status: {e}")

if __name__ == "__main__":
    print("🚀 Spaced Repetition Migration Tool")
    print("=" * 50)
    
    # Check current status
    print("📊 Current Status:")
    check_migration_status()
    
    print("\n" + "=" * 50)
    
    # Run migration
    success = migrate_flashcards()
    
    if success:
        print("\n📊 Final Status:")
        check_migration_status()
        
        print("\n✅ Migration completed successfully!")
        print("🎯 Next steps:")
        print("   1. Test the spaced repetition interface")
        print("   2. Check the dashboard for due cards")
        print("   3. Start your first review session")
    else:
        print("\n❌ Migration failed. Please check the error messages above.")
