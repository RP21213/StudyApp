#!/usr/bin/env python3
"""
Test Firebase connection and run migration
"""

import os
import json
import base64
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime, timezone
from models import SpacedRepetitionCard

def test_firebase_connection():
    """Test if Firebase is working"""
    try:
        # Check if Firebase is already initialized
        if firebase_admin._apps:
            app = firebase_admin.get_app()
        else:
            # Try to initialize Firebase
            firebase_key_b64 = os.getenv("FIREBASE_KEY_B64")
            if firebase_key_b64:
                firebase_key_json = base64.b64decode(firebase_key_b64).decode('utf-8')
                cred_dict = json.loads(firebase_key_json)
                cred = credentials.Certificate(cred_dict)
            else:
                print("❌ No Firebase credentials found!")
                return False
            
            bucket_name = os.getenv("FIREBASE_BUCKET_NAME", "ai-study-hub-f3040.firebasestorage.app")
            app = firebase_admin.initialize_app(cred, {'storageBucket': bucket_name})
        
        db = firestore.client()
        
        # Test basic connection
        print("🔍 Testing Firebase connection...")
        
        # Try to read a simple collection
        try:
            # Test reading activities
            activities = list(db.collection('activities').limit(1).stream())
            print(f"✅ Firebase connection successful! Found {len(activities)} activities")
            
            # Test reading hubs
            hubs = list(db.collection('hubs').limit(1).stream())
            print(f"✅ Found {len(hubs)} hubs")
            
            return True
            
        except Exception as e:
            print(f"❌ Error reading from Firestore: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Firebase connection failed: {e}")
        return False

def run_migration():
    """Run the migration"""
    try:
        db = firestore.client()
        
        print("🚀 Starting migration...")
        
        # Get all flashcard activities
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

if __name__ == "__main__":
    print("🧪 Firebase Connection Test & Migration")
    print("=" * 50)
    
    # Test connection first
    if test_firebase_connection():
        print("\n" + "=" * 50)
        print("🚀 Running Migration...")
        success = run_migration()
        
        if success:
            print("\n✅ Migration completed successfully!")
            print("🎯 Next steps:")
            print("   1. Add Firestore indexes (see instructions above)")
            print("   2. Test the spaced repetition interface")
            print("   3. Check the dashboard for due cards")
        else:
            print("\n❌ Migration failed. Please check the error messages above.")
    else:
        print("\n❌ Cannot proceed without Firebase connection.")
        print("Please check your Firebase credentials and try again.")
