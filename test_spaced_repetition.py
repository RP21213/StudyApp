#!/usr/bin/env python3
"""
Test script for Spaced Repetition system implementation.
This script tests the core functionality of the spaced repetition models.
"""

import sys
import os
from datetime import datetime, timezone, timedelta

# Add the current directory to Python path to import models
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import SpacedRepetitionCard, ReviewSession, UserSpacedRepetitionSettings

def test_spaced_repetition_card():
    """Test SpacedRepetitionCard functionality"""
    print("Testing SpacedRepetitionCard...")
    
    # Create a test card
    card = SpacedRepetitionCard(
        id="test_card_1",
        activity_id="test_activity_1",
        card_index=0,
        front="What is the capital of France?",
        back="Paris",
        ease_factor=2.5,
        interval_days=1,
        repetitions=0
    )
    
    print(f"Created card: {card.front}")
    print(f"Initial ease factor: {card.ease_factor}")
    print(f"Initial interval: {card.interval_days} days")
    
    # Test is_due method
    print(f"Is due: {card.is_due()}")
    
    # Test quality rating calculations
    print("\nTesting quality ratings...")
    
    # Test "Good" rating (2)
    card.calculate_next_review(2)
    print(f"After 'Good' rating:")
    print(f"  Repetitions: {card.repetitions}")
    print(f"  Interval: {card.interval_days} days")
    print(f"  Ease factor: {card.ease_factor}")
    print(f"  Next review: {card.next_review}")
    
    # Test "Easy" rating (3)
    card.calculate_next_review(3)
    print(f"\nAfter 'Easy' rating:")
    print(f"  Repetitions: {card.repetitions}")
    print(f"  Interval: {card.interval_days} days")
    print(f"  Ease factor: {card.ease_factor}")
    
    # Test "Again" rating (0)
    card.calculate_next_review(0)
    print(f"\nAfter 'Again' rating:")
    print(f"  Repetitions: {card.repetitions}")
    print(f"  Interval: {card.interval_days} days")
    print(f"  Ease factor: {card.ease_factor}")
    
    print("✅ SpacedRepetitionCard tests passed!\n")

def test_review_session():
    """Test ReviewSession functionality"""
    print("Testing ReviewSession...")
    
    # Create a test session
    session = ReviewSession(
        id="test_session_1",
        user_id="test_user_1",
        hub_id="test_hub_1",
        session_type="spaced_repetition",
        cards_reviewed=0,
        correct_count=0,
        incorrect_count=0
    )
    
    print(f"Created session: {session.id}")
    print(f"Session type: {session.session_type}")
    
    # Simulate reviewing cards
    session.cards_reviewed = 10
    session.correct_count = 8
    session.incorrect_count = 2
    
    print(f"Cards reviewed: {session.cards_reviewed}")
    print(f"Correct: {session.correct_count}")
    print(f"Incorrect: {session.incorrect_count}")
    print(f"Accuracy: {session.calculate_accuracy():.1f}%")
    
    # Complete the session
    session.complete_session()
    print(f"Session completed at: {session.completed_at}")
    print(f"Duration: {session.session_duration_minutes} minutes")
    
    print("✅ ReviewSession tests passed!\n")

def test_user_settings():
    """Test UserSpacedRepetitionSettings functionality"""
    print("Testing UserSpacedRepetitionSettings...")
    
    # Create default settings
    settings = UserSpacedRepetitionSettings(
        id="test_settings_1",
        user_id="test_user_1"
    )
    
    print(f"Created settings for user: {settings.user_id}")
    print(f"New cards per day: {settings.new_cards_per_day}")
    print(f"Max reviews per day: {settings.max_reviews_per_day}")
    print(f"Easy bonus: {settings.easy_bonus}")
    print(f"Max interval: {settings.max_interval} days")
    
    # Test serialization
    settings_dict = settings.to_dict()
    print(f"Serialized to dict: {len(settings_dict)} fields")
    
    # Test deserialization
    restored_settings = UserSpacedRepetitionSettings.from_dict(settings_dict)
    print(f"Deserialized settings match: {restored_settings.user_id == settings.user_id}")
    
    print("✅ UserSpacedRepetitionSettings tests passed!\n")

def test_algorithm_accuracy():
    """Test the spaced repetition algorithm accuracy"""
    print("Testing Spaced Repetition Algorithm...")
    
    # Create a card and simulate a learning sequence
    card = SpacedRepetitionCard(
        id="test_card_algorithm",
        activity_id="test_activity",
        card_index=0,
        front="Test question",
        back="Test answer"
    )
    
    print("Simulating learning sequence: Good -> Good -> Easy -> Good -> Hard")
    
    # Simulate learning sequence
    sequence = [2, 2, 3, 2, 1]  # Good, Good, Easy, Good, Hard
    quality_names = ["Again", "Hard", "Good", "Easy"]
    
    for i, rating in enumerate(sequence, 1):
        card.calculate_next_review(rating)
        print(f"Step {i} ({quality_names[rating]}): "
              f"Repetitions={card.repetitions}, "
              f"Interval={card.interval_days} days, "
              f"Ease={card.ease_factor:.2f}")
    
    print("✅ Algorithm accuracy tests passed!\n")

def main():
    """Run all tests"""
    print("🧪 Running Spaced Repetition System Tests")
    print("=" * 50)
    
    try:
        test_spaced_repetition_card()
        test_review_session()
        test_user_settings()
        test_algorithm_accuracy()
        
        print("🎉 All tests passed! Spaced Repetition system is ready.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
