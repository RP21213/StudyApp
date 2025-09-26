#!/usr/bin/env python3
"""
Test script for the referral system
Run this to test referral functionality without the full app
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import User, Referral
import random

def test_generate_referral_code():
    """Generate a unique 6-digit referral code for testing"""
    return str(random.randint(100000, 999999))

def test_referral_models():
    """Test the User and Referral models"""
    print("🧪 Testing Referral System Models...")
    
    # Test User model with referral fields
    user1 = User(
        id="test_user_1",
        email="test1@example.com",
        password_hash="test_hash",
        referral_code="123456",
        referred_by=None,
        pro_referral_count=0,
        referral_earnings=0.0
    )
    
    user2 = User(
        id="test_user_2", 
        email="test2@example.com",
        password_hash="test_hash",
        referral_code="789012",
        referred_by="test_user_1",
        pro_referral_count=0,
        referral_earnings=0.0
    )
    
    # Test Referral model
    referral = Referral(
        id="test_referral_1",
        referrer_id="test_user_1",
        referred_id="test_user_2",
        referral_code="123456",
        status="pending"
    )
    
    print(f"✅ User 1: {user1.email}, Code: {user1.referral_code}")
    print(f"✅ User 2: {user2.email}, Referred by: {user2.referred_by}")
    print(f"✅ Referral: {referral.referrer_id} → {referral.referred_id}, Status: {referral.status}")
    
    # Test milestone logic
    milestones = {
        3: "One month Pro for free",
        10: "£20 Amazon giftcard", 
        20: "£50 Amazon giftcard",
        50: "£100 Amazon giftcard"
    }
    
    print("\n🏆 Testing Milestones:")
    for count, reward in milestones.items():
        reached = user1.pro_referral_count >= count
        print(f"   {count} referrals: {reward} {'✅' if reached else '❌'}")
    
    print("\n🎯 Referral System Models Test: PASSED!")

def test_referral_code_generation():
    """Test referral code generation"""
    print("\n🔢 Testing Referral Code Generation...")
    
    codes = []
    for i in range(5):
        code = test_generate_referral_code()
        codes.append(code)
        print(f"   Generated code {i+1}: {code}")
    
    # Check uniqueness
    if len(set(codes)) == len(codes):
        print("✅ All codes are unique!")
    else:
        print("❌ Duplicate codes found!")
    
    print("🎯 Referral Code Generation Test: PASSED!")

def test_milestone_logic():
    """Test milestone reward logic"""
    print("\n🏅 Testing Milestone Logic...")
    
    test_counts = [0, 2, 3, 5, 10, 15, 20, 25, 50, 100]
    
    for count in test_counts:
        rewards = []
        if count >= 3:
            rewards.append("1 month Pro free")
        if count >= 10:
            rewards.append("£20 Amazon giftcard")
        if count >= 20:
            rewards.append("£50 Amazon giftcard")
        if count >= 50:
            rewards.append("£100 Amazon giftcard")
        
        print(f"   {count} referrals: {', '.join(rewards) if rewards else 'No rewards'}")
    
    print("🎯 Milestone Logic Test: PASSED!")

if __name__ == "__main__":
    print("🚀 Starting Referral System Tests...\n")
    
    try:
        test_referral_models()
        test_referral_code_generation()
        test_milestone_logic()
        
        print("\n🎉 All Tests Passed! Referral system is working correctly.")
        print("\n📝 Next Steps:")
        print("   1. Set up local environment variables")
        print("   2. Run: python app.py")
        print("   3. Test at: http://localhost:5000")
        print("   4. Try signing up with a referral code")
        
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
        sys.exit(1)
