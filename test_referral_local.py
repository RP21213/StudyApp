#!/usr/bin/env python3
"""
Local test server for referral system
This runs a minimal Flask app to test referral functionality without Firebase
"""

from flask import Flask, request, render_template_string, jsonify, session, redirect, url_for
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import random
import os

# Create Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'test-secret-key-for-development'

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Simple in-memory storage for testing
users_db = {}
referrals_db = {}

# Simple User class for testing
class TestUser(UserMixin):
    def __init__(self, id, email, password_hash, referral_code=None, referred_by=None, pro_referral_count=0):
        self.id = id
        self.email = email
        self.password_hash = password_hash
        self.referral_code = referral_code
        self.referred_by = referred_by
        self.pro_referral_count = pro_referral_count

@login_manager.user_loader
def load_user(user_id):
    return users_db.get(user_id)

def generate_referral_code():
    """Generate a unique 6-digit referral code"""
    while True:
        code = str(random.randint(100000, 999999))
        if not any(user.referral_code == code for user in users_db.values()):
            return code

def validate_referral_code(code):
    """Validate if a referral code exists and return the referrer user"""
    if not code or len(code) != 6 or not code.isdigit():
        return None
    
    for user in users_db.values():
        if user.referral_code == code:
            return user
    return None

# Routes
@app.route('/')
def index():
    return '''
    <h1>🧪 Referral System Test</h1>
    <p><a href="/signup">Sign Up</a> | <a href="/login">Login</a> | <a href="/dashboard">Dashboard</a></p>
    <p><a href="/api/referrals/user-stats">API: User Stats</a> | <a href="/api/referrals/leaderboard">API: Leaderboard</a></p>
    '''

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        referral_code = request.form.get('referral_code', '').strip()
        
        if not email or not password:
            return "Email and password required", 400
        
        # Check if email already exists
        if any(user.email == email for user in users_db.values()):
            return "Email already exists", 400
        
        # Validate referral code if provided
        referred_by = None
        if referral_code:
            referrer = validate_referral_code(referral_code)
            if referrer:
                referred_by = referrer.id
                print(f"✅ Valid referral code {referral_code} from {referrer.email}")
            else:
                return f"Invalid referral code: {referral_code}", 400
        
        # Create new user
        user_id = f"user_{len(users_db) + 1}"
        new_referral_code = generate_referral_code()
        
        user = TestUser(
            id=user_id,
            email=email,
            password_hash=generate_password_hash(password),
            referral_code=new_referral_code,
            referred_by=referred_by
        )
        
        users_db[user_id] = user
        
        # Create referral record if applicable
        if referred_by:
            referral_id = f"referral_{len(referrals_db) + 1}"
            referrals_db[referral_id] = {
                'id': referral_id,
                'referrer_id': referred_by,
                'referred_id': user_id,
                'referral_code': referral_code,
                'status': 'pending'
            }
            print(f"✅ Created referral record: {referral_id}")
        
        login_user(user)
        return redirect(url_for('dashboard'))
    
    return '''
    <h2>Sign Up</h2>
    <form method="POST">
        <p>Email: <input type="email" name="email" required></p>
        <p>Password: <input type="password" name="password" required></p>
        <p>Referral Code (optional): <input type="text" name="referral_code" maxlength="6" placeholder="123456"></p>
        <p><button type="submit">Sign Up</button></p>
    </form>
    <p><a href="/">Back to Home</a></p>
    '''

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Find user
        user = None
        for u in users_db.values():
            if u.email == email and check_password_hash(u.password_hash, password):
                user = u
                break
        
        if user:
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            return "Invalid credentials", 400
    
    return '''
    <h2>Login</h2>
    <form method="POST">
        <p>Email: <input type="email" name="email" required></p>
        <p>Password: <input type="password" name="password" required></p>
        <p><button type="submit">Login</button></p>
    </form>
    <p><a href="/">Back to Home</a></p>
    '''

@app.route('/dashboard')
@login_required
def dashboard():
    return f'''
    <h2>Dashboard - {current_user.email}</h2>
    <p><strong>Your Referral Code:</strong> {current_user.referral_code}</p>
    <p><strong>Pro Referrals:</strong> {current_user.pro_referral_count}</p>
    <p><strong>Referred By:</strong> {current_user.referred_by or "No one"}</p>
    
    <h3>Test Actions:</h3>
    <p><a href="/simulate-pro-subscription">🎯 Simulate Pro Subscription</a></p>
    <p><a href="/api/referrals/user-stats">📊 View Stats API</a></p>
    <p><a href="/api/referrals/leaderboard">🏆 View Leaderboard API</a></p>
    
    <h3>All Users:</h3>
    {get_all_users_html()}
    
    <p><a href="/logout">Logout</a></p>
    '''

@app.route('/simulate-pro-subscription')
@login_required
def simulate_pro_subscription():
    """Simulate a Pro subscription to test referral rewards"""
    
    # Find referral records for this user
    for referral_id, referral in referrals_db.items():
        if referral['referred_id'] == current_user.id and referral['status'] == 'pending':
            # Update referral status
            referral['status'] = 'pro_subscribed'
            
            # Update referrer's stats
            referrer = users_db.get(referral['referrer_id'])
            if referrer:
                referrer.pro_referral_count += 1
                
                # Check milestones
                milestones = {3: "1 month Pro free", 10: "£20 giftcard", 20: "£50 giftcard", 50: "£100 giftcard"}
                for count, reward in milestones.items():
                    if referrer.pro_referral_count == count:
                        print(f"🎉 {referrer.email} reached {count} referrals! Reward: {reward}")
            
            print(f"✅ Processed Pro subscription for {current_user.email} -> {referrer.email if referrer else 'Unknown'}")
            break
    
    return redirect(url_for('dashboard'))

@app.route('/api/referrals/user-stats')
@login_required
def api_user_stats():
    milestones = {
        "3": {"reached": current_user.pro_referral_count >= 3, "reward": "One month Pro for free"},
        "10": {"reached": current_user.pro_referral_count >= 10, "reward": "£20 Amazon giftcard"},
        "20": {"reached": current_user.pro_referral_count >= 20, "reward": "£50 Amazon giftcard"},
        "50": {"reached": current_user.pro_referral_count >= 50, "reward": "£100 Amazon giftcard"}
    }
    
    stats = {
        "pro_referral_count": current_user.pro_referral_count,
        "referral_earnings": 0.0,
        "referral_code": current_user.referral_code,
        "milestones": milestones
    }
    
    return jsonify({"success": True, "stats": stats})

@app.route('/api/referrals/leaderboard')
@login_required
def api_leaderboard():
    leaderboard = []
    for user in users_db.values():
        if user.pro_referral_count > 0:
            leaderboard.append({
                "display_name": user.email.split('@')[0],
                "pro_referral_count": user.pro_referral_count,
                "avatar_url": "/static/images/default-avatar.svg"
            })
    
    # Sort by referral count
    leaderboard.sort(key=lambda x: x['pro_referral_count'], reverse=True)
    
    return jsonify({"success": True, "leaderboard": leaderboard})

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

def get_all_users_html():
    html = "<ul>"
    for user in users_db.values():
        html += f"<li>{user.email} (Code: {user.referral_code}, Referrals: {user.pro_referral_count})</li>"
    html += "</ul>"
    return html

if __name__ == '__main__':
    print("🚀 Starting Referral System Test Server...")
    print("📱 Access at: http://localhost:5001")
    print("🧪 This is a test server - no real data is stored")
    app.run(debug=True, port=5001)
