# ==============================================================================
# 1. SETUP & IMPORTS
# ==============================================================================
import os
import io
import json
import markdown
import re
import random
import traceback
import sys
from datetime import datetime, timezone
from flask import Flask, request, render_template, redirect, url_for, Response, send_file, flash, jsonify, session
from werkzeug.security import check_password_hash, generate_password_hash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from openai import OpenAI
import PyPDF2
import ast
from fpdf import FPDF
import csv
from flask import Response
from datetime import datetime, timedelta, timezone
from twilio.rest import Client
import threading
import firebase_admin
from firebase_admin import credentials, firestore, storage
import stripe
import base64 
import json   
from models import Hub, Activity, Note, Lecture, StudySession, Folder, Notification, Assignment, CalendarEvent, User, SharedFolder, AnnotatedSlideDeck, StudyGroup, StudyGroupMember, SharedResource, Referral, SpacedRepetitionCard, ReviewSession, UserSpacedRepetitionSettings
import asyncio 
from flask_socketio import SocketIO, emit, join_room, leave_room 
import assemblyai as aai 
from assemblyai.streaming.v3 import (
    BeginEvent,
    StreamingClient,
    StreamingClientOptions,
    StreamingError,
    StreamingEvents,
    StreamingParameters,
    StreamingSessionParameters,
    TerminationEvent,
    TurnEvent,
)
from flask import request

# --- NEW: Imports for Spotify ---
import requests
import urllib.parse
from functools import wraps

# --- NEW: Imports for Authentication ---
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash

# --- NEW: Imports for .ics parsing ---
from icalendar import Calendar
from werkzeug.utils import secure_filename
import uuid


# --- UPDATED: Import the new models, including Lecture ---
from models import Hub, Activity, Note, Lecture

# --- NEW: Imports for AI Tutor (LangChain) ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
# --- UPDATED: Import new LangChain history-aware components ---
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage


# --- PDF Generation Class ---
class NotesPDF(FPDF):
    def header(self):
        self.set_font('DejaVu', 'B', 12)
        self.cell(0, 10, 'AI Generated Study Notes', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('DejaVu', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def add_fonts(self):
        font_dir = os.path.dirname(os.path.abspath(__file__))
        self.add_font('DejaVu', '', os.path.join(font_dir, 'DejaVuSans.ttf'), uni=True)
        self.add_font('DejaVu', 'B', os.path.join(font_dir, 'DejaVuSans-Bold.ttf'), uni=True)
        self.add_font('DejaVu', 'I', os.path.join(font_dir, 'DejaVuSans-Oblique.ttf'), uni=True)

    def write_html_to_pdf(self, html_content):
        # Basic HTML to PDF conversion for interactive notes
        temp_markdown = re.sub(r'<span class="keyword".*?>(.*?)</span>', r'**\1**', html_content)
        temp_markdown = re.sub(r'<span class="formula".*?>(.*?)</span>', r'*\1*', temp_markdown)
        temp_markdown = temp_markdown.replace('<h1>', '# ').replace('</h1>', '\n')
        temp_markdown = temp_markdown.replace('<h2>', '## ').replace('</h2>', '\n')
        temp_markdown = temp_markdown.replace('<h3>', '### ').replace('</h3>', '\n')
        temp_markdown = temp_markdown.replace('<p>', '').replace('</p>', '\n')
        temp_markdown = temp_markdown.replace('<ul>', '').replace('</ul>', '')
        temp_markdown = temp_markdown.replace('<li>', '* ').replace('</li>', '\n')
        temp_markdown = temp_markdown.replace('<strong>', '**').replace('</strong>', '**')
        self.write_markdown(temp_markdown)

    def write_markdown(self, markdown_text):
        lines = markdown_text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                self.ln(5)
                continue
            if line.startswith('### '):
                self.set_font('DejaVu', 'B', 12)
                self.multi_cell(0, 8, line[4:])
                self.ln(2)
            elif line.startswith('## '):
                self.set_font('DejaVu', 'B', 14)
                self.multi_cell(0, 9, line[3:])
                self.ln(3)
            elif line.startswith('# '):
                self.set_font('DejaVu', 'B', 16)
                self.multi_cell(0, 10, line[2:])
                self.ln(4)
            elif line.startswith('* ') or line.startswith('- '):
                self.set_font('DejaVu', '', 11)
                self.cell(5)
                self.multi_cell(0, 6, chr(149) + " " + line[2:])
                self.ln(1)
            else:
                self.set_font('DejaVu', '', 11)
                parts = re.split(r'(\*\*.*?\*\*)', line)
                for part in parts:
                    if part.startswith('**') and part.endswith('**'):
                        self.set_font('', 'B')
                        self.write(6, part.strip('*'))
                        self.set_font('', '')
                    else:
                        self.write(6, part)
                self.ln(6)



# --- Initialize Firebase ---

try:
    # Check if the encoded key is in environment variables (for production on Render)
    firebase_key_b64 = os.getenv("FIREBASE_KEY_B64")
    if firebase_key_b64:
        # Decode the base64 string back into a JSON string
        firebase_key_json = base64.b64decode(firebase_key_b64).decode('utf-8')
        # Load the JSON string into a Python dictionary
        cred_dict = json.loads(firebase_key_json)
        cred = credentials.Certificate(cred_dict)
    else:
        # Fallback to local file for development on your own computer
        base_dir = os.path.dirname(os.path.abspath(__file__))
        key_path = os.path.join(base_dir, "firebase_key.json")
        cred = credentials.Certificate(key_path)

    BUCKET_NAME = os.getenv("FIREBASE_BUCKET_NAME", "ai-study-hub-f3040.firebasestorage.app")
    firebase_admin.initialize_app(cred, {'storageBucket': BUCKET_NAME})
    print("Firebase initialized successfully!")
    
    # Initialize Firestore and Storage only if Firebase init succeeded
    db = firestore.client()
    bucket = storage.bucket()
    
except Exception as e:
    print(f"Firebase initialization failed. Error: {e}")
    print("⚠️  Running in development mode without Firebase - some features will be limited")
    # Set dummy objects for development
    db = None
    bucket = None

# --- Flask App and OpenAI Client Initialization (MODIFIED) ---
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv("FLASK_SECRET_KEY", "a-default-secret-key-for-development")

# Custom JSON encoder to handle Undefined values
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if str(obj) == 'Undefined' or (hasattr(obj, '__class__') and 'Undefined' in str(obj.__class__)):
            return None
        return super().default(obj)

app.json_encoder = CustomJSONEncoder

socketio = SocketIO(app, async_mode='threading', cors_allowed_origins="*") # NEW: Initialize SocketIO
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# --- MODIFIED: Spotify Configuration ---
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
SPOTIFY_AUTH_URL = "https://accounts.spotify.com/authorize"
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"

# --- NEW: Twilio Configuration ---
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN) if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN else None

# --- NEW: Referral System Utilities ---
def generate_referral_code():
    """Generate a unique 6-digit referral code"""
    while True:
        code = str(random.randint(100000, 999999))
        # Check if code already exists
        existing_user = db.collection('users').where(filter=firestore.FieldFilter('referral_code', '==', code)).limit(1).stream()
        if not list(existing_user):
            return code

def validate_referral_code(code):
    """Validate if a referral code exists and return the referrer user"""
    if not code or len(code) != 6 or not code.isdigit():
        return None
    
    users = db.collection('users').where(filter=firestore.FieldFilter('referral_code', '==', code)).limit(1).stream()
    for user_doc in users:
        user_data = user_doc.to_dict()
        return User.from_dict(user_data)
    return None

def process_referral_rewards(user_id):
    """Process referral rewards when a user subscribes to Pro"""
    try:
        print(f"🔍 Getting user data for {user_id}")
        # Get the user who just subscribed
        user_doc = db.collection('users').document(user_id).get()
        if not user_doc.exists:
            print(f"❌ User {user_id} not found for referral processing")
            return
        
        user_data = user_doc.to_dict()
        user = User.from_dict(user_data)
        print(f"📧 User email: {user.email}")
        print(f"🔗 User referred_by: {user.referred_by}")
        
        # Check if this user was referred
        if not user.referred_by:
            print(f"ℹ️ User {user.email} was not referred - no rewards to process")
            return
        
        # Find the referral record
        referrals_query = db.collection('referrals').where(filter=firestore.FieldFilter('referred_id', '==', user_id)).where(filter=firestore.FieldFilter('status', '==', 'pending')).stream()
        
        for referral_doc in referrals_query:
            referral_data = referral_doc.to_dict()
            referral = Referral.from_dict(referral_data)
            
            # Update referral status
            referral.status = 'pro_subscribed'
            referral.pro_subscribed_at = datetime.now(timezone.utc)
            referral_doc.reference.update({
                'status': 'pro_subscribed',
                'pro_subscribed_at': referral.pro_subscribed_at
            })
            
            # Update referrer's stats
            referrer_ref = db.collection('users').document(referral.referrer_id)
            referrer_ref.update({
                'pro_referral_count': firestore.Increment(1)
            })
            
            # Get updated referrer data to check milestones
            referrer_doc = referrer_ref.get()
            referrer_data = referrer_doc.to_dict()
            new_count = referrer_data.get('pro_referral_count', 0)
            
            # Process milestone rewards
            reward_processed = False
            if new_count == 4:
                # Give £10 Amazon giftcard (milestone reached)
                # Note: Actual giftcard will be sent via email claim process
                reward_processed = True
                print(f"🎉 {referrer_data.get('email', 'Unknown')} reached 4 referrals! Reward: £10 Amazon giftcard")
                
            elif new_count in [10, 20, 50]:
                # Mark for gift card reward (manual processing needed)
                gift_amounts = {10: 20, 20: 50, 50: 100}
                referral.reward_type = 'giftcard'
                referral.reward_amount = gift_amounts[new_count]
                reward_processed = True
                print(f"🎉 {referrer_data.get('email', 'Unknown')} reached {new_count} referrals! Reward: £{gift_amounts[new_count]} Amazon giftcard")
            
            # Update referral record with reward info
            referral_doc.reference.update({
                'reward_type': referral.reward_type,
                'reward_amount': referral.reward_amount
            })
            
            print(f"✅ Processed referral for user {user.email} -> {referrer_data.get('email', 'Unknown')}, new count: {new_count}")
            
            # Send notification to referrer (optional - you can implement this later)
            # send_referral_notification(referral.referrer_id, new_count, reward_processed)
            
    except Exception as e:
        print(f"Error in process_referral_rewards: {e}")
        raise

SPOTIFY_API_BASE_URL = "https://api.spotify.com/v1/"
SPOTIFY_REDIRECT_URI = os.getenv("YOUR_DOMAIN", "http://127.0.0.1:5000") + "/spotify/callback"
# Added "streaming", "user-read-email", "user-read-private" for the SDK
SPOTIFY_SCOPES = "streaming user-read-email user-read-private user-read-playback-state user-modify-playback-state user-read-currently-playing playlist-read-private"



# --- NEW: In-memory cache for AI Tutor vector stores ---
vector_store_cache = {}

# --- NEW: Configure AssemblyAI ---
aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")

# --- NEW: Real-time transcription configuration ---
REALTIME_TRANSCRIPTION_CONFIG = {
    "sample_rate": 16000,
    "format_turns": True
}

# --- NEW: Real-time transcription session storage ---
realtime_sessions = {}  # Store active transcription sessions

# --- NEW: Real-time transcription functions ---
def create_realtime_transcriber():
    """Create and configure AssemblyAI v3 Streaming Client."""
    try:
        print(f"Creating v3 streaming client with config: {REALTIME_TRANSCRIPTION_CONFIG}")
        print(f"AssemblyAI API key present: {bool(aai.settings.api_key)}")
        
        # Check if API key is available
        if not aai.settings.api_key:
            print("No AssemblyAI API key found, using fallback mode")
            return "fallback"  # Return a special value for fallback mode
        
        # Create v3 streaming client
        client = StreamingClient(
            StreamingClientOptions(
                api_key=aai.settings.api_key,
                api_host="streaming.assemblyai.com",
            )
        )
        
        print("v3 Streaming client created successfully")
        return client
    except Exception as e:
        print(f"Error creating v3 streaming client: {e}")
        import traceback
        traceback.print_exc()
        print("Using fallback mode")
        return "fallback"

def create_streaming_client_for_session(session_id):
    """Create a new StreamingClient for a specific session."""
    try:
        if not aai.settings.api_key:
            return "fallback"
        
        # Create a new client for this session
        client = StreamingClient(
            StreamingClientOptions(
                api_key=aai.settings.api_key,
                api_host="streaming.assemblyai.com",
            )
        )
        
        # Set up event handlers
        setup_streaming_events(client, session_id)
        
        return client
    except Exception as e:
        print(f"Error creating streaming client for session: {e}")
        return "fallback"

def setup_streaming_events(client, session_id):
    """Set up event handlers for the streaming client."""
    def on_begin(event: BeginEvent):
        print(f"Session started: {event.id}")
        # Store session ID in our session data
        if session_id in realtime_sessions:
            realtime_sessions[session_id]['assemblyai_session_id'] = event.id
    
    def on_turn(event: TurnEvent):
        print(f"Turn received: {event.transcript} (end_of_turn: {event.end_of_turn})")
        
        # Update session with new transcript
        if session_id in realtime_sessions:
            session_data = realtime_sessions[session_id]
            session_data['transcript_buffer'] += event.transcript + " "
            
            # Generate updated notes
            updated_notes = generate_live_notes_update(
                session_data['current_notes'], 
                event.transcript
            )
            session_data['current_notes'] = updated_notes
            
            # Emit to client via WebSocket
            socketio.emit('notes_updated', {
                'notes_html': updated_notes,
                'transcript_chunk': event.transcript
            }, room=session_id)
    
    def on_terminated(event: TerminationEvent):
        print(f"Session terminated: {event.audio_duration_seconds} seconds of audio processed")
    
    def on_error(error: StreamingError):
        print(f"AssemblyAI streaming error: {error}")
        # Emit error to client
        socketio.emit('error', {'message': f'Transcription error: {error}'}, room=session_id)
    
    # Set up event handlers
    client.on(StreamingEvents.Begin, on_begin)
    client.on(StreamingEvents.Turn, on_turn)
    client.on(StreamingEvents.Termination, on_terminated)
    client.on(StreamingEvents.Error, on_error)
    
    return client

# --- NEW: Configure Stripe ---
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
YOUR_DOMAIN = os.getenv("YOUR_DOMAIN", "http://127.0.0.1:5000")


# --- NEW: Setup Flask-Login ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login' # Redirect to /login if user is not authenticated

@login_manager.user_loader
def load_user(user_id):
    user_doc = db.collection('users').document(user_id).get()
    if user_doc.exists:
        return User.from_dict(user_doc.to_dict())
    return None


# --- Jinja Filter for Relative Time ---
def timesince_filter(dt, default="just now"):
    if not dt: return default
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
    diff = now - dt
    periods = ( (diff.days // 365, "year"), (diff.days // 30, "month"), (diff.days // 7, "week"), (diff.days, "day"), (diff.seconds // 3600, "hour"), (diff.seconds // 60, "minute"), (diff.seconds, "second"),)
    for period, a_unit in periods:
        if period >= 1: return "%d %s%s" % (int(period), a_unit, "s" if int(period) > 1 else "")
    return default
app.jinja_env.filters['timesince'] = timesince_filter

# --- NEW: Context Processor to make datetime available in all templates ---
@app.context_processor
def inject_now():
    """Injects the current UTC datetime into all templates."""
    return {'now': datetime.now(timezone.utc)}

# ==============================================================================
# 2. CORE UTILITY & HELPER FUNCTIONS
# ==============================================================================

def _get_user_stats(user_id):
    """Helper function to calculate stats for a given user."""
    hubs_ref = db.collection('hubs').where('user_id', '==', current_user.id).stream()
    hubs_list = []
    for doc in hubs_ref:
        hub_data = doc.to_dict()
        hub_data['id'] = doc.id  # <-- THE CRITICAL FIX IS HERE
        hubs_list.append(Hub.from_dict(hub_data))
    
    total_study_minutes = 0
    longest_streak = 0
    
    for hub in hubs_list:
        if hub.streak_days > longest_streak:
            longest_streak = hub.streak_days
        sessions_query = db.collection('sessions').where('hub_id', '==', hub.id).stream()
        for _ in sessions_query:
            total_study_minutes += 30 
            
    total_study_hours = round(total_study_minutes / 60, 1)

    return {
        "total_study_hours": total_study_hours,
        "longest_streak": longest_streak,
        "hubs": hubs_list
    }

# --- NEW: Helper function for updating hub progress ---
def update_hub_progress(hub_id, xp_to_add):
    """
    Atomically updates a hub's total XP and calculates the study streak.
    """
    try:
        hub_ref = db.collection('hubs').document(hub_id)
        hub_doc = hub_ref.get()
        if not hub_doc.exists:
            return

        hub_data = hub_doc.to_dict()
        
        # Streak Calculation Logic
        today = datetime.now(timezone.utc).date()
        last_study_date = hub_data.get('last_study_date')
        
        # Convert Firestore timestamp to date if it exists
        if last_study_date:
            last_study_date = last_study_date.date()

        current_streak = hub_data.get('streak_days', 0)
        new_streak = current_streak

        if last_study_date is None:
            # First activity ever
            new_streak = 1
        elif last_study_date == today:
            # Already studied today, streak doesn't increase
            pass
        elif last_study_date == today - timedelta(days=1):
            # Studied yesterday, increment streak
            new_streak += 1
        else:
            # Missed a day or more, reset streak to 1
            new_streak = 1
            
        # Use a transaction to safely update XP and streak data
        hub_ref.update({
            'total_xp': firestore.Increment(xp_to_add),
            'streak_days': new_streak,
            'last_study_date': datetime.now(timezone.utc)
        })

    except Exception as e:
        print(f"Error updating hub progress for hub {hub_id}: {e}")

def pdf_to_text(pdf_file_stream):
    try:
        reader = PyPDF2.PdfReader(pdf_file_stream)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text: text += page_text + "\n"
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

def get_text_from_hub_files(selected_files):
    combined_text = ""
    if not selected_files: return ""
    for file_path in selected_files:
        blob = bucket.blob(file_path)
        file_name = os.path.basename(file_path)
        try:
            pdf_bytes = blob.download_as_bytes()
            pdf_stream = io.BytesIO(pdf_bytes)
            text = pdf_to_text(pdf_stream)
            combined_text += f"--- DOCUMENT: {file_name} ---\n\n{text}\n\n"
        except Exception as e:
            print(f"Failed to process file {file_path}: {e}")
    return combined_text

def parse_flashcards(raw_text):
    flashcards = []
    card_blocks = raw_text.strip().split('---')
    for block in card_blocks:
        if 'Front:' in block and 'Back:' in block:
            parts = block.strip().split('\nBack:')
            front_part = parts[0].replace('Front:', '').strip()
            back_part = parts[1].strip()
            flashcards.append({'front': front_part, 'back': back_part})
    return flashcards

def safe_load_json(data):
    if not data: return {}
    try: return json.loads(data)
    except json.JSONDecodeError:
        try: return ast.literal_eval(data)
        except (ValueError, SyntaxError): return {}

# ==============================================================================
# 3. AI GENERATION SERVICES
# ==============================================================================

# --- NEW: AI Functions for Slide Note-Taking Feature ---
def generate_summary_from_slide_text(text):
    """Generates a concise summary from the text content of a single slide."""
    prompt = f"You are an expert academic summarizer. Condense the key points from the following slide content into a clear, concise summary (2-3 bullet points in Markdown). Content:\n---\n{text}"
    response = client.chat.completions.create(
        model="gpt-4o-mini", 
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def generate_flashcards_from_slide_text(text, count=3):
    """Generates flashcards from the text content with specified count."""
    prompt = f"""
    Based on the following slide content, create exactly {count} key flashcards.
    Follow this format strictly for each card:
    Front: [Question or Term]
    Back: [Answer or Definition]
    ---
    Here is the text:
    {text}
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini", 
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# --- AI function for Revision Pack preview ---
def generate_revision_preview(text):
    """Generates a summary, key terms, and sample items for a revision pack preview."""
    prompt = f"""
    You are an AI assistant analyzing a lecture document for a student. Your task is to extract a concise preview for a revision pack.
    Your response MUST be a single, valid JSON object with the following keys: "summary", "key_terms", "sample_flashcards", "sample_quiz".

    - "summary": A 1-2 paragraph summary of the entire document.
    - "key_terms": An array of 5-10 of the most important single words or short phrases.
    - "sample_flashcards": An array of exactly 3 flashcard objects. Each object must have "front" and "back" keys.
    - "sample_quiz": An array of exactly 2 multiple-choice question objects. Each object must have "question", "options" (an array of 4 strings), and "correct_answer".

    Analyze the following text to generate the preview:
    ---
    {text}
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content

# --- AI function for Exam Prep Kit preview ---
def generate_exam_preview(text):
    """Generates key concepts and sample items for an exam prep kit preview."""
    prompt = f"""
    You are an AI assistant analyzing a lecture document to help a student prepare for an exam.
    Your response MUST be a single, valid JSON object with the following keys: "key_concepts", "cheat_sheet_sample", "sample_questions".

    - "key_concepts": An array of 3-4 advanced or frequently mentioned topics from the text.
    - "cheat_sheet_sample": A single, well-structured HTML block for a cheat sheet, focusing on the most important formula, definition, or process in the text. Use `<h4>`, `<p>`, `<strong>`.
    - "sample_questions": An array of 1-2 challenging short-answer questions. Each object must have "question" and "model_answer".

    Analyze the following text to generate the preview:
    ---
    {text}
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content


# --- AI function for live, contextual summarization ---
def generate_live_summary_update(previous_summary, new_transcript_chunk):
    """
    Takes the previous summary and a new chunk of text, and returns an updated summary.
    """
    prompt = f"""
    You are an AI note-taker for a university lecture. Your task is to continuously update a set of notes.
    
    This is the summary of the lecture so far:
    --- PREVIOUS NOTES ---
    {previous_summary}
    --- END PREVIOUS NOTES ---

    Here is the newest chunk of transcribed text from the lecture:
    --- NEW TRANSCRIPT ---
    {new_transcript_chunk}
    --- END NEW TRANSCRIPT ---

    Your instructions are:
    1.  Read the "NEW TRANSCRIPT" and identify the key points, new concepts, or important details.
    2.  Integrate these new points into the "PREVIOUS NOTES". You can add new bullet points, create new headings (using <h2> or <h3>), or elaborate on existing points.
    3.  DO NOT repeat information that is already well-covered in the "PREVIOUS NOTES".
    4.  Ensure the output is well-structured, clean HTML. Use headings, paragraphs, lists, and bold tags.
    5.  Your output MUST be ONLY the complete, updated HTML notes. Do not include any other text or explanation.

    Return the complete, updated set of notes as a single block of HTML.
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini", # Use a faster model for real-time updates
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# --- NEW: Real-time note generation functions ---
def generate_initial_notes_structure(lecture_title, is_fallback=False):
    """Generate an initial HTML structure for live note-taking."""
    if is_fallback:
        content = """
            <p><em>Live transcription is not available - AssemblyAI API key needed for real-time transcription.</em></p>
            <p><em>Please add your AssemblyAI API key to enable live transcription and note-taking.</em></p>
        """
    else:
        content = '<p><em>Live transcription and note-taking in progress...</em></p>'
    
    return f"""
    <div class="live-notes">
        <h1>{lecture_title}</h1>
        <div class="notes-timestamp">Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        <div class="notes-content">
            {content}
        </div>
    </div>
    """

def generate_live_notes_update(previous_notes, new_transcript, is_final=False):
    """
    Generate updated notes from new transcript chunk.
    Uses a more sophisticated approach for real-time note generation.
    """
    # Extract the content area from previous notes
    import re
    content_match = re.search(r'<div class="notes-content">(.*?)</div>', previous_notes, re.DOTALL)
    previous_content = content_match.group(1) if content_match else ""
    
    prompt = f"""
    You are an expert AI note-taker for university lectures. Your task is to update lecture notes in real-time.
    
    CURRENT NOTES CONTENT:
    {previous_content}
    
    NEW TRANSCRIPT CHUNK:
    {new_transcript}
    
    Instructions:
    1. Analyze the new transcript for key concepts, definitions, examples, and important points
    2. Update the notes by:
       - Adding new concepts as bullet points or sections
       - Expanding on existing topics if the new content relates
       - Creating new headings for major topic changes
       - Highlighting important terms with <strong> tags
       - Adding examples or explanations
    3. Maintain clean HTML structure with proper headings (h2, h3), paragraphs, and lists
    4. Keep notes concise but comprehensive
    5. If this is the final chunk, add a conclusion section
    
    Return ONLY the updated content for the notes-content div, without the wrapper div.
    """
    
    try:
        print(f"Calling OpenAI API for live notes update...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3  # Lower temperature for more consistent formatting
        )
        
        updated_content = response.choices[0].message.content.strip()
        print(f"OpenAI API response received successfully")
        
        # Replace the content in the previous notes
        updated_notes = re.sub(
            r'<div class="notes-content">.*?</div>',
            f'<div class="notes-content">{updated_content}</div>',
            previous_notes,
            flags=re.DOTALL
        )
        
        return updated_notes
        
    except Exception as e:
        print(f"Error generating live notes update: {e}")
        import traceback
        traceback.print_exc()
        # Return previous notes with error message
        error_content = f"{previous_content}<p><em>Error updating notes: {str(e)}</em></p>"
        return re.sub(
            r'<div class="notes-content">.*?</div>',
            f'<div class="notes-content">{error_content}</div>',
            previous_notes,
            flags=re.DOTALL
        )


# --- NEW: Route to render the live lecture page ---
@app.route("/hub/<hub_id>/live_lecture")
@login_required
def live_lecture_page(hub_id):
    hub_doc = db.collection('hubs').document(hub_id).get()
    if not hub_doc.exists or Hub.from_dict(hub_doc.to_dict()).user_id != current_user.id:
        flash("Hub not found or you don't have access.", "error")
        return redirect(url_for('dashboard'))
    return render_template("live_lecture_realtime.html", hub_id=hub_id)

# --- NEW: WebSocket handlers for real-time lecture capture ---
@socketio.on('start_live_lecture')
def handle_start_live_lecture(data):
    """Handle starting a live lecture session."""
    print(f"=== START LIVE LECTURE EVENT RECEIVED ===")
    print(f"Data: {data}")
    print(f"User: {current_user.id if current_user.is_authenticated else 'Not authenticated'}")
    print(f"Session ID: {request.sid}")
    
    # Check authentication
    if not current_user.is_authenticated:
        print("User not authenticated")
        emit('error', {'message': 'Authentication required'})
        return
    
    hub_id = data.get('hub_id')
    lecture_title = data.get('title', f"Live Lecture - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    session_id = request.sid  # Get unique session ID
    print(f"Hub ID: {hub_id}, Title: {lecture_title}")
    
    # Verify user has access to the hub
    hub_doc = db.collection('hubs').document(hub_id).get()
    if not hub_doc.exists or Hub.from_dict(hub_doc.to_dict()).user_id != current_user.id:
        print("Hub access denied")
        emit('error', {'message': 'Hub not found or access denied'})
        return
    
    # Create session-specific streaming client
    print("Creating streaming client for session...")
    streaming_client = create_streaming_client_for_session(session_id)
    if not streaming_client:
        print("Failed to create streaming client")
        emit('error', {'message': 'Failed to initialize transcription service. Please check your AssemblyAI API key.'})
        return
    print("Streaming client created successfully")
    
    # Initialize session data
    is_fallback = (streaming_client == "fallback")
    session_data = {
        'hub_id': hub_id,
        'title': lecture_title,
        'start_time': datetime.now().isoformat(),
        'current_notes': generate_initial_notes_structure(lecture_title, is_fallback),
        'transcript_buffer': '',
        'is_active': True,
        'streaming_client': streaming_client,
        'client_connected': False
    }
    
    # Store session data
    realtime_sessions[session_id] = session_data
    
    # Join the session room for targeted messaging
    join_room(session_id)
    print(f"Joined room: {session_id}")
    
    print("Emitting lecture_started event...")
    emit('lecture_started', {
        'title': lecture_title,
        'initial_notes': session_data['current_notes']
    })
    print("lecture_started event emitted")

@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    """Handle incoming audio chunk for real-time transcription."""
    session_id = request.sid
    audio_data = data.get('audio_data')
    
    if not audio_data:
        emit('error', {'message': 'Missing audio_data'})
        return
    
    # Check audio data size to prevent payload errors
    if len(audio_data) > 5 * 1024 * 1024:  # 5MB limit (base64 encoded)
        print(f"Audio chunk too large: {len(audio_data)} bytes, skipping")
        return
    
    # Get session data
    session_data = realtime_sessions.get(session_id)
    if not session_data or not session_data.get('is_active'):
        emit('error', {'message': 'No active lecture session'})
        return
    
    try:
        streaming_client = session_data.get('streaming_client')
        if not streaming_client:
            emit('error', {'message': 'Transcription service not available'})
            return
        
        # Handle fallback mode (no AssemblyAI API key)
        if streaming_client == "fallback":
            print("Using fallback transcription mode - no AssemblyAI API key")
            
            # Check if this is a test session (user can enable test mode)
            test_mode = session_data.get('test_mode', False)
            
            if test_mode:
                # Simulate realistic speech for testing
                import time
                time.sleep(0.5)  # Simulate processing time
                
                # Generate realistic test transcript
                test_transcripts = [
                    "Today we're going to discuss machine learning algorithms.",
                    "The first concept we need to understand is supervised learning.",
                    "Supervised learning uses labeled training data to make predictions.",
                    "Examples include classification and regression problems.",
                    "In classification, we predict discrete categories like spam or not spam.",
                    "Regression predicts continuous values like house prices.",
                    "The key difference is the type of output we're trying to predict."
                ]
                
                # Use a simple counter to cycle through test transcripts
                chunk_count = session_data.get('test_chunk_count', 0)
                if chunk_count < len(test_transcripts):
                    simulated_transcript = test_transcripts[chunk_count]
                    session_data['test_chunk_count'] = chunk_count + 1
                    
                    # Update transcript buffer
                    session_data['transcript_buffer'] += simulated_transcript + " "
                    
                    # Generate updated notes
                    updated_notes = generate_live_notes_update(
                        session_data['current_notes'], 
                        simulated_transcript
                    )
                    session_data['current_notes'] = updated_notes
                    
                    # Emit updated notes to client
                    emit('notes_updated', {
                        'notes_html': updated_notes,
                        'transcript_chunk': simulated_transcript
                    })
            else:
                # Just acknowledge the audio chunk was received
                print(f"Audio chunk received (size: {len(audio_data)} bytes) - AssemblyAI API key needed for real transcription")
            return
        
        # Connect streaming client if not already connected
        if not session_data.get('client_connected'):
            try:
                # Connect to streaming service
                streaming_client.connect(
                    StreamingParameters(
                        sample_rate=REALTIME_TRANSCRIPTION_CONFIG["sample_rate"],
                        format_turns=REALTIME_TRANSCRIPTION_CONFIG["format_turns"],
                    )
                )
                session_data['client_connected'] = True
                print("Connected to AssemblyAI v3 streaming service")
            except Exception as e:
                print(f"Error connecting to AssemblyAI: {e}")
                import traceback
                traceback.print_exc()
                emit('error', {'message': f'Failed to connect to transcription service: {str(e)}'})
                return
        
        # Convert base64 audio data to bytes
        import base64
        try:
            audio_bytes = base64.b64decode(audio_data)
            print(f"Decoded audio bytes: {len(audio_bytes)} bytes")
            
            # Stream audio to AssemblyAI v3
            streaming_client.stream(audio_bytes)
            print("Audio streamed to AssemblyAI successfully")
        except Exception as decode_error:
            print(f"Error decoding base64 audio: {decode_error}")
            emit('error', {'message': f'Error processing audio data: {str(decode_error)}'})
            return
        
    except Exception as e:
        print(f"Error processing audio chunk: {e}")
        emit('error', {'message': f'Error processing audio: {str(e)}'})

@socketio.on('stop_live_lecture')
def handle_stop_live_lecture(data):
    """Handle stopping a live lecture session."""
    session_id = request.sid
    
    # Get session data
    session_data = realtime_sessions.get(session_id)
    if not session_data:
        emit('error', {'message': 'No active lecture session'})
        return
    
    try:
        # Disconnect streaming client
        streaming_client = session_data.get('streaming_client')
        if streaming_client and streaming_client != "fallback" and session_data.get('client_connected'):
            streaming_client.disconnect(terminate=True)
            session_data['client_connected'] = False
            print("Disconnected from AssemblyAI v3 streaming service")
        
        # Set end time but keep session active until processing is complete
        session_data['end_time'] = datetime.now().isoformat()
        
        # Generate final notes
        final_notes = generate_live_notes_update(
            session_data['current_notes'],
            session_data['transcript_buffer'],
            is_final=True
        )
        
        # Calculate duration properly
        try:
            start_time = datetime.fromisoformat(session_data.get('start_time', ''))
            end_time = datetime.fromisoformat(session_data.get('end_time', ''))
            duration = (end_time - start_time).total_seconds()
        except (ValueError, TypeError):
            duration = 0
        
        emit('lecture_stopped', {
            'final_notes': final_notes,
            'full_transcript': session_data['transcript_buffer'],
            'duration': duration
        })
        
    except Exception as e:
        print(f"Error stopping lecture: {e}")
        emit('error', {'message': f'Error stopping lecture: {str(e)}'})
    
    finally:
        # Mark session as inactive and clean up
        if session_id in realtime_sessions:
            realtime_sessions[session_id]['is_active'] = False
        realtime_sessions.pop(session_id, None)
        leave_room(session_id)

@socketio.on('test_connection')
def handle_test_connection(data):
    """Test WebSocket connection."""
    print(f"Test connection received: {data}")
    emit('test_response', {'message': 'Hello from server', 'timestamp': datetime.now().isoformat()})

@socketio.on('enable_test_mode')
def handle_enable_test_mode(data):
    """Enable test mode for fallback transcription."""
    session_id = request.sid
    session_data = realtime_sessions.get(session_id)
    
    if session_data:
        session_data['test_mode'] = True
        session_data['test_chunk_count'] = 0
        print(f"Test mode enabled for session {session_id}")
        emit('test_mode_enabled', {'message': 'Test mode enabled - realistic speech simulation active'})
    else:
        emit('error', {'message': 'No active session found'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection and cleanup."""
    session_id = request.sid
    
    # Clean up any active transcription session
    if session_id in realtime_sessions:
        session_data = realtime_sessions[session_id]
        
        # Disconnect transcriber if active
        transcriber = session_data.get('transcriber')
        if transcriber and transcriber != "fallback" and session_data.get('transcriber_active'):
            try:
                transcriber.close()
            except:
                pass
        
        # Remove session
        realtime_sessions.pop(session_id, None)
        leave_room(session_id)

# --- NEW: Route to save live lecture notes ---
@app.route("/hub/<hub_id>/save_live_lecture_notes", methods=["POST"])
@login_required
def save_live_lecture_notes(hub_id):
    """Save live lecture notes to the database."""
    try:
        title = request.form.get('title', f"Live Lecture - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        notes_html = request.form.get('notes_html', '')
        full_transcript = request.form.get('full_transcript', '')
        
        if not notes_html:
            return jsonify({"success": False, "message": "No notes content provided"}), 400
        
        # Create note in database
        note_ref = db.collection('notes').document()
        new_note = Note(
            id=note_ref.id,
            hub_id=hub_id,
            title=title,
            content_html=notes_html
        )
        note_ref.set(new_note.to_dict())
        
        # If there's a transcript, also create a raw transcript note
        if full_transcript:
            transcript_ref = db.collection('notes').document()
            transcript_note = Note(
                id=transcript_ref.id,
                hub_id=hub_id,
                title=f"{title} - Full Transcript",
                content_html=f"<div class='transcript'><pre>{full_transcript}</pre></div>"
            )
            transcript_ref.set(transcript_note.to_dict())
        
        return jsonify({
            "success": True, 
            "message": "Live lecture notes saved successfully!",
            "note_id": note_ref.id
        })
        
    except Exception as e:
        print(f"Error saving live lecture notes: {e}")
        return jsonify({"success": False, "message": str(e)}), 500


def generate_cheat_sheet_json(text):
    """Generates content for a multi-column cheat sheet as a JSON object."""
    prompt = f"""
    You are an expert content creator for cheat sheets. Analyze the provided text and extract the most critical information (key definitions, code snippets, important formulas, core concepts).
    Your output MUST be a single, valid JSON object structured for a 3-column layout.

    The JSON structure must be:
    {{
      "title": "Cheat Sheet Title",
      "columns": [
        {{ "blocks": [ {{ "title": "Block Title", "content_html": "HTML content..." }} ] }},
        {{ "blocks": [ {{ "title": "Block Title", "content_html": "HTML content..." }} ] }},
        {{ "blocks": [ {{ "title": "Block Title", "content_html": "HTML content..." }} ] }}
      ]
    }}

    - The root object has "title" and "columns".
    - "columns" is an array that should contain exactly 3 column objects.
    - Each column object has a "blocks" array.
    - Each block object has a "title" and "content_html".
    - The `content_html` should be well-formed HTML using `<p>`, `<ul>`, `<li>`, `<strong>`, and `<pre><code>` for code. Distribute the content logically across the three columns.

    Analyze this text:
    ---
    {text}
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content

def generate_audio_from_text(text, hub_id, session_id, topic_index):
    """Generates an MP3 audio file from text and saves it to Firebase Storage."""
    try:
        # Generate audio data from OpenAI's TTS API
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy", # Other voices: echo, fable, onyx, nova, shimmer
            input=text
        )
        audio_data = response.content

        # Define a unique path in Firebase Storage
        file_name = f"recap_topic_{topic_index}.mp3"
        file_path = f"hubs/{hub_id}/sessions/{session_id}/{file_name}"
        
        # Upload the audio data to the bucket
        blob = bucket.blob(file_path)
        blob.upload_from_string(audio_data, content_type='audio/mpeg')

        # Make the blob publicly accessible and get its URL
        blob.make_public()
        return blob.public_url

    except Exception as e:
        print(f"Error generating or uploading audio: {e}")
        return None

def generate_remedial_materials(text, topic):
    """Generates focused notes and a 3-question quiz for a specific weak topic."""
    prompt = f"""
    You are an expert tutor creating remedial study materials for a student who is struggling with a specific topic.
    The topic is: "{topic}"

    Your task is to generate a JSON object with two keys: "notes_html" and "questions".

    1.  "notes_html": Create a concise, clear, and well-structured HTML summary of the key concepts for ONLY the topic of "{topic}", based on the provided text. Use p, ul, li, and strong tags.
    2.  "questions": Create an array of 3 multiple-choice questions that specifically test the most important aspects of "{topic}". Each question object must have "question", "options" (an array of 4 strings), and "correct_answer".

    Base all content strictly on the provided text below.
    ---
    {text}
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content

def generate_full_study_session(text, duration, focus):
    """
    Generates a complete, multi-modal, 3-phase study plan in a single, robust AI call.
    """
    focus_instructions = {
        "recall": "Prioritize 'checkpoint_quiz' interactions to reinforce memory.",
        "understanding": "Prioritize 'teach_back' to deepen comprehension.",
        "exam_prep": "Use a mix of 'checkpoint_quiz' with varied questions and 'teach_back' for conceptual clarity."
    }
    duration_map = {
        "15": {"topics": "2-3", "detail": "brief"},
        "30": {"topics": "4-5", "detail": "standard"},
        "60": {"topics": "6-7", "detail": "in-depth"}
    }
    session_config = duration_map.get(str(duration), {"topics": "4-5", "detail": "standard"})
    num_topics = session_config["topics"]
    detail_level = session_config["detail"]

    detail_instructions = {
        "brief": "For each topic, the 'content_html' should be a concise summary of 2-3 paragraphs. Follow it with only one simple interaction block ('checkpoint_quiz' or 'teach_back').",
        "standard": "For each topic, the 'content_html' should be moderately detailed. Follow it with a mix of two interaction blocks.",
        "in-depth": "For each topic, the 'content_html' should be comprehensive and detailed. Follow it with at least two varied and challenging interaction blocks."
    }

    prompt = f"""
    You are an expert AI instructional designer creating a complete, personalized study session from text.
    The user wants a ~{duration} minute session focused on '{focus}'.

    **SESSION DIRECTIVES:**
    - **Topic Count:** Structure the content into {num_topics} logical topics.
    - **Detail Level:** The session must be '{detail_level}'. Follow this rule: {detail_instructions.get(detail_level)}
    - **Focus:** Adhere to the user's focus on '{focus}'. Follow this rule: {focus_instructions.get(focus)}

    Your response MUST be a single, valid JSON object with one key: "topics".
    Each topic object in the "topics" array must contain:
    1. "title": A string for the topic title.
    2. "content_html": A string of rich, well-formed HTML content for this topic. This is the main learning material. It MUST be detailed and comprehensive as per the detail level. Include interactive keywords by wrapping them in `<span class="keyword" title="concise definition">keyword</span>`.
    3. "interactions": An ARRAY of interaction block objects to follow the content.

    **Interaction Block Types:**
    - type: "checkpoint_quiz"
      - "questions": An array of 1-2 multiple-choice objects. Each must have "question", "options" (an array of 4 strings), and "correct_answer".
    - type: "teach_back"
      - "prompt": A string asking the user to explain the topic's core concept.
      - "model_answer": A concise, ideal answer for evaluation.

    **CRITICAL INSTRUCTIONS:**
    - The `content_html` for each topic MUST be substantial and detailed. This is the primary learning content.
    - The `interactions` you generate for a topic MUST be based ONLY on the `content_html` you just generated for that same topic.

    Here is the text to build the entire plan from:
    ---
    {text}
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content

def generate_interactive_notes_html(text):
    """Generates interactive study notes in HTML format using the AI."""
    prompt = f"""
    You are an expert study assistant creating comprehensive lecture-style notes. Your task is to read the uploaded material and convert it into detailed, thorough study notes that a student would take during a lecture. The notes must be:

    **DETAILED AND COMPREHENSIVE:**
    - Include substantial detail for each concept, not just brief summaries
    - Explain concepts thoroughly with context and background information
    - Include multiple examples, case studies, and practical applications
    - Cover all important subtopics and supporting details
    - Provide enough depth for someone to understand the material without the original source

    **WELL-ORGANIZED STRUCTURE:**
    - Use clear hierarchical headings and subheadings
    - Organize content logically with smooth transitions between topics
    - Use bullet points and numbered lists for clarity
    - Include cross-references between related concepts

    **LECTURE-STYLE CONTENT:**
    - Write as if explaining to a student in a classroom setting
    - Include "why" explanations, not just "what" facts
    - Add context about how concepts relate to each other
    - Include practical implications and real-world applications
    - Explain the significance and importance of each topic

    **INTERACTIVE ELEMENTS:**
    - Highlight key terms with definitions
    - Emphasize important formulas and equations
    - Include step-by-step explanations for complex processes
    - Add notes about common misconceptions or tricky points

    **COMPREHENSIVE COVERAGE:**
    - Don't skip important details or assume prior knowledge
    - Include background information where needed
    - Cover both theoretical concepts and practical applications
    - Address different perspectives or approaches when relevant

    At the end, include a detailed 'Key Takeaways' section that summarizes the most critical points, but keep the main content substantial and detailed.

    Your response MUST be a single block of well-formed HTML.

    Follow these rules precisely:
    1.  **Structure:** Use standard HTML tags like `<h1>`, `<h2>`, `<h3>`, `<p>`, `<ul>`, `<ol>`, `<li>`, and `<strong>`.
    2.  **Keywords:** For every important keyword or key term, wrap it in a `<span>` with `class="keyword"` and a `title` attribute containing its detailed definition.
        - **Example:** `<span class="keyword" title="A resource with economic value that is expected to provide a future benefit and can be owned or controlled to produce positive economic value.">Asset</span>`
    3.  **Formulas:** For every mathematical or scientific formula, wrap it in a `<span>` with `class="formula"` and a `data-formula` attribute containing the exact formula as a string.
        - **Example:** `<span class="formula" data-formula="Assets = Liabilities + Equity">Assets = Liabilities + Equity</span>`
    4.  **Key Takeaways:** End with a comprehensive section titled "Key Takeaways" that summarizes the most important points in detail.

    Do not include any text, explanations, or code outside of the final HTML output.

    Here is the text to transform:
    ---
    {text}
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an interactive content architect, skilled at creating rich HTML study materials."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def generate_flashcards_from_text(text, num_cards=20):
    """Generates a specific number of flashcards from text using the AI."""
    prompt = f"""
    Based on the following raw text, create a set of approximately {num_cards} flashcards. Identify key terms, concepts, and questions.
    
    Follow this format strictly for each card:
    Front: [Question or Term]
    Back: [Answer or Definition]
    ---

    Here is the text:
    {text}
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"You are a helpful assistant that creates approximately {num_cards} flashcards from study materials. Separate each flashcard with '---'."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


def generate_quiz_from_text(text, num_questions=10):
    """Generates a quiz with a specific number of questions in JSON format from text using the AI."""
    prompt = f"""
    You are an expert educator. Your task is to create a {num_questions}-question practice quiz from the text below.

    Your response MUST be a single, valid JSON object and nothing else.

    The JSON object must contain one key: "questions", which is an array of question objects.

    **Each question object in the array is REQUIRED to have the following keys:**
    - "topic": A concise topic name (2-3 words) that the question is about. Infer this from the text.
    - "type": Must be either "multiple_choice" or "short_answer".
    - "question": The question itself.
    - "options" and "correct_answer": For "multiple_choice" questions ONLY.
    - "model_answer": For "short_answer" questions ONLY.

    Analyze the following text to generate the quiz:
    ---
    {text}
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content


def generate_exam_questions(text, difficulty, question_type, num_questions):
    """Generates a specific type of question (mcq, short, or essay)."""
    
    if num_questions <= 0:
        return []

    if question_type == "mcq":
        prompt_details = f"""
        - Generate exactly {num_questions} multiple-choice questions.
        - Your response MUST be a valid JSON object with a single key "questions".
        - Each object in the "questions" array MUST contain: "question", "options" (an array of 4 strings), and "correct_answer".
        """
    elif question_type == "short":
        prompt_details = f"""
        - Generate exactly {num_questions} short-answer questions.
        - Your response MUST be a valid JSON object with a single key "questions".
        - Each object in the "questions" array MUST contain: "question" and "model_answer".
        """
    else: # essay
        prompt_details = f"""
        - Generate exactly {num_questions} essay-style questions.
        - Your response MUST be a valid JSON object with a single key "questions".
        - Each object in the "questions" array MUST contain: "question" and "model_answer".
        """

    prompt = f"""
    You are an expert university examiner creating a '{difficulty}' difficulty exam based on the provided text.
    {prompt_details}

    Here is the text to use:
    ---
    {text}
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    try:
        data = json.loads(response.choices[0].message.content)
        return data.get("questions", [])
    except json.JSONDecodeError:
        return []

def analyse_papers_with_ai(text):
    """Analyzes past exam papers to identify trends using the AI."""
    prompt = f"""
    You are an expert academic analyst. Analyze the provided past exam papers to identify key trends for a student. Generate a report in Markdown format with three sections:
    
    ### 1. Common Recurring Topics
    Identify the top 5-7 most frequently tested topics.
    
    ### 2. Likely Future Exam Focus Areas
    Predict 3-4 areas likely to be a major focus in an upcoming exam, with justification.
    
    ### 3. Structural & Difficulty Trends
    Comment on trends in exam format, question style, or difficulty.

    Here is the combined text from the past papers:
    ---
    {text}
    """
    response = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content

def grade_answer_with_ai(question, model_answer, student_answer):
    """Grades a student's open-ended answer against a model answer using the AI."""
    prompt = f"""
    You are an expert examiner grading a student's answer.
    - **Exam Question:** {question}
    - **Model Answer:** {model_answer}
    - **Student's Answer:** {student_answer}

    Your task is to evaluate the student's answer. Your response MUST be a valid JSON object with two keys: "score" (an integer from 0 to 10) and "feedback" (a string with constructive comments).
    Example: {{"score": 8, "feedback": "This is a strong answer..."}}
    """
    response = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"})
    return response.choices[0].message.content

def generate_interactive_heatmap_data(text):
    """Generates topic and per-document weight data for an interactive heatmap."""
    prompt = f"""
    You are an expert academic analyst. Your task is to perform a topic modeling analysis on the provided collection of documents.

    Follow these steps precisely:
    1.  **Identify Main Topics:** First, identify the 5 most significant, overarching topics present across all the documents combined. These topics should be concise phrases (2-4 words).
    2.  **Calculate Topic Weights per Document:** For each individual document, calculate the proportion of each of the 5 main topics. The weights for each document should be normalized so they sum to 1.0.
    3.  **Format Output:** Return a single, valid JSON object with NO other text or explanations. The object must contain two keys:
        - "topics": An array of 5 strings representing the main topic names.
        - "documents": An array of objects, where each object represents a document and has two keys:
            - "name": The document's filename (e.g., "lecture_week_1.pdf").
            - "weights": An array of 5 floating-point numbers corresponding to the weights of the topics in the "topics" array for that specific document.

    Example Response Structure:
    {{
        "topics": ["Quantum Mechanics", "General Relativity", "String Theory", "Thermodynamics", "Particle Physics"],
        "documents": [
            {{
                "name": "physics_paper_1.pdf",
                "weights": [0.6, 0.2, 0.1, 0.05, 0.05]
            }},
            {{
                "name": "cosmology_notes.pdf",
                "weights": [0.1, 0.7, 0.1, 0.05, 0.05]
            }}
        ]
    }}

    Here is the text containing one or more documents, separated by '--- DOCUMENT: ... ---':
    ---
    {text}
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content

def generate_study_plan_from_syllabus(syllabus_text, deadline_str):
    """Analyzes a syllabus and creates a structured study plan as JSON."""
    prompt = f"""
    You are an expert academic advisor. Your task is to create a structured, week-by-week study plan from a syllabus.

    Today's date is {datetime.now(timezone.utc).strftime('%Y-%m-%d')}.
    The final exam/deadline is on {deadline_str}.

    Your response MUST be a single, valid JSON object with one key: "plan".
    The "plan" should be an array of "week" objects.
    Each "week" object MUST contain:
    - "week_number": An integer (e.g., 1).
    - "topic": A concise string describing the main topic for that week (e.g., "Introduction to Microeconomics").
    - "tasks": An array of 2-3 specific, actionable tasks for the week (e.g., "Read Chapter 1 & 2", "Generate flashcards for key terms", "Take a practice quiz on supply and demand").

    Analyze the provided syllabus text to identify the weekly topics and structure. Create a logical progression of tasks for each topic. Distribute the topics evenly across the weeks available until the deadline.

    Syllabus Text:
    ---
    {syllabus_text}
    ---
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content

# ==============================================================================
# 4. SPOTIFY AUTH & API ROUTES
# ==============================================================================

def spotify_api_request(method, endpoint, data=None, json_data=None):
    """A helper function to make authenticated requests to the Spotify API."""
    access_token = get_valid_spotify_token()
    if not access_token:
        return jsonify({"success": False, "error": "Not connected to Spotify or token expired."}), 401

    headers = {'Authorization': f'Bearer {access_token}'}
    
    try:
        if method.upper() == 'GET':
            response = requests.get(SPOTIFY_API_BASE_URL + endpoint, headers=headers, params=data)
        elif method.upper() == 'POST':
            response = requests.post(SPOTIFY_API_BASE_URL + endpoint, headers=headers, json=json_data, data=data)
        elif method.upper() == 'PUT':
            response = requests.put(SPOTIFY_API_BASE_URL + endpoint, headers=headers, json=json_data, data=data)
        else:
            return jsonify({"success": False, "error": "Unsupported HTTP method"}), 400
        
        response.raise_for_status() # Raises an exception for bad status codes (4xx or 5xx)
        
        # Some Spotify API responses have no content (e.g., PUT for play/pause)
        if response.status_code == 204:
            return jsonify({"success": True, "status_code": 204})
        return response.json()
        
    except requests.exceptions.HTTPError as e:
        print(f"Spotify API HTTP Error: {e.response.status_code} - {e.response.text}")
        if e.response.status_code == 401:
            # Token might have been revoked, clear it
            db.collection('users').document(current_user.id).update({
                'spotify_access_token': None,
                'spotify_refresh_token': None,
                'spotify_token_expires_at': None
            })
        return jsonify({"success": False, "error": f"Spotify API Error: {e.response.reason}", "status_code": e.response.status_code}), e.response.status_code
    except Exception as e:
        print(f"Spotify API General Error: {e}")
        return jsonify({"success": False, "error": "An unexpected error occurred"}), 500

def get_valid_spotify_token():
    """Checks if the current user's token is valid, refreshes if not, and returns a valid token."""
    if not current_user.is_authenticated or not current_user.spotify_refresh_token:
        return None

    # Check if the token is expired or expires within the next 60 seconds
    if not current_user.spotify_token_expires_at or current_user.spotify_token_expires_at <= datetime.now(timezone.utc) + timedelta(seconds=60):
        try:
            auth_str = f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}"
            b64_auth_str = base64.b64encode(auth_str.encode()).decode()
            
            response = requests.post(SPOTIFY_TOKEN_URL, data={
                'grant_type': 'refresh_token',
                'refresh_token': current_user.spotify_refresh_token
            }, headers={
                'Authorization': f'Basic {b64_auth_str}'
            })
            response.raise_for_status()
            token_info = response.json()
            
            new_access_token = token_info['access_token']
            expires_in = token_info['expires_in']
            new_expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
            
            # Update user in database and current session
            user_ref = db.collection('users').document(current_user.id)
            user_ref.update({
                'spotify_access_token': new_access_token,
                'spotify_token_expires_at': new_expires_at
            })
            current_user.spotify_access_token = new_access_token
            current_user.spotify_token_expires_at = new_expires_at
            
            return new_access_token
        except Exception as e:
            print(f"Error refreshing Spotify token: {e}")
            return None
            
    return current_user.spotify_access_token
    
def spotify_connected(f):
    """Decorator to ensure user is connected to Spotify."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not get_valid_spotify_token():
            return jsonify({"success": False, "error": "Spotify not connected."}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route('/spotify/login')
@login_required
def spotify_login():
    params = {
        'client_id': SPOTIFY_CLIENT_ID,
        'response_type': 'code',
        'redirect_uri': SPOTIFY_REDIRECT_URI,
        'scope': SPOTIFY_SCOPES,
        'show_dialog': 'true'
    }
    auth_url = f"{SPOTIFY_AUTH_URL}?{urllib.parse.urlencode(params)}"
    return redirect(auth_url)

@app.route('/spotify/callback')
@login_required
def spotify_callback():
    code = request.args.get('code')
    error = request.args.get('error')

    if error:
        flash(f"Spotify connection failed: {error}", "error")
        return redirect(url_for('dashboard'))

    auth_str = f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}"
    b64_auth_str = base64.b64encode(auth_str.encode()).decode()

    response = requests.post(SPOTIFY_TOKEN_URL, data={
        'grant_type': 'authorization_code',
        'code': code,
        'redirect_uri': SPOTIFY_REDIRECT_URI
    }, headers={
        'Authorization': f'Basic {b64_auth_str}'
    })
    
    token_info = response.json()
    
    user_ref = db.collection('users').document(current_user.id)
    user_ref.update({
        'spotify_access_token': token_info['access_token'],
        'spotify_refresh_token': token_info['refresh_token'],
        'spotify_token_expires_at': datetime.now(timezone.utc) + timedelta(seconds=token_info['expires_in'])
    })

    flash("Successfully connected your Spotify account!", "success")
    return redirect(url_for('dashboard'))

# --- NEW: Endpoint for the Web Playback SDK ---
@app.route('/spotify/access_token')
@login_required
def get_spotify_access_token():
    token = get_valid_spotify_token()
    if token:
        return jsonify({'access_token': token})
    else:
        return jsonify({'error': 'User not connected to Spotify or token refresh failed'}), 401

@app.route('/spotify/player_state')
@login_required
@spotify_connected
def player_state():
    return spotify_api_request('GET', 'me/player')

@app.route('/spotify/playlists')
@login_required
@spotify_connected
def get_playlists():
    return spotify_api_request('GET', 'me/playlists', data={'limit': 50})

@app.route('/spotify/play', methods=['PUT'])
@login_required
@spotify_connected
def play():
    device_id = request.json.get('device_id')
    context_uri = request.json.get('context_uri')
    
    endpoint = 'me/player/play'
    if device_id:
        endpoint += f'?device_id={device_id}'
        
    data = {}
    if context_uri:
        data['context_uri'] = context_uri
        
    return spotify_api_request('PUT', endpoint, json_data=data)

@app.route('/spotify/pause', methods=['PUT'])
@login_required
@spotify_connected
def pause():
    return spotify_api_request('PUT', 'me/player/pause')

@app.route('/spotify/next', methods=['POST'])
@login_required
@spotify_connected
def next_track():
    return spotify_api_request('POST', 'me/player/next')

@app.route('/spotify/previous', methods=['POST'])
@login_required
@spotify_connected
def previous_track():
    return spotify_api_request('POST', 'me/player/previous')

@app.route('/spotify/volume', methods=['PUT'])
@login_required
@spotify_connected
def set_volume():
    volume_percent = request.json.get('volume_percent')
    return spotify_api_request('PUT', f'me/player/volume?volume_percent={volume_percent}')

# --- Phase 2: Advanced Spotify Features ---
@app.route('/spotify/search')
@login_required
@spotify_connected
def search_spotify():
    """Search user's Spotify library"""
    query = request.args.get('q', '')
    if not query:
        return jsonify({'error': 'Query parameter required'}), 400
    
    # Search for tracks, albums, artists, and playlists
    search_params = {
        'q': query,
        'type': 'track,album,artist,playlist',
        'limit': 20
    }
    
    return spotify_api_request('GET', 'search', data=search_params)

@app.route('/spotify/recommendations')
@login_required
@spotify_connected
def get_recommendations():
    """Get AI-powered music recommendations based on current track"""
    seed_tracks = request.args.get('seed_tracks', '')
    if not seed_tracks:
        return jsonify({'error': 'seed_tracks parameter required'}), 400
    
    # Get recommendations with study-focused parameters
    rec_params = {
        'seed_tracks': seed_tracks,
        'limit': 10,
        'target_energy': 0.5,  # Moderate energy for study
        'target_valence': 0.6,  # Slightly positive mood
        'target_acousticness': 0.3,  # Mix of acoustic and electronic
        'target_instrumentalness': 0.4,  # Some instrumental tracks
        'target_tempo': 120,  # Moderate tempo
        'min_popularity': 20  # Not too obscure
    }
    
    return spotify_api_request('GET', 'recommendations', data=rec_params)

@app.route('/spotify/audio-features/<track_id>')
@login_required
@spotify_connected
def get_audio_features(track_id):
    """Get detailed audio features for a track"""
    return spotify_api_request('GET', f'audio-features/{track_id}')

@app.route('/spotify/currently-playing')
@login_required
@spotify_connected
def get_currently_playing():
    """Get currently playing track with full details"""
    return spotify_api_request('GET', 'me/player/currently-playing')

# --- Homepage and Auth Routes ---
@app.route("/")
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template("index.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        users_ref = db.collection('users').where('email', '==', email).limit(1).stream()
        user_list = list(users_ref)

        if not user_list:
            flash('Invalid email or password.')
            return redirect(url_for('login'))

        user_data = user_list[0].to_dict()
        user = User.from_dict(user_data)
        
        if not check_password_hash(user.password_hash, password):
            flash('Invalid email or password.')
            return redirect(url_for('login'))
        
        login_user(user, remember=True)
        return redirect(url_for('dashboard'))

    return render_template('login.html')

# --- NEW: Phone Verification Routes ---
@app.route('/send-verification-code', methods=['POST'])
def send_verification_code():
    """Send SMS verification code to phone number"""
    try:
        data = request.get_json()
        phone_number = data.get('phone_number')
        
        if not phone_number:
            return jsonify({"success": False, "message": "Phone number is required"}), 400
        
        # Validate phone number format (basic validation)
        if not re.match(r'^\+?[1-9]\d{1,14}$', phone_number.replace(' ', '').replace('-', '')):
            return jsonify({"success": False, "message": "Invalid phone number format"}), 400
        
        # Check if phone number is already registered
        users_ref = db.collection('users').where('phone_number', '==', phone_number).limit(1).stream()
        if list(users_ref):
            return jsonify({"success": False, "message": "Phone number already registered"}), 400
        
        # Generate 6-digit verification code
        verification_code = str(random.randint(100000, 999999))
        
        # Store verification code in session (expires in 5 minutes)
        session[f'verification_code_{phone_number}'] = verification_code
        session[f'verification_expiry_{phone_number}'] = datetime.now().timestamp() + 300  # 5 minutes
        
        # Send SMS via Twilio
        if twilio_client:
            try:
                message = twilio_client.messages.create(
                    body=f'Your Foci verification code is: {verification_code}. This code expires in 5 minutes.',
                    from_=TWILIO_PHONE_NUMBER,
                    to=phone_number
                )
                return jsonify({"success": True, "message": "Verification code sent successfully"})
            except Exception as e:
                print(f"Twilio error: {e}")
                return jsonify({"success": False, "message": "Failed to send verification code. Please try again."}), 500
        else:
            # Development mode - just log the code
            print(f"DEV MODE: Verification code for {phone_number} is: {verification_code}")
            return jsonify({"success": True, "message": f"Verification code sent (DEV: {verification_code})"})
            
    except Exception as e:
        print(f"Error sending verification code: {e}")
        return jsonify({"success": False, "message": "An error occurred. Please try again."}), 500

@app.route('/verify-phone-code', methods=['POST'])
def verify_phone_code():
    """Verify the SMS code entered by user"""
    try:
        data = request.get_json()
        phone_number = data.get('phone_number')
        entered_code = data.get('code')
        
        if not phone_number or not entered_code:
            return jsonify({"success": False, "message": "Phone number and code are required"}), 400
        
        # Check if verification code exists and hasn't expired
        stored_code = session.get(f'verification_code_{phone_number}')
        expiry_time = session.get(f'verification_expiry_{phone_number}', 0)
        
        if not stored_code:
            return jsonify({"success": False, "message": "No verification code found. Please request a new one."}), 400
        
        if datetime.now().timestamp() > expiry_time:
            # Clean up expired code
            session.pop(f'verification_code_{phone_number}', None)
            session.pop(f'verification_expiry_{phone_number}', None)
            return jsonify({"success": False, "message": "Verification code expired. Please request a new one."}), 400
        
        if stored_code != entered_code:
            return jsonify({"success": False, "message": "Invalid verification code"}), 400
        
        # Code is valid - mark phone as verified in session
        session[f'phone_verified_{phone_number}'] = True
        session.pop(f'verification_code_{phone_number}', None)
        session.pop(f'verification_expiry_{phone_number}', None)
        
        return jsonify({"success": True, "message": "Phone number verified successfully"})
        
    except Exception as e:
        print(f"Error verifying phone code: {e}")
        return jsonify({"success": False, "message": "An error occurred. Please try again."}), 500

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        phone_number = request.form.get('phone_number')

        # Validate required fields
        if not all([email, password, phone_number]):
            flash('All fields are required.')
            return redirect(url_for('signup'))

        # Check if email already exists
        users_ref = db.collection('users').where('email', '==', email).limit(1).stream()
        if list(users_ref):
            flash('Email address already exists.')
            return redirect(url_for('signup'))
        
        # Check if phone number is already registered
        phone_ref = db.collection('users').where('phone_number', '==', phone_number).limit(1).stream()
        if list(phone_ref):
            flash('Phone number already registered.')
            return redirect(url_for('signup'))
        
        # Verify phone number was verified in session (skip in development)
        if not session.get(f'phone_verified_{phone_number}') and not app.debug:
            flash('Please verify your phone number first.')
            return redirect(url_for('signup'))
        
        # --- NEW: Handle Referral Code ---
        referral_code = request.form.get('referral_code', '').strip()
        referred_by = None
        
        if referral_code:
            referrer_user = validate_referral_code(referral_code)
            if referrer_user:
                referred_by = referrer_user.id
                print(f"Valid referral code {referral_code} from user {referrer_user.email}")
            else:
                flash('Invalid referral code.')
                return redirect(url_for('signup'))
        
        password_hash = generate_password_hash(password, method='pbkdf2:sha256')
        
        # Generate unique referral code for new user
        new_referral_code = generate_referral_code()
        
        user_ref = db.collection('users').document()
        new_user = User(
            id=user_ref.id, 
            email=email, 
            password_hash=password_hash,
            phone_number=phone_number,
            phone_verified=True,
            has_completed_onboarding=False, # Explicitly set for new users
            referral_code=new_referral_code, # Generate unique code for new user
            referred_by=referred_by # Set if they were referred
        )
        user_ref.set(new_user.to_dict())

        # --- NEW: Create Referral Record if applicable ---
        if referred_by:
            referral_ref = db.collection('referrals').document()
            referral = Referral(
                id=referral_ref.id,
                referrer_id=referred_by,
                referred_id=new_user.id,
                referral_code=referral_code,
                status='pending'
            )
            referral_ref.set(referral.to_dict())
            print(f"Created referral record: {referral_ref.id}")

        # --- ONBOARDING LOGIC ---
        # 1. Create the Welcome Hub
        hub_ref = db.collection('hubs').document()
        welcome_hub = Hub(
            id=hub_ref.id,
            name="My First Hub: Welcome to Foci!",
            user_id=new_user.id
        )
        
        # 2. Upload the welcome PDF from static assets
        try:
            welcome_pdf_path = os.path.join(app.root_path, 'static', 'assets', 'welcome_to_foci.pdf')
            
            if os.path.exists(welcome_pdf_path):
                file_path_in_bucket = f"hubs/{hub_ref.id}/Welcome_to_Foci.pdf"
                blob = bucket.blob(file_path_in_bucket)
                
                with open(welcome_pdf_path, 'rb') as f:
                    blob.upload_from_file(f, content_type='application/pdf')
                
                file_size = os.path.getsize(welcome_pdf_path)
                file_info = {'name': 'Welcome_to_Foci.pdf', 'path': file_path_in_bucket, 'size': file_size}
                
                # 3. Update the hub document with the file info
                welcome_hub.files = [file_info]
            else:
                print(f"WARNING: 'welcome_to_foci.pdf' not found at {welcome_pdf_path}. The welcome hub will be empty.")
        
        except Exception as e:
            print(f"An error occurred during welcome hub creation for user {new_user.id}: {e}")
        
        hub_ref.set(welcome_hub.to_dict())
        # --- END ONBOARDING LOGIC ---

        # Clean up phone verification session
        session.pop(f'phone_verified_{phone_number}', None)
        
        login_user(new_user, remember=True)
        return redirect(url_for('dashboard'))

    return render_template('signup.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

# --- Main Application Routes (UPDATED) ---

@app.route("/dashboard")
@login_required
def dashboard():
    # --- Part 1: Hubs List for the main dashboard view ---
    hubs_ref = db.collection('hubs').where('user_id', '==', current_user.id).stream()
    hubs_list = [Hub.from_dict(doc.to_dict()) for doc in hubs_ref]
    
    # --- Part 2: Data aggregation for the Progress Tab ---
    total_study_minutes = 0
    longest_streak = 0
    all_quiz_scores = []
    topic_performance = {}

    for hub in hubs_list:
        if hub.streak_days > longest_streak:
            longest_streak = hub.streak_days

        # Aggregate study time (assuming 30 mins per session for now)
        sessions_query = db.collection('sessions').where('hub_id', '==', hub.id).stream()
        for session_doc in sessions_query:
            total_study_minutes += 30 
        
        # Aggregate quiz scores and topic performance
        activities_query = db.collection('activities').where(filter=firestore.FieldFilter('hub_id', '==', hub.id)).where(filter=firestore.FieldFilter('status', '==', 'graded')).stream()
        for activity_doc in activities_query:
            activity = Activity.from_dict(activity_doc.to_dict())
            if 'Quiz' in activity.type or 'Exam' in activity.type:
                all_quiz_scores.append({
                    'x': activity.created_at.isoformat(),
                    'y': activity.score
                })
                
                for answer in activity.graded_results.get('graded_answers', []):
                    topic = answer.get('topic', 'General')
                    if topic not in topic_performance:
                        topic_performance[topic] = {'scores': [], 'hub_name': hub.name}
                    
                    score = answer.get('score', 10 if answer.get('correct') else 0)
                    topic_performance[topic]['scores'].append(score)

    total_study_hours = round(total_study_minutes / 60, 1)
    
    # Calculate average scores and identify top 5 weaknesses
    weak_topics = []
    for topic, data in topic_performance.items():
        if data['scores']:
            average_score = (sum(data['scores']) / (len(data['scores']) * 10)) * 100
            if average_score < 70:
                weak_topics.append({
                    'topic': topic,
                    'score': average_score,
                    'hub_name': data['hub_name']
                })
            
    weak_topics = sorted(weak_topics, key=lambda x: x['score'])[:5]

    # --- UPDATED: Fetch data for the Community Tab ---
    shared_folders_hydrated = []
    total_shared_count = 0
    if current_user.subscription_tier in ['pro', 'admin']:
        
        all_shared_docs = db.collection('shared_folders').stream()
        total_shared_count = len(list(all_shared_docs))

        query = db.collection('shared_folders')
        sort_by = request.args.get('sort', 'created_at')
        if sort_by == 'likes':
            query = query.order_by('likes', direction=firestore.Query.DESCENDING)
        elif sort_by == 'imports':
            query = query.order_by('imports', direction=firestore.Query.DESCENDING)
        else:
            query = query.order_by('created_at', direction=firestore.Query.DESCENDING)
        
        shared_folders_docs = query.limit(50).stream()
        shared_folders = [SharedFolder.from_dict(doc.to_dict()) for doc in shared_folders_docs]
        
        # 1. Collect all IDs needed for hydration
        owner_ids = list(set(sf.owner_id for sf in shared_folders))
        original_folder_ids = list(set(sf.original_folder_id for sf in shared_folders))
        
        # 2. Batch fetch user and folder data
        users_data = {}
        if owner_ids:
            user_docs = db.collection('users').where('id', 'in', owner_ids).stream()
            users_data = {user.to_dict()['id']: user.to_dict() for user in user_docs}

        original_folders_data = {}
        if original_folder_ids:
            folder_docs = db.collection('folders').where('id', 'in', original_folder_ids).stream()
            original_folders_data = {f.to_dict()['id']: f.to_dict() for f in folder_docs}

        # 3. NEW: Batch fetch all items (notes/activities) for previews
        all_item_ids = {'note': [], 'activity': []}
        for folder_id, folder_info in original_folders_data.items():
            for item in folder_info.get('items', []):
                item_id = item.get('id')
                if item.get('type') == 'note':
                    all_item_ids['note'].append(item_id)
                elif item.get('type') in ['quiz', 'flashcards']:
                    all_item_ids['activity'].append(item_id)
        
        all_items_data = {}
        if all_item_ids['note']:
            note_docs = db.collection('notes').where('id', 'in', list(set(all_item_ids['note']))).stream()
            for doc in note_docs: all_items_data[doc.id] = doc.to_dict()
        if all_item_ids['activity']:
            activity_docs = db.collection('activities').where('id', 'in', list(set(all_item_ids['activity']))).stream()
            for doc in activity_docs: all_items_data[doc.id] = doc.to_dict()

        hydrated_cache = {}

        def _hydrate_folder_info(sf):
            if sf.id in hydrated_cache: return hydrated_cache[sf.id]

            owner_info = users_data.get(sf.owner_id)
            original_folder_info = original_folders_data.get(sf.original_folder_id)
            
            item_count = 0
            folder_type = "Pack"
            items_preview = []

            if original_folder_info:
                items = original_folder_info.get('items', [])
                item_count = len(items)

                for item in items:
                    item_data = all_items_data.get(item.get('id'))
                    if item_data:
                        item_type_map = {'note': 'Note', 'Quiz': 'Quiz', 'Flashcards': 'Flashcards'}
                        raw_type = item_data.get('type', item.get('type')) # Handles both 'note' and 'Quiz'
                        preview_type = item_type_map.get(raw_type, 'Item')
                        
                        # --- THIS IS THE KEY CHANGE ---
                        # We now include the item ID so the frontend can fetch its content.
                        items_preview.append({
                            'id': item_data.get('id'), # ADDED
                            'title': item_data.get('title', 'Untitled'), 
                            'type': preview_type
                        })
                        # --- END OF CHANGE ---
                
                types = {item.get('type') for item in items}
                if len(types) == 1:
                    type_map = {'note': 'Notes', 'quiz': 'Quiz', 'flashcards': 'Flashcards'}
                    folder_type = type_map.get(types.pop(), "Pack")
            
            hydrated_info = {
                'folder': sf,
                'owner_name': owner_info.get('display_name', 'Unknown User') if owner_info else 'Unknown User',
                'owner_avatar': owner_info.get('avatar_url', 'default_avatar.png') if owner_info else 'default_avatar.png',
                'item_count': item_count,
                'folder_type': folder_type,
                'items_preview': items_preview,
                'current_user_has_liked': current_user.id in sf.liked_by,
                'current_user_has_imported': current_user.id in sf.imported_by
            }
            hydrated_cache[sf.id] = hydrated_info
            return hydrated_info

        shared_folders_hydrated = [_hydrate_folder_info(sf) for sf in shared_folders]

    all_user_folders = []
    for hub in hubs_list:
        folders_query = db.collection('folders').where('hub_id', '==', hub.id).stream()
        for folder_doc in folders_query:
            folder = Folder.from_dict(folder_doc.to_dict())
            all_user_folders.append({"id": folder.id, "name": folder.name, "hub_name": hub.name, "hub_id": hub.id})

    hubs_for_json = [hub.to_dict() for hub in hubs_list]
    stats = _get_user_stats(current_user.id)
    spotify_connected_status = True if current_user.spotify_refresh_token else False

    # --- NEW: Onboarding Flag ---
    needs_onboarding = not current_user.has_completed_onboarding
    print(f"Dashboard: User {current_user.id} has_completed_onboarding = {current_user.has_completed_onboarding}, needs_onboarding = {needs_onboarding}")

    return render_template(
        "dashboard.html", 
        hubs=hubs_list,
        hubs_for_json=hubs_for_json,
        total_study_hours=total_study_hours,
        longest_streak=longest_streak,
        quiz_scores_json=json.dumps(all_quiz_scores),
        weak_topics=weak_topics,
        shared_folders=shared_folders_hydrated,
        total_shared_count=total_shared_count,
        all_user_folders=all_user_folders,
        stats=stats,
        spotify_connected=spotify_connected_status,
        needs_onboarding=needs_onboarding # Pass the flag to the template
    )

# --- NEW: API Route to fetch item content for preview ---
@app.route("/api/item_content/<item_id>")
@login_required
def get_item_preview_content(item_id):
    """
    Fetches the content of a note or activity for the preview modal.
    This is a simplified security model; it assumes any item ID passed
    is from a visible shared folder on the community page.
    """
    try:
        # Check if it's a note
        note_doc = db.collection('notes').document(item_id).get()
        if note_doc.exists:
            note_data = note_doc.to_dict()
            return jsonify({
                "success": True, 
                "content": note_data.get('content_html', 'No content available.')
            })

        # Check if it's an activity (flashcards)
        activity_doc = db.collection('activities').document(item_id).get()
        if activity_doc.exists:
            activity_data = activity_doc.to_dict()
            if activity_data.get('type') == 'Flashcards':
                 return jsonify({
                    "success": True,
                    "content": activity_data.get('data', {}).get('cards', [])
                })
            else: # It's a quiz, which we don't preview the content of
                return jsonify({
                    "success": True,
                    "content": "Preview is not available for quizzes."
                })
        
        # If not found in either collection
        return jsonify({"success": False, "message": "Item not found."}), 404

    except Exception as e:
        print(f"Error fetching item content for preview {item_id}: {e}")
        return jsonify({"success": False, "message": "An error occurred."}), 500


@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    user_ref = db.collection('users').document(current_user.id)

    if request.method == 'POST':
        display_name = request.form.get('display_name')
        bio = request.form.get('bio')
        update_data = {
            'display_name': display_name,
            'bio': bio
        }
        
        if 'avatar' in request.files:
            file = request.files['avatar']
            if file.filename != '':
                filename = secure_filename(file.filename)
                file_path = f"avatars/{current_user.id}/{filename}"
                blob = bucket.blob(file_path)
                
                blob.upload_from_file(file, content_type=file.content_type)
                blob.make_public()
                update_data['avatar_url'] = blob.public_url

        user_ref.update(update_data)
        flash('Your profile has been updated successfully!', 'success')
        return redirect(url_for('dashboard', _anchor='profile'))

    # For GET requests, redirect to the dashboard's profile tab
    return redirect(url_for('dashboard', _anchor='profile'))

@app.route("/add_hub", methods=["POST"])
@login_required
def add_hub():
    hub_name = request.form.get('hub_name')
    if hub_name:
        hub_ref = db.collection('hubs').document()
        
        # UPDATED: We now pass the current_user.id when creating a new Hub
        new_hub = Hub(
            id=hub_ref.id, 
            name=hub_name, 
            user_id=current_user.id # This is the crucial change
        )
        hub_ref.set(new_hub.to_dict())
    return redirect(url_for('dashboard'))

@app.route("/onboarding/create_demo_hub", methods=["POST"])
@login_required
def create_demo_hub():
    """Creates a demo hub with sample content for onboarding."""
    try:
        # Create the demo hub
        hub_ref = db.collection('hubs').document()
        demo_hub = Hub(
            id=hub_ref.id, 
            name="Welcome to FociAI Hub", 
            user_id=current_user.id,
            color="#6366f1"
        )
        db.collection('hubs').document(hub_ref.id).set(demo_hub.to_dict())
        
        # Create sample content for the demo hub
        batch = db.batch()
        
        # Sample lecture content
        sample_lecture_text = """
        Introduction to Machine Learning
        
        Machine learning is a subset of artificial intelligence that focuses on the development of algorithms and statistical models that enable computer systems to improve their performance on a specific task through experience, without being explicitly programmed.
        
        Key Concepts:
        1. Supervised Learning: Learning with labeled training data
        2. Unsupervised Learning: Finding patterns in data without labels
        3. Reinforcement Learning: Learning through interaction with an environment
        
        Applications:
        - Image recognition and computer vision
        - Natural language processing
        - Recommendation systems
        - Autonomous vehicles
        - Medical diagnosis
        
        The machine learning process typically involves:
        1. Data collection and preprocessing
        2. Feature selection and engineering
        3. Model selection and training
        4. Model evaluation and validation
        5. Deployment and monitoring
        """
        
        # Create sample note
        note_ref = db.collection('notes').document()
        sample_note = Note(
            id=note_ref.id,
            hub_id=hub_ref.id,
            title="Introduction to Machine Learning",
            content_html=f"<h1>Introduction to Machine Learning</h1><p>{sample_lecture_text.replace(chr(10), '</p><p>')}</p>",
            created_at=datetime.now(timezone.utc)
        )
        batch.set(note_ref, sample_note.to_dict())
        
        # Upload the actual sample PDF file to Firebase Storage
        import os
        
        # Use the existing Firebase Storage bucket
        try:
            bucket = storage.bucket()
        except Exception as e:
            print(f"Firebase Storage not available: {e}")
            bucket = None
        
        # Path to the sample PDF in static directory
        sample_pdf_path = os.path.join('static', 'EC131 Week 9.pdf')
        
        if os.path.exists(sample_pdf_path) and bucket:
            # Upload the PDF to Firebase Storage
            blob_name = f'hubs/{hub_ref.id}/EC131_Week_9.pdf'
            blob = bucket.blob(blob_name)
            
            with open(sample_pdf_path, 'rb') as pdf_file:
                blob.upload_from_file(pdf_file, content_type='application/pdf')
            
            # Get file size
            file_size = os.path.getsize(sample_pdf_path)
            
            # Create file reference in database
            sample_file_ref = db.collection('files').document()
            sample_file = {
                'id': sample_file_ref.id,
                'hub_id': hub_ref.id,
                'filename': 'EC131_Week_9.pdf',
                'original_filename': 'EC131 Week 9.pdf',
                'file_path': blob_name,
                'file_size': file_size,
                'upload_date': datetime.now(timezone.utc),
                'file_type': 'pdf',
                'content': sample_lecture_text,  # Keep the text content for AI processing
                'is_sample': True  # Mark as sample file for onboarding
            }
            batch.set(sample_file_ref, sample_file)
            
            # Update the hub's files field to include the sample file
            hub_ref.update({
                'files': [{
                    'name': 'EC131 Week 9.pdf',
                    'path': blob_name,
                    'size': file_size
                }]
            })
            
            print(f"✅ Sample PDF uploaded successfully: {blob_name}")
        else:
            print(f"❌ Sample PDF not found at: {sample_pdf_path}")
            # Fallback to the old method if file doesn't exist
            sample_file_ref = db.collection('files').document()
            sample_file = {
                'id': sample_file_ref.id,
                'hub_id': hub_ref.id,
                'filename': 'machine_learning_intro.pdf',
                'original_filename': 'machine_learning_intro.pdf',
                'file_path': f'hubs/{hub_ref.id}/machine_learning_intro.pdf',
                'file_size': 1024000,  # 1MB
                'upload_date': datetime.now(timezone.utc),
                'file_type': 'pdf',
                'content': sample_lecture_text,
                'is_sample': True
            }
            batch.set(sample_file_ref, sample_file)
            
            hub_ref.update({
                'files': [{
                    'name': 'machine_learning_intro.pdf',
                    'path': f'hubs/{hub_ref.id}/machine_learning_intro.pdf',
                    'size': 1024000
                }]
            })
        
        # Create sample flashcards
        flashcard_ref = db.collection('activities').document()
        sample_flashcards = [
            {"front": "What is machine learning?", "back": "A subset of AI that enables computers to learn and improve from experience without explicit programming."},
            {"front": "What is supervised learning?", "back": "Learning with labeled training data where the algorithm learns to map inputs to outputs."},
            {"front": "What is unsupervised learning?", "back": "Finding patterns in data without labeled examples or guidance."},
            {"front": "What is reinforcement learning?", "back": "Learning through interaction with an environment using rewards and penalties."},
            {"front": "Name three applications of machine learning", "back": "Image recognition, natural language processing, recommendation systems, autonomous vehicles, medical diagnosis."}
        ]
        sample_flashcard_activity = Activity(
            id=flashcard_ref.id,
            hub_id=hub_ref.id,
            type='Flashcards',
            title='Machine Learning Basics',
            data={'cards': sample_flashcards},
            status='completed',
            created_at=datetime.now(timezone.utc)
        )
        batch.set(flashcard_ref, sample_flashcard_activity.to_dict())
        
        # Create sample quiz
        quiz_ref = db.collection('activities').document()
        sample_quiz = Activity(
            id=quiz_ref.id,
            hub_id=hub_ref.id,
            type='Quiz',
            title='Machine Learning Quiz',
            data={
                'questions': [
                    {
                        'question': 'Which type of learning uses labeled training data?',
                        'options': ['Supervised Learning', 'Unsupervised Learning', 'Reinforcement Learning', 'Deep Learning'],
                        'correct_answer': 'Supervised Learning'
                    },
                    {
                        'question': 'What is the main goal of machine learning?',
                        'options': ['To replace human intelligence', 'To improve performance through experience', 'To create robots', 'To write code automatically'],
                        'correct_answer': 'To improve performance through experience'
                    }
                ]
            },
            status='completed',
            created_at=datetime.now(timezone.utc)
        )
        batch.set(quiz_ref, sample_quiz.to_dict())
        
        batch.commit()
        
        return jsonify({
            "success": True, 
            "hub_id": hub_ref.id,
            "message": "Demo hub created successfully!"
        })
        
    except Exception as e:
        print(f"Error creating demo hub: {e}")
        return jsonify({"success": False, "message": "Failed to create demo hub."}), 500


@app.route("/hub/<hub_id>")
@login_required
def hub_page(hub_id):
    hub_doc = db.collection('hubs').document(hub_id).get()
    if not hub_doc.exists:
        flash("Hub not found.", "error")
        return redirect(url_for('dashboard'))
    
    hub = Hub.from_dict(hub_doc.to_dict())

    # SECURITY CHECK: Ensure the logged-in user owns this hub
    if hub.user_id != current_user.id:
        flash("You do not have permission to view this hub.", "error")
        return redirect(url_for('dashboard'))

    def clean_undefined_values(obj):
        """Recursively clean Undefined values from any object."""
        if isinstance(obj, dict):
            return {k: clean_undefined_values(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_undefined_values(item) for item in obj]
        elif str(obj) == 'Undefined' or obj is None or hasattr(obj, '__class__') and 'Undefined' in str(obj.__class__):
            return None
        else:
            return obj

    activities_query = db.collection('activities').where(filter=firestore.FieldFilter('hub_id', '==', hub_id)).stream()
    all_activities = []
    for doc in activities_query:
        try:
            activity_data = doc.to_dict()
            # Clean any Undefined values from the data recursively
            cleaned_data = clean_undefined_values(activity_data)
            
            activity = Activity.from_dict(cleaned_data)
            all_activities.append(activity)
        except Exception as e:
            print(f"Error processing activity {doc.id}: {e}")
            continue
    
    notes_query = db.collection('notes').where(filter=firestore.FieldFilter('hub_id', '==', hub_id)).stream()
    all_notes = []
    for note in notes_query:
        try:
            note_data = note.to_dict()
            # Clean any Undefined values from the data recursively
            cleaned_data = clean_undefined_values(note_data)
            
            note_obj = Note.from_dict(cleaned_data)
            all_notes.append(note_obj)
        except Exception as e:
            print(f"Error processing note {note.id}: {e}")
            continue
    
    sessions_query = db.collection('sessions').where(filter=firestore.FieldFilter('hub_id', '==', hub_id)).order_by('created_at', direction=firestore.Query.DESCENDING).stream()
    all_sessions = []
    for doc in sessions_query:
        try:
            session_data = doc.to_dict()
            # Clean any Undefined values from the data recursively
            cleaned_data = clean_undefined_values(session_data)
            
            session = StudySession.from_dict(cleaned_data)
            all_sessions.append(session)
        except Exception as e:
            print(f"Error processing session {doc.id}: {e}")
            continue

    folders_query = db.collection('folders').where(filter=firestore.FieldFilter('hub_id', '==', hub_id)).order_by('created_at', direction=firestore.Query.DESCENDING).stream()
    all_folders = []
    for doc in folders_query:
        try:
            folder_data = doc.to_dict()
            # Clean any Undefined values from the data recursively
            cleaned_data = clean_undefined_values(folder_data)
            
            folder = Folder.from_dict(cleaned_data)
            all_folders.append(folder)
        except Exception as e:
            print(f"Error processing folder {doc.id}: {e}")
            continue

    # --- NEW: Fetch Annotated Slide Decks ---
    slide_notes_query = db.collection('annotated_slide_decks').where(filter=firestore.FieldFilter('hub_id', '==', hub_id)).order_by('created_at', direction=firestore.Query.DESCENDING).stream()
    all_slide_notes = []
    for doc in slide_notes_query:
        try:
            slide_data = doc.to_dict()
            # Clean any Undefined values from the data recursively
            cleaned_data = clean_undefined_values(slide_data)
            
            slide_note = AnnotatedSlideDeck.from_dict(cleaned_data)
            all_slide_notes.append(slide_note)
        except Exception as e:
            print(f"Error processing slide note {doc.id}: {e}")
            continue

    notes_map = {note.id: note for note in all_notes}
    activities_map = {activity.id: activity for activity in all_activities}

    for folder in all_folders:
        hydrated_items = []
        for item_ref in folder.items:
            doc_id = item_ref.get('id')
            doc_type = item_ref.get('type')
            
            item = None
            if doc_type == 'note' and doc_id in notes_map:
                item = notes_map[doc_id]
            elif doc_type in ['quiz', 'flashcards', 'notes'] and doc_id in activities_map:
                item = activities_map[doc_id]
            
            if item:
                hydrated_items.append(item)
        
        folder.hydrated_items = hydrated_items

    # Get all activity IDs that are part of folders (packs)
    pack_activity_ids = set()
    for folder in all_folders:
        for item in folder.items:
            pack_activity_ids.add(item.get('id'))
    
    # Filter out pack activities from individual collections
    all_flashcards = [
        activity for activity in all_activities 
        if activity.type == 'Flashcards' and activity.id not in pack_activity_ids
    ]
    graded_activities = [activity for activity in all_activities if activity.status == 'graded']
    
    # --- FIX: Create a list of all quizzes/exams, not just graded ones ---
    all_quizzes_and_exams = [
        activity for activity in all_activities 
        if ('Quiz' in activity.type or 'Exam' in activity.type) and activity.id not in pack_activity_ids
    ]
    
    # Filter out pack notes from individual notes collection
    # Pack notes are stored as activities with note content, so we need to exclude them from all_notes
    pack_note_activity_ids = set()
    for activity in all_activities:
        if activity.id in pack_activity_ids and activity.data and activity.data.get('type') == 'note':
            pack_note_activity_ids.add(activity.id)
    
    # Filter out notes that are part of packs
    all_notes = [note for note in all_notes if note.id not in pack_note_activity_ids]
    
    # ADD THIS NEW CODE BLOCK
    recent_quiz_scores = []
    for activity in graded_activities:
     if 'Quiz' in activity.type or 'Exam' in activity.type:
        recent_quiz_scores.append({
            'x': activity.created_at.isoformat(),
            'y': activity.score
        })
# Sort by date to ensure the graph is chronological
    recent_quiz_scores = sorted(recent_quiz_scores, key=lambda d: d['x'])
# END OF NEW CODE BLOCK    

    topic_performance = {}
    topic_history = {}
    last_seen_topic = {}

    for activity in graded_activities:
        if 'Quiz' in activity.type or 'Exam' in activity.type:
            for answer in activity.graded_results.get('graded_answers', []):
                topic = answer.get('topic', 'General')
                if topic not in topic_performance:
                    topic_performance[topic] = {'correct': 0, 'total': 0}
                    topic_history[topic] = []
                
                is_correct = 1 if answer.get('correct') or (answer.get('score', 0) >= 7) else 0
                topic_performance[topic]['correct'] += is_correct
                topic_performance[topic]['total'] += 1
                topic_history[topic].append(is_correct)
                last_seen_topic[topic] = activity.created_at

    topic_mastery = sorted(
        [{"topic": t, "score": int((d['correct'] / d['total']) * 100)} for t, d in topic_performance.items() if d['total'] > 0],
        key=lambda x: x['score'], reverse=True
    )
    
    potential_spotlights = []
    
    if topic_mastery:
        weakest = min(topic_mastery, key=lambda x: x['score'])
        if weakest['score'] < 70:
             potential_spotlights.append({'type': 'weakest', 'topic': weakest['topic'], 'score': weakest['score']})

    best_improvement = {'topic': None, 'change': 0}
    for topic, history in topic_history.items():
        if len(history) >= 4:
            midpoint = len(history) // 2
            first_half_avg = sum(history[:midpoint]) / midpoint
            second_half_avg = sum(history[midpoint:]) / (len(history) - midpoint)
            improvement = (second_half_avg - first_half_avg) * 100
            if improvement > best_improvement['change']:
                best_improvement = {'topic': topic, 'change': int(improvement)}
    
    if best_improvement['topic'] and best_improvement['change'] > 10:
        potential_spotlights.append({'type': 'improved', 'topic': best_improvement['topic'], 'change': best_improvement['change']})
        
    review_candidates = [t for t in topic_mastery if t['score'] < 90]
    if review_candidates:
        oldest_first = sorted(review_candidates, key=lambda x: last_seen_topic.get(x['topic']))
        potential_spotlights.append({'type': 'recommend', 'topic': oldest_first[0]['topic']})

    spotlight = random.choice(potential_spotlights) if potential_spotlights else None

    # --- REFACTORED: Use persistent progress data from the Hub ---
    total_xp = hub.total_xp
    streak_days = hub.streak_days

    # --- Yesterday/Today stats are still calculated dynamically ---
    today_xp = 0
    yesterday_activities = {'quizzes': 0, 'flashcards': 0}
    today = datetime.now(timezone.utc).date()
    yesterday = today - timedelta(days=1)
    
    # This loop is now only for calculating today/yesterday stats
    for activity in all_activities:
        if activity.status in ['graded', 'completed']:
            activity_date = activity.created_at.date()
            xp_gain = 15 if activity.status == 'graded' else 5
            
            if activity_date == today:
                today_xp += xp_gain
            elif activity_date == yesterday:
                if 'Quiz' in activity.type or 'Exam' in activity.type:
                    yesterday_activities['quizzes'] += 1
                elif activity.type == 'Flashcards':
                    yesterday_activities['flashcards'] += 1
    
    # --- REFACTORED: Level calculation now uses the persistent total_xp ---
    level_data = {}
    current_level = 1
    xp_for_next = 100
    xp_cumulative = 0
    while total_xp >= xp_cumulative + xp_for_next:
        xp_cumulative += xp_for_next
        current_level += 1
        xp_for_next = int(100 * (current_level ** 1.5))
    level_data = {"current_level": current_level, "xp_in_level": total_xp - xp_cumulative, "xp_for_next_level": xp_for_next}

    # --- UI Counts are still dynamic ---
    hub.notes_count = len(all_notes)
    hub.flashcard_count = len(all_flashcards)
    hub.quizzes_taken = len([q for q in all_quizzes_and_exams if q.status == 'graded'])
    
    # --- NEW: Fetch Notifications ---
    notifications_query = db.collection('notifications').order_by('created_at', direction=firestore.Query.DESCENDING).limit(10).stream()
    notifications = [Notification.from_dict(doc.to_dict()) for doc in notifications_query]
    unread_notifications_count = sum(1 for n in notifications if not n.read)

    # --- NEW: Check for Spotify connection ---
    spotify_connected_status = True if current_user.spotify_refresh_token else False

    # --- NEW: Check onboarding state ---
    needs_onboarding = not current_user.has_completed_onboarding
    print(f"Hub: User {current_user.id} has_completed_onboarding = {current_user.has_completed_onboarding}, needs_onboarding = {needs_onboarding}")

    # Convert all objects to dictionaries to ensure JSON serialization
    def obj_to_dict(obj):
        """Convert object to dictionary, handling Undefined values."""
        if hasattr(obj, 'to_dict'):
            result = clean_undefined_values(obj.to_dict())
            # For folders, preserve the hydrated_items attribute
            if hasattr(obj, 'hydrated_items'):
                result['hydrated_items'] = [obj_to_dict(item) for item in obj.hydrated_items]
            return result
        elif isinstance(obj, list):
            return [obj_to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            return clean_undefined_values(obj)
        else:
            return clean_undefined_values(obj)

    return render_template(
        "hub.html", 
        hub=hub,
        # Pass the new persistent data to the template
        total_xp=total_xp,
        streak_days=streak_days,
        level_data=level_data,
        # Pass all other data as before - convert to dicts for JSON serialization
        all_activities=[obj_to_dict(activity) for activity in all_activities],
        all_notes=[obj_to_dict(note) for note in all_notes], 
        all_sessions=[obj_to_dict(session) for session in all_sessions], 
        all_flashcards=[obj_to_dict(fc) for fc in all_flashcards], 
        all_quizzes_and_exams=[obj_to_dict(q) for q in all_quizzes_and_exams],
        topic_mastery=topic_mastery,
        spotlight=spotlight,
        today_xp=today_xp,
        all_folders=[obj_to_dict(folder) for folder in all_folders],
        all_slide_notes=[obj_to_dict(slide) for slide in all_slide_notes], # NEW: Pass slide notes to template
        yesterday_activities=yesterday_activities,
        notifications=notifications,
        unread_notifications_count=unread_notifications_count,
        recent_quiz_scores_json=json.dumps(recent_quiz_scores),
        spotify_connected=spotify_connected_status, # NEW: Pass this to the template
        needs_onboarding=needs_onboarding # NEW: Pass onboarding state to template
    )

# ==============================================================================
# 4. NEW SETTINGS & ACCOUNT MANAGEMENT ROUTES
# ==============================================================================
@app.route("/settings/update", methods=["POST"])
@login_required
def update_settings():
    """A general-purpose route to update various user settings."""
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "message": "No data provided."}), 400
    
    # Define which keys are allowed to be updated through this endpoint
    allowed_keys = [
        'profile_visible', 'activity_visible', 'default_note_privacy',
        'font_size_preference', 'high_contrast_mode', 'language',
        'background_preference',
        # Onboarding housekeeping fields
        'referral_source', 'goals', 'email_opt_in', 'theme_preference'
    ]
    
    update_data = {}
    for key, value in data.items():
        if key in allowed_keys:
            update_data[key] = value

    if not update_data:
        return jsonify({"success": False, "message": "No valid settings provided."}), 400

    try:
        db.collection('users').document(current_user.id).update(update_data)
        return jsonify({"success": True})
    except Exception as e:
        print(f"Error updating user settings for {current_user.id}: {e}")
        return jsonify({"success": False, "message": "An internal error occurred."}), 500

@app.route("/settings/update_password", methods=["POST"])
@login_required
def update_password():
    data = request.get_json()
    new_password = data.get('new_password')
    
    if not new_password or len(new_password) < 6:
        return jsonify({"success": False, "message": "Password must be at least 6 characters long."}), 400
        
    try:
        new_password_hash = generate_password_hash(new_password, method='pbkdf2:sha256')
        db.collection('users').document(current_user.id).update({'password_hash': new_password_hash})
        return jsonify({"success": True, "message": "Password updated successfully!"})
    except Exception as e:
        print(f"Error updating password for {current_user.id}: {e}")
        return jsonify({"success": False, "message": "An error occurred."}), 500

# --- Favourites Management Routes ---
@app.route("/api/favourites", methods=["GET"])
@login_required
def get_favourites():
    """Get user's favourite tools."""
    try:
        user_doc = db.collection('users').document(current_user.id).get()
        if user_doc.exists:
            user_data = user_doc.to_dict()
            favourites = user_data.get('favourite_tools', [])
            return jsonify({"success": True, "favourites": favourites})
        return jsonify({"success": False, "message": "User not found"}), 404
    except Exception as e:
        print(f"Error getting favourites for {current_user.id}: {e}")
        return jsonify({"success": False, "message": "An error occurred"}), 500

@app.route("/api/favourites/add", methods=["POST"])
@login_required
def add_favourite():
    """Add a tool to user's favourites."""
    try:
        data = request.get_json()
        tool_id = data.get('tool_id')
        tool_name = data.get('tool_name')
        tool_url = data.get('tool_url')
        
        if not tool_id or not tool_name:
            return jsonify({"success": False, "message": "Tool ID and name are required"}), 400
        
        # Get current favourites
        user_ref = db.collection('users').document(current_user.id)
        user_doc = user_ref.get()
        
        if not user_doc.exists:
            return jsonify({"success": False, "message": "User not found"}), 404
        
        current_favourites = user_doc.to_dict().get('favourite_tools', [])
        
        # Check if already exists
        if any(fav['tool_id'] == tool_id for fav in current_favourites):
            return jsonify({"success": False, "message": "Tool already in favourites"}), 400
        
        # Check if limit reached
        if len(current_favourites) >= 5:
            return jsonify({"success": False, "message": "Maximum 5 favourite tools allowed"}), 400
        
        # Add new favourite
        new_favourite = {
            'tool_id': tool_id,
            'tool_name': tool_name,
            'tool_url': tool_url or f"#{tool_id}",
            'added_at': datetime.now(timezone.utc).isoformat()
        }
        
        current_favourites.append(new_favourite)
        
        # Update in database
        user_ref.update({'favourite_tools': current_favourites})
        
        return jsonify({"success": True, "favourites": current_favourites})
        
    except Exception as e:
        print(f"Error adding favourite for {current_user.id}: {e}")
        return jsonify({"success": False, "message": "An error occurred"}), 500

@app.route("/api/favourites/remove", methods=["POST"])
@login_required
def remove_favourite():
    """Remove a tool from user's favourites."""
    try:
        data = request.get_json()
        tool_id = data.get('tool_id')
        
        if not tool_id:
            return jsonify({"success": False, "message": "Tool ID is required"}), 400
        
        # Get current favourites
        user_ref = db.collection('users').document(current_user.id)
        user_doc = user_ref.get()
        
        if not user_doc.exists:
            return jsonify({"success": False, "message": "User not found"}), 404
        
        current_favourites = user_doc.to_dict().get('favourite_tools', [])
        
        # Remove the tool
        updated_favourites = [fav for fav in current_favourites if fav['tool_id'] != tool_id]
        
        # Update in database
        user_ref.update({'favourite_tools': updated_favourites})
        
        return jsonify({"success": True, "favourites": updated_favourites})
        
    except Exception as e:
        print(f"Error removing favourite for {current_user.id}: {e}")
        return jsonify({"success": False, "message": "An error occurred"}), 500

@app.route("/api/favourites/reorder", methods=["POST"])
@login_required
def reorder_favourites():
    """Reorder user's favourite tools."""
    try:
        data = request.get_json()
        new_order = data.get('favourites', [])
        
        if len(new_order) > 5:
            return jsonify({"success": False, "message": "Maximum 5 favourite tools allowed"}), 400
        
        # Update in database
        user_ref = db.collection('users').document(current_user.id)
        user_ref.update({'favourite_tools': new_order})
        
        return jsonify({"success": True, "favourites": new_order})
        
    except Exception as e:
        print(f"Error reordering favourites for {current_user.id}: {e}")
        return jsonify({"success": False, "message": "An error occurred"}), 500

@app.route("/account/request_data")
@login_required
def request_data():
    """Gathers all user data and returns it as a downloadable JSON file."""
    user_data = {}
    try:
        # 1. User Info
        user_doc = db.collection('users').document(current_user.id).get()
        if user_doc.exists:
            user_data['profile'] = user_doc.to_dict()
            # Remove sensitive info
            user_data['profile'].pop('password_hash', None)
            user_data['profile'].pop('stripe_customer_id', None)
            user_data['profile'].pop('stripe_subscription_id', None)
            
        # 2. Hubs and their contents
        user_data['hubs'] = []
        hubs_ref = db.collection('hubs').where('user_id', '==', current_user.id).stream()
        for hub_doc in hubs_ref:
            hub_data = hub_doc.to_dict()
            hub_id = hub_doc.id
            hub_data['notes'] = [doc.to_dict() for doc in db.collection('notes').where('hub_id', '==', hub_id).stream()]
            hub_data['activities'] = [doc.to_dict() for doc in db.collection('activities').where('hub_id', '==', hub_id).stream()]
            hub_data['folders'] = [doc.to_dict() for doc in db.collection('folders').where('hub_id', '==', hub_id).stream()]
            hub_data['sessions'] = [doc.to_dict() for doc in db.collection('sessions').where('hub_id', '==', hub_id).stream()]
            user_data['hubs'].append(hub_data)

        # Convert datetime objects to strings for JSON serialization
        def default_serializer(o):
            if isinstance(o, (datetime)):
                return o.isoformat()
            raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

        json_data = json.dumps(user_data, indent=4, default=default_serializer)
        
        return Response(
            json_data,
            mimetype="application/json",
            headers={"Content-Disposition": f"attachment;filename=my_study_hub_data_{current_user.id}.json"}
        )
    except Exception as e:
        print(f"Error exporting data for user {current_user.id}: {e}")
        flash("Could not export your data at this time.", "error")
        return redirect(url_for('dashboard', _anchor='settings'))


@app.route("/account/delete", methods=["POST"])
@login_required
def delete_account():
    """Permanently deletes a user and all their associated data."""
    try:
        user_id = current_user.id
        
        # 1. Get all hubs owned by the user
        hubs_query = db.collection('hubs').where('user_id', '==', user_id).stream()
        hub_ids = [hub.id for hub in hubs_query]

        # 2. Delete all content within each hub
        for hub_id in hub_ids:
            # Delete files from Cloud Storage
            blobs = bucket.list_blobs(prefix=f"hubs/{hub_id}/")
            for blob in blobs:
                blob.delete()
            # Delete Firestore documents
            collections_to_delete = ['notes', 'activities', 'folders', 'sessions', 'annotated_slide_decks', 'notifications']
            for coll in collections_to_delete:
                docs = db.collection(coll).where('hub_id', '==', hub_id).stream()
                for doc in docs:
                    doc.reference.delete()
            # Delete the hub document itself
            db.collection('hubs').document(hub_id).delete()
            
        # 3. Delete user's avatar from storage
        avatar_blobs = bucket.list_blobs(prefix=f"avatars/{user_id}/")
        for blob in avatar_blobs:
            blob.delete()
            
        # 4. Delete shared folders owned by the user
        shared_folders_query = db.collection('shared_folders').where('owner_id', '==', user_id).stream()
        for doc in shared_folders_query:
            doc.reference.delete()

        # 5. Delete the user document itself
        db.collection('users').document(user_id).delete()

        # 6. Log the user out
        logout_user()
        
        return jsonify({"success": True, "message": "Your account has been permanently deleted."})

    except Exception as e:
        print(f"Error deleting account for user {user_id}: {e}")
        return jsonify({"success": False, "message": "An error occurred during account deletion."}), 500

@app.route("/admin/delete-user/<user_id>", methods=["POST"])
@login_required
def admin_delete_user(user_id):
    """Admin function to delete any user by ID."""
    # Add admin check - you can modify this based on your admin logic
    if not current_user.email.endswith('@yourdomain.com'):  # Example admin check
        return jsonify({"success": False, "message": "Unauthorized - Admin access required"}), 403
    
    try:
        # Get the user to delete
        user_doc = db.collection('users').document(user_id).get()
        if not user_doc.exists:
            return jsonify({"success": False, "message": "User not found"}), 404
        
        user_data = user_doc.to_dict()
        
        # Use the same deletion logic as delete_account()
        # 1. Get all hubs owned by the user
        hubs_query = db.collection('hubs').where('user_id', '==', user_id).stream()
        hub_ids = [hub.id for hub in hubs_query]

        # 2. Delete all content within each hub
        for hub_id in hub_ids:
            # Delete files from Cloud Storage
            blobs = bucket.list_blobs(prefix=f"hubs/{hub_id}/")
            for blob in blobs:
                blob.delete()
            # Delete Firestore documents
            collections_to_delete = ['notes', 'activities', 'folders', 'sessions', 'annotated_slide_decks', 'notifications']
            for coll in collections_to_delete:
                docs = db.collection(coll).where('hub_id', '==', hub_id).stream()
                for doc in docs:
                    doc.reference.delete()
            # Delete the hub document itself
            db.collection('hubs').document(hub_id).delete()
            
        # 3. Delete user's avatar from storage
        avatar_blobs = bucket.list_blobs(prefix=f"avatars/{user_id}/")
        for blob in avatar_blobs:
            blob.delete()
            
        # 4. Delete shared folders owned by the user
        shared_folders_query = db.collection('shared_folders').where('owner_id', '==', user_id).stream()
        for doc in shared_folders_query:
            doc.reference.delete()

        # 5. Delete the user document itself
        db.collection('users').document(user_id).delete()
        
        return jsonify({"success": True, "message": f"User {user_data.get('email', user_id)} deleted successfully"})

    except Exception as e:
        print(f"Error deleting user {user_id}: {e}")
        return jsonify({"success": False, "message": "An error occurred during user deletion."}), 500

@app.route("/help-center")
def help_center():
    # In a real app, you would render a full help center template.
    # For now, this is a placeholder.
    flash("The Help Center is not yet implemented.", "info")
    return redirect(url_for('dashboard'))

# ==============================================================================
# 5. SLIDE NOTE-TAKING ROUTES
# ==============================================================================

@app.route("/hub/<hub_id>/create_slide_notes_session", methods=["POST"])
@login_required
def create_slide_notes_session(hub_id):
    title = request.form.get('title', 'Untitled Lecture Notes')
    selected_file_id = request.form.get('selected_file_id')
    selected_file_name = request.form.get('selected_file_name')
    
    # Check if a file was uploaded or selected from existing files
    file_uploaded = 'slide_file' in request.files and request.files['slide_file'].filename != ''
    file_selected = (selected_file_id is not None and selected_file_id != '') or (selected_file_name is not None and selected_file_name != '')
    
    if not file_uploaded and not file_selected:
        flash("Please select a file from existing files or upload a new file.", "error")
        return redirect(url_for('hub_page', hub_id=hub_id))

    try:
        if file_selected:
            # Use existing file from hub
            file_doc = None
            file_data = None
            
            if selected_file_id and selected_file_id != 'unknown':
                # Try to find by file ID first
                print(f"Using existing file with ID: {selected_file_id}")
                file_doc = db.collection('files').document(selected_file_id).get()
                if file_doc.exists:
                    file_data = file_doc.to_dict()
            
            if not file_data and selected_file_name:
                # Try to find by file name
                print(f"Looking for file with name: {selected_file_name}")
                files_query = db.collection('files').where('hub_id', '==', hub_id).where('filename', '==', selected_file_name).limit(1)
                files = list(files_query.stream())
                if not files:
                    # Try original_filename
                    files_query = db.collection('files').where('hub_id', '==', hub_id).where('original_filename', '==', selected_file_name).limit(1)
                    files = list(files_query.stream())
                
                if files:
                    file_data = files[0].to_dict()
                    print(f"Found file by name: {selected_file_name}")
                else:
                    print(f"No file found with name: {selected_file_name}")
            
            if not file_data:
                print(f"File not found - ID: {selected_file_id}, Name: {selected_file_name}")
                flash("Selected file not found.", "error")
                return redirect(url_for('hub_page', hub_id=hub_id))
            
            file_path = file_data.get('file_path') or file_data.get('path')
            filename = file_data.get('filename') or file_data.get('original_filename') or file_data.get('name')
            print(f"File data: {file_data}")
            print(f"Extracted: path={file_path}, name={filename}")
            
            # Get the file from Firebase Storage
            blob = bucket.blob(file_path)
            if not blob.exists():
                print(f"Blob does not exist at path: {file_path}")
                flash("Selected file not found in storage.", "error")
                return redirect(url_for('hub_page', hub_id=hub_id))
            
            # Make it publicly accessible (ignore errors if already public)
            try:
                blob.make_public()
                pdf_url = blob.public_url
                print(f"Made blob public, URL: {pdf_url}")
            except Exception as e:
                print(f"Error making blob public: {e}")
                # Try to get the URL anyway
                try:
                    pdf_url = blob.public_url
                    print(f"Using existing public URL: {pdf_url}")
                except Exception as e2:
                    print(f"Error getting public URL: {e2}")
                    flash("Error accessing selected file.", "error")
                    return redirect(url_for('hub_page', hub_id=hub_id))
            
        else:
            # Upload new file
            file = request.files['slide_file']
            
            # Only allow PDFs for now
            if not file.filename.lower().endswith('.pdf'):
                flash("Please select a valid PDF file.", "error")
                return redirect(url_for('hub_page', hub_id=hub_id))
            
            filename = secure_filename(file.filename)
            file_path = f"hubs/{hub_id}/slides/{uuid.uuid4()}_{filename}"
            blob = bucket.blob(file_path)

            # Upload file to Firebase
            blob.upload_from_file(file, content_type=file.content_type)

            # Make it publicly accessible
            blob.make_public()
            pdf_url = blob.public_url

        # IMPORTANT: create the document in annotated_slide_decks (not sessions)
        session_ref = db.collection('annotated_slide_decks').document()
        new_session = AnnotatedSlideDeck(
            id=session_ref.id,
            hub_id=hub_id,
            user_id=current_user.id,                # <-- ensure save endpoint's authorization check will pass
            title=title,
            source_file_path=file_path,             # slide_notes_workspace reads session.source_file_path
            slides_data=[],                          # start empty
        )
        session_ref.set(new_session.to_dict())
        print(f"Created session: {new_session.id}")

        # Render the workspace with the working PDF URL and the new annotated_slide_deck
        print(f"Rendering slide_notes_workspace.html with pdf_url: {pdf_url}")
        return render_template(
            "slide_notes_workspace.html",
            session=new_session,
            pdf_url=pdf_url,
            slides_data=[]
        )

    except Exception as e:
        print(f"Error creating slide notes session: {e}")
        flash("Failed to create slide notes session.", "error")
        return redirect(url_for('hub_page', hub_id=hub_id))



@app.route("/slide_notes/<session_id>")
@login_required
def slide_notes_workspace(session_id):
    session_doc = db.collection("annotated_slide_decks").document(session_id).get()
    if not session_doc.exists:
        flash("Lecture not found", "error")
        return redirect(url_for("dashboard"))

    session = AnnotatedSlideDeck.from_dict(session_doc.to_dict())

    # SECURITY CHECK
    if session.user_id != current_user.id:
        flash("You do not have permission to view this.", "error")
        return redirect(url_for('dashboard'))

    try:
        # Convert storage path to a usable URL
        blob = bucket.blob(session.source_file_path)
        if not blob.public_url:
            blob.make_public()
        pdf_url = blob.public_url
    except Exception as e:
        print(f"Error making PDF public: {e}")
        flash("Could not load lecture file", "error")
        return redirect(url_for("dashboard"))

    return render_template(
        "slide_notes_workspace.html",
        session=session,
        pdf_url=pdf_url,
        slides_data=session.slides_data if session.slides_data else []
    )


@app.route("/slide_notes/<session_id>/save_data", methods=["POST"])
@login_required
def save_slide_notes_data(session_id):
    session_ref = db.collection('annotated_slide_decks').document(session_id)
    session_doc = session_ref.get()
    if not session_doc.exists or session_doc.to_dict().get('user_id') != current_user.id:
        return jsonify({"success": False, "message": "Unauthorized"}), 403

    data = request.get_json()
    slides_data = data.get('slides_data')
    if slides_data is None:
        return jsonify({"success": False, "message": "No data provided"}), 400
    
    try:
        session_ref.update({'slides_data': slides_data})
        return jsonify({"success": True, "message": "Saved successfully"})
    except Exception as e:
        print(f"Error saving slide data for session {session_id}: {e}")
        return jsonify({"success": False, "message": "Server error"}), 500

@app.route("/slide_notes/ai_assist", methods=["POST"])
@login_required
def slide_notes_ai_assist():
    data = request.get_json()
    action = data.get('action')
    slide_text = data.get('slide_text')
    hub_id = data.get('hub_id')
    slide_num = data.get('slide_number', 'current')
    
    if not action or not slide_text or not hub_id:
        return jsonify({"success": False, "message": "Missing action, hub_id, or slide text"}), 400
    
    try:
        if action == 'summarize':
            result_text = generate_summary_from_slide_text(slide_text)
            return jsonify({"success": True, "html_content": markdown.markdown(result_text)})
        
        elif action == 'flashcards':
            raw_flashcards = generate_flashcards_from_slide_text(slide_text)
            parsed_cards = parse_flashcards(raw_flashcards)
            if not parsed_cards:
                return jsonify({"success": False, "message": "Could not generate valid flashcards from this slide."}), 500

            fc_ref = db.collection('activities').document()
            new_fc = Activity(id=fc_ref.id, hub_id=hub_id, type='Flashcards', title=f"Flashcards from Slide {slide_num}", data={'cards': parsed_cards}, status='completed')
            fc_ref.set(new_fc.to_dict())

            return jsonify({"success": True, "message": f"Flashcard set created! You can find it in 'My Flashcards'.", "redirect_url": url_for('edit_flashcard_set', activity_id=fc_ref.id)})
        
        else:
            return jsonify({"success": False, "message": "Invalid action"}), 400

    except Exception as e:
        print(f"Error in AI assist for slides: {e}")
        return jsonify({"success": False, "message": "AI assistant failed to generate a response."}), 500


@app.route("/slide_notes/<session_id>/generate_full_lecture_flashcards", methods=["POST"])
@login_required
def generate_full_lecture_flashcards(session_id):
    """Generate flashcards for the entire lecture in the background."""
    try:
        # Get the session and verify ownership
        session_doc = db.collection('annotated_slide_decks').document(session_id).get()
        if not session_doc.exists:
            return jsonify({"success": False, "message": "Session not found"}), 404
        
        session_data = session_doc.to_dict()
        if session_data.get('user_id') != current_user.id:
            return jsonify({"success": False, "message": "Unauthorized"}), 403
        
        # Check if flashcards are already being generated or completed
        if session_data.get('flashcards_status') in ['generating', 'completed']:
            return jsonify({"success": False, "message": "Flashcards are already being generated or completed"}), 400
        
        # Get flashcard count from request
        data = request.get_json() or {}
        flashcard_count = data.get('flashcard_count', 10)  # Default to 10
        
        # Validate flashcard count
        if not isinstance(flashcard_count, int) or flashcard_count < 1 or flashcard_count > 50:
            return jsonify({"success": False, "message": "Flashcard count must be between 1 and 50"}), 400
        
        # Update status to generating
        session_ref = db.collection('annotated_slide_decks').document(session_id)
        session_ref.update({'flashcards_status': 'generating'})
        
        # Start background task with flashcard count
        import threading
        thread = threading.Thread(target=generate_lecture_flashcards_background, args=(session_id, flashcard_count))
        thread.daemon = True
        thread.start()
        
        return jsonify({"success": True, "message": f"Flashcard generation started for {flashcard_count} flashcards"})
        
    except Exception as e:
        print(f"Error starting flashcard generation: {e}")
        return jsonify({"success": False, "message": "Failed to start flashcard generation"}), 500


def migrate_flashcards_to_spaced_repetition(activity_id, flashcards_data):
    """Migrate flashcards from activities to spaced repetition system"""
    try:
        print(f"Migrating {len(flashcards_data)} flashcards to spaced repetition system...")
        
        # Check if already migrated
        existing_cards = db.collection('spaced_repetition_cards').where('activity_id', '==', activity_id).stream()
        if list(existing_cards):
            print(f"Flashcards for activity {activity_id} already migrated")
            return
        
        migrated_count = 0
        valid_cards = []
        
        # First pass: validate and collect valid cards
        for i, card_data in enumerate(flashcards_data):
            try:
                # Validate card data
                front = card_data.get('front', '').strip()
                back = card_data.get('back', '').strip()
                
                if not front or not back:
                    print(f"Skipping invalid card {i}: empty front or back")
                    continue
                
                valid_cards.append({
                    'index': i,
                    'front': front,
                    'back': back,
                    'data': card_data
                })
                
            except Exception as card_error:
                print(f"Error validating card {i}: {card_error}")
                continue
        
        # Second pass: create spaced repetition cards with correct indices
        for valid_card in valid_cards:
            try:
                # Create spaced repetition card
                sr_card_ref = db.collection('spaced_repetition_cards').document()
                sr_card = SpacedRepetitionCard(
                    id=sr_card_ref.id,
                    activity_id=activity_id,
                    front=valid_card['front'],
                    back=valid_card['back'],
                    card_index=valid_card['index'],
                    difficulty='medium'
                )
                sr_card_ref.set(sr_card.to_dict())
                migrated_count += 1
                
            except Exception as card_error:
                print(f"Error migrating card {valid_card['index']}: {card_error}")
                continue
        
        print(f"Successfully migrated {migrated_count}/{len(flashcards_data)} flashcards to spaced repetition system")
        
    except Exception as e:
        print(f"Error migrating flashcards to spaced repetition: {e}")
        import traceback
        traceback.print_exc()

def find_matching_flashcard(sr_card, flashcards):
    """Find a flashcard that matches the spaced repetition card by content"""
    sr_front = sr_card.front.strip()
    sr_back = sr_card.back.strip()
    
    for idx, fc in enumerate(flashcards):
        fc_front = fc.get('front', '').strip()
        fc_back = fc.get('back', '').strip()
        
        if fc_front == sr_front and fc_back == sr_back:
            return idx, fc
    
    return None, None

def repair_spaced_repetition_indices(activity_id):
    """Repair card indices for a specific activity by matching content"""
    try:
        print(f"🔧 REPAIR: Starting index repair for activity {activity_id}")
        
        # Get the activity's flashcards
        activity_doc = db.collection('activities').document(activity_id).get()
        if not activity_doc.exists:
            print(f"❌ ERROR: Activity {activity_id} not found")
            return False
        
        activity_data = activity_doc.to_dict()
        flashcards = activity_data.get('data', {}).get('cards', [])
        print(f"🔧 REPAIR: Found {len(flashcards)} flashcards in activity")
        
        # Get spaced repetition cards for this activity
        sr_cards_query = db.collection('spaced_repetition_cards').where('activity_id', '==', activity_id)
        sr_cards = list(sr_cards_query.stream())
        print(f"🔧 REPAIR: Found {len(sr_cards)} spaced repetition cards")
        
        repaired_count = 0
        for sr_card_doc in sr_cards:
            sr_card_data = sr_card_doc.to_dict()
            sr_card = SpacedRepetitionCard.from_dict(sr_card_data)
            
            # Try to find matching flashcard
            match_index, match_flashcard = find_matching_flashcard(sr_card, flashcards)
            
            if match_index is not None:
                # Update the card index if it's different
                if sr_card.card_index != match_index:
                    print(f"🔧 REPAIR: Updating card {sr_card.id} index from {sr_card.card_index} to {match_index}")
                    sr_card_doc.reference.update({'card_index': match_index})
                    repaired_count += 1
                else:
                    print(f"✅ REPAIR: Card {sr_card.id} index {match_index} is correct")
            else:
                print(f"❌ REPAIR: No matching flashcard found for SR card {sr_card.id}")
        
        print(f"🔧 REPAIR: Completed repair for activity {activity_id}, updated {repaired_count} cards")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: Failed to repair indices for activity {activity_id}: {e}")
        return False

def recover_missing_flashcards(activity_id):
    """Recover missing flashcards from spaced repetition data"""
    try:
        print(f"🔄 RECOVERY: Starting flashcard recovery for activity {activity_id}")
        
        # Get the activity
        activity_doc = db.collection('activities').document(activity_id).get()
        if not activity_doc.exists:
            print(f"❌ ERROR: Activity {activity_id} not found")
            return False
        
        activity_data = activity_doc.to_dict()
        current_cards = activity_data.get('data', {}).get('cards', [])
        
        # Get spaced repetition cards for this activity
        sr_cards_query = db.collection('spaced_repetition_cards').where('activity_id', '==', activity_id)
        sr_cards = list(sr_cards_query.stream())
        
        if not sr_cards:
            print(f"❌ ERROR: No spaced repetition cards found for activity {activity_id}")
            return False
        
        print(f"🔄 RECOVERY: Found {len(sr_cards)} spaced repetition cards")
        
        # If activity has no cards but SR cards exist, recover from SR data
        if len(current_cards) == 0 and len(sr_cards) > 0:
            print(f"🔄 RECOVERY: Activity has no cards, recovering from SR data")
            
            # Create cards array from SR data, sorted by card_index
            recovered_cards = []
            sr_cards_data = []
            
            for sr_card_doc in sr_cards:
                sr_card_data = sr_card_doc.to_dict()
                sr_cards_data.append(sr_card_data)
            
            # Sort by card_index to maintain order
            sr_cards_data.sort(key=lambda x: x.get('card_index', 0))
            
            for sr_card_data in sr_cards_data:
                card = {
                    'front': sr_card_data.get('front', ''),
                    'back': sr_card_data.get('back', '')
                }
                recovered_cards.append(card)
            
            # Update the activity with recovered cards
            activity_doc.reference.update({
                'data': {
                    **activity_data.get('data', {}),
                    'cards': recovered_cards
                }
            })
            
            print(f"✅ RECOVERY: Recovered {len(recovered_cards)} cards for activity {activity_id}")
            return True
        
        # If activity has some cards, try to merge with SR data
        elif len(current_cards) > 0 and len(sr_cards) > 0:
            print(f"🔄 RECOVERY: Activity has {len(current_cards)} cards, checking for missing ones")
            
            # Find SR cards that don't have matching activity cards
            missing_cards = []
            for sr_card_doc in sr_cards:
                sr_card_data = sr_card_doc.to_dict()
                sr_front = sr_card_data.get('front', '').strip()
                sr_back = sr_card_data.get('back', '').strip()
                
                # Check if this card exists in activity
                found = False
                for activity_card in current_cards:
                    if (activity_card.get('front', '').strip() == sr_front and 
                        activity_card.get('back', '').strip() == sr_back):
                        found = True
                        break
                
                if not found:
                    missing_cards.append({
                        'front': sr_front,
                        'back': sr_back
                    })
            
            if missing_cards:
                # Add missing cards to activity
                updated_cards = current_cards + missing_cards
                activity_doc.reference.update({
                    'data': {
                        **activity_data.get('data', {}),
                        'cards': updated_cards
                    }
                })
                print(f"✅ RECOVERY: Added {len(missing_cards)} missing cards to activity {activity_id}")
                return True
            else:
                print(f"✅ RECOVERY: No missing cards found for activity {activity_id}")
                return True
        
        else:
            print(f"ℹ️ RECOVERY: Activity {activity_id} already has cards, no recovery needed")
            return True
        
    except Exception as e:
        print(f"❌ ERROR: Failed to recover flashcards for activity {activity_id}: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_lecture_flashcards_background(session_id, flashcard_count=10):
    """Background function to generate flashcards for the entire lecture."""
    try:
        # Get the session
        session_doc = db.collection('annotated_slide_decks').document(session_id).get()
        if not session_doc.exists:
            return
        
        session_data = session_doc.to_dict()
        session_ref = db.collection('annotated_slide_decks').document(session_id)
        
        # Get the PDF file
        source_file_path = session_data.get('source_file_path')
        if not source_file_path:
            session_ref.update({'flashcards_status': 'failed'})
            return
        
        # Download PDF from Cloud Storage
        blob = bucket.blob(source_file_path)
        pdf_content = blob.download_as_bytes()
        
        # Extract text from all pages
        import PyPDF2
        import io
        
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
        all_text = ""
        
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            all_text += f"\n--- Slide {page_num + 1} ---\n{page_text}\n"
        
        if len(all_text.strip()) < 50:
            session_ref.update({'flashcards_status': 'failed'})
            return
        
        # Generate flashcards from all text with specified count
        raw_flashcards = generate_flashcards_from_slide_text(all_text, count=flashcard_count)
        parsed_cards = parse_flashcards(raw_flashcards)
        
        if not parsed_cards:
            session_ref.update({'flashcards_status': 'failed'})
            return
        
        # Limit to requested count if more were generated
        if len(parsed_cards) > flashcard_count:
            parsed_cards = parsed_cards[:flashcard_count]
        
        # Save flashcards to the session
        session_ref.update({
            'flashcards_data': parsed_cards,
            'flashcards_status': 'completed'
        })
        
        print(f"Successfully generated {len(parsed_cards)} flashcards for session {session_id}")
        
    except Exception as e:
        print(f"Error generating flashcards in background: {e}")
        # Update status to failed
        try:
            session_ref = db.collection('annotated_slide_decks').document(session_id)
            session_ref.update({'flashcards_status': 'failed'})
        except:
            pass


@app.route("/slide_notes/<session_id>/flashcards_status", methods=["GET"])
@login_required
def get_flashcards_status(session_id):
    """Get the current status of flashcard generation."""
    try:
        session_doc = db.collection('annotated_slide_decks').document(session_id).get()
        if not session_doc.exists:
            return jsonify({"success": False, "message": "Session not found"}), 404
        
        session_data = session_doc.to_dict()
        if session_data.get('user_id') != current_user.id:
            return jsonify({"success": False, "message": "Unauthorized"}), 403
        
        return jsonify({
            "success": True,
            "status": session_data.get('flashcards_status', 'none'),
            "flashcards_count": len(session_data.get('flashcards_data', []))
        })
        
    except Exception as e:
        print(f"Error getting flashcard status: {e}")
        return jsonify({"success": False, "message": "Failed to get status"}), 500


@app.route("/slide_notes/<session_id>/generate_full_lecture_quiz", methods=["POST"])
@login_required
def generate_full_lecture_quiz(session_id):
    """Generate quiz for the entire lecture in the background."""
    try:
        # Get the session and verify ownership
        session_doc = db.collection('annotated_slide_decks').document(session_id).get()
        if not session_doc.exists:
            return jsonify({"success": False, "message": "Session not found"}), 404
        
        session_data = session_doc.to_dict()
        if session_data.get('user_id') != current_user.id:
            return jsonify({"success": False, "message": "Unauthorized"}), 403
        
        # Check if quiz is already being generated or completed
        if session_data.get('quiz_status') in ['generating', 'completed']:
            return jsonify({"success": False, "message": "Quiz is already being generated or completed"}), 400
        
        # Get quiz count from request
        data = request.get_json() or {}
        quiz_count = data.get('quiz_count', 10)  # Default to 10
        
        # Validate quiz count
        if not isinstance(quiz_count, int) or quiz_count < 5 or quiz_count > 50:
            return jsonify({"success": False, "message": "Quiz count must be between 5 and 50"}), 400
        
        # Update status to generating
        session_ref = db.collection('annotated_slide_decks').document(session_id)
        session_ref.update({'quiz_status': 'generating'})
        
        # Start background task with quiz count
        import threading
        thread = threading.Thread(target=generate_lecture_quiz_background, args=(session_id, quiz_count))
        thread.daemon = True
        thread.start()
        
        return jsonify({"success": True, "message": f"Quiz generation started for {quiz_count} questions"})
        
    except Exception as e:
        print(f"Error starting quiz generation: {e}")
        return jsonify({"success": False, "message": "Failed to start quiz generation"}), 500


def generate_lecture_quiz_background(session_id, quiz_count=10):
    """Background function to generate quiz for the entire lecture."""
    try:
        # Get the session
        session_doc = db.collection('annotated_slide_decks').document(session_id).get()
        if not session_doc.exists:
            return
        
        session_data = session_doc.to_dict()
        session_ref = db.collection('annotated_slide_decks').document(session_id)
        
        # Get the PDF file
        source_file_path = session_data.get('source_file_path')
        if not source_file_path:
            session_ref.update({'quiz_status': 'failed'})
            return
        
        # Download PDF from Cloud Storage
        blob = bucket.blob(source_file_path)
        pdf_content = blob.download_as_bytes()
        
        # Extract text from all pages
        import PyPDF2
        import io
        
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
        all_text = ""
        
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            if text.strip():
                all_text += f"Page {page_num + 1}:\n{text}\n\n"
        
        if not all_text.strip():
            session_ref.update({'quiz_status': 'failed'})
            return
        
        # Generate quiz using AI
        quiz_json = generate_quiz_from_text(all_text, quiz_count)
        quiz_data = safe_load_json(quiz_json)
        
        if not quiz_data.get('questions'):
            session_ref.update({'quiz_status': 'failed'})
            return
        
        # Create a new activity for the quiz
        activity_ref = db.collection('activities').document()
        quiz_title = f"Quiz for {session_data.get('title', 'Lecture')}"
        new_quiz = Activity(
            id=activity_ref.id,
            hub_id=session_data.get('hub_id'),
            type='Quiz',
            title=quiz_title,
            data={'questions': quiz_data['questions']}
        )
        
        # Store the quiz activity
        activity_ref.set(new_quiz.to_dict())
        
        # Update session with quiz info
        session_ref.update({
            'quiz_id': activity_ref.id,
            'quiz_status': 'completed',
            'question_count': len(quiz_data['questions'])
        })
        
    except Exception as e:
        print(f"Error generating quiz in background: {e}")
        try:
            session_ref = db.collection('annotated_slide_decks').document(session_id)
            session_ref.update({'quiz_status': 'failed'})
        except:
            pass


@app.route("/slide_notes/<session_id>/quiz_status", methods=["GET"])
@login_required
def get_quiz_status(session_id):
    """Get the current status of quiz generation."""
    try:
        session_doc = db.collection('annotated_slide_decks').document(session_id).get()
        if not session_doc.exists:
            return jsonify({"success": False, "message": "Session not found"}), 404
        
        session_data = session_doc.to_dict()
        if session_data.get('user_id') != current_user.id:
            return jsonify({"success": False, "message": "Unauthorized"}), 403
        
        return jsonify({
            "success": True,
            "status": session_data.get('quiz_status', 'none'),
            "quiz_id": session_data.get('quiz_id'),
            "question_count": session_data.get('question_count', 0)
        })
        
    except Exception as e:
        print(f"Error getting quiz status: {e}")
        return jsonify({"success": False, "message": "Failed to get status"}), 500


@app.route("/slide_notes/<session_id>/view_quiz")
@login_required
def view_lecture_quiz(session_id):
    """View quiz for a specific lecture session."""
    try:
        session_doc = db.collection('annotated_slide_decks').document(session_id).get()
        if not session_doc.exists:
            return "Session not found", 404
        
        session_data = session_doc.to_dict()
        if session_data.get('user_id') != current_user.id:
            return "Unauthorized", 403
        
        quiz_id = session_data.get('quiz_id')
        if not quiz_id:
            return "No quiz found for this session", 404
        
        # Get the quiz activity
        quiz_doc = db.collection('activities').document(quiz_id).get()
        if not quiz_doc.exists:
            return "Quiz not found", 404
        
        quiz_data = quiz_doc.to_dict()
        quiz_activity = Activity.from_dict(quiz_data)
        
        return redirect(url_for('take_lecture_quiz', activity_id=quiz_id))
        
    except Exception as e:
        print(f"Error viewing quiz: {e}")
        return "Error loading quiz", 500


@app.route("/slide_notes/<session_id>/ask_ai_tutor", methods=["POST"])
@login_required
def ask_ai_tutor(session_id):
    """AI Tutor endpoint for answering questions about the lecture."""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        current_slide = data.get('current_slide', 1)
        
        if not question:
            return jsonify({"success": False, "message": "No question provided"}), 400
        
        # Get the session and verify ownership
        session_doc = db.collection('annotated_slide_decks').document(session_id).get()
        if not session_doc.exists:
            return jsonify({"success": False, "message": "Session not found"}), 404
        
        session_data = session_doc.to_dict()
        if session_data.get('user_id') != current_user.id:
            return jsonify({"success": False, "message": "Unauthorized"}), 403
        
        # Get the processed lecture content
        lecture_content = session_data.get('processed_content')
        if not lecture_content:
            # If not processed yet, process it now
            print(f"Processing lecture content for existing session {session_id}")
            lecture_content = process_lecture_for_ai_tutor(session_data)
            if not lecture_content:
                print(f"Failed to process lecture content for session {session_id}")
                return jsonify({"success": False, "message": "Unable to process lecture content"}), 500
            
            # Save processed content to session for future use
            session_ref = db.collection('annotated_slide_decks').document(session_id)
            session_ref.update({'processed_content': lecture_content})
            print(f"Saved processed content for existing session {session_id}")
        else:
            print(f"Using cached processed content for session {session_id}")
        
        # Generate AI response
        print(f"Generating AI response for question: {question}")
        ai_response = generate_ai_tutor_response(question, lecture_content, current_slide)
        print(f"Generated AI response: {ai_response[:100]}...")
        
        return jsonify({
            "success": True,
            "answer": ai_response
        })
        
    except Exception as e:
        print(f"Error in AI tutor: {e}")
        return jsonify({"success": False, "message": "Failed to get AI response"}), 500


def process_lecture_for_ai_tutor(session_data):
    """Process lecture content for AI tutor analysis - FAST VERSION."""
    try:
        # Get the PDF file
        source_file_path = session_data.get('source_file_path')
        if not source_file_path:
            print("No source_file_path found in session data")
            return None
        
        print(f"Fast processing PDF from path: {source_file_path}")
        
        # Download PDF from Cloud Storage
        blob = bucket.blob(source_file_path)
        pdf_content = blob.download_as_bytes()
        print(f"Downloaded PDF with {len(pdf_content)} bytes")
        
        # Extract text from all pages - NO AI CALLS
        import PyPDF2
        import io
        
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
        total_pages = len(pdf_reader.pages)
        print(f"PDF has {total_pages} pages")
        
        processed_content = {
            'total_pages': total_pages,
            'pages': []
        }
        
        # Process all pages quickly without AI summaries
        for page_num in range(total_pages):
            try:
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                if text.strip():
                    # Create simple summary by taking first 200 characters
                    simple_summary = text.strip()[:200] + "..." if len(text.strip()) > 200 else text.strip()
                    
                    processed_content['pages'].append({
                        'page_number': page_num + 1,
                        'text': text.strip(),
                        'summary': simple_summary  # No AI call - just truncate
                    })
                    print(f"Processed page {page_num + 1} with {len(text)} characters")
            except Exception as page_error:
                print(f"Error processing page {page_num + 1}: {page_error}")
                continue
        
        print(f"Successfully processed {len(processed_content['pages'])} pages in FAST mode")
        return processed_content
        
    except Exception as e:
        print(f"Error processing lecture for AI tutor: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_page_summary(page_text):
    """Generate a brief summary of a page for AI tutor context."""
    try:
        # Truncate page text to avoid token limits
        truncated_text = page_text[:1000] if len(page_text) > 1000 else page_text
        
        prompt = f"""Summarize this slide content in 1-2 sentences focusing on key concepts:

{truncated_text}

SUMMARY:"""
        
        response = client.chat.completions.create(
            model="gpt-4",  # Changed to GPT-4 for consistency
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80,  # Reduced for faster processing
            temperature=0.2  # Lower temperature for more focused summaries
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Error generating page summary: {e}")
        return "Content summary unavailable"


def generate_ai_tutor_response(question, lecture_content, current_slide):
    """Generate AI tutor response based on question and lecture content - ULTRA FAST."""
    try:
        # Build minimal context - only what's needed
        context_pages = []
        
        # Always include current slide first with more text
        current_page = None
        for page in lecture_content['pages']:
            if page['page_number'] == current_slide:
                current_page = page
                break
        
        if current_page:
            context_pages.append(f"Slide {current_page['page_number']}: {current_page['text'][:800]}")
        
        # Add only 3 other slides with minimal text
        other_pages = [p for p in lecture_content['pages'][:6] if p['page_number'] != current_slide]
        for page in other_pages[:3]:
            context_pages.append(f"Slide {page['page_number']}: {page['text'][:300]}")
        
        context = "\n".join(context_pages)
        
        # Detailed prompt similar to study hub tutor but optimized for brief responses
        prompt = f"""You are an expert AI tutor helping a student understand their lecture material. Answer the user's question based on the provided lecture content. Be concise, helpful, and accurate.

LECTURE CONTENT:
{context}

STUDENT QUESTION: {question}

INSTRUCTIONS:
- Provide a clear, helpful answer based on the lecture content
- Keep your response to 1-2 sentences maximum
- Be direct and precise in your explanation
- If the question mentions a specific slide number, reference that slide specifically
- If asking about a specific slide, focus on that slide's content
- If the information isn't clearly available in the content, say so politely
- Use a friendly, encouraging tone
- Reference slide numbers when relevant

ANSWER:"""
        
        print(f"Sending ultra-fast prompt to GPT-4 with {len(prompt)} characters")
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,  # Increased for more detailed brief responses
            temperature=0.3  # Slightly higher for more natural responses
        )
        
        answer = response.choices[0].message.content.strip()
        print(f"Received ultra-fast GPT-4 response: {answer}")
        
        return answer
        
    except Exception as e:
        print(f"Error generating AI tutor response: {e}")
        return "Sorry, I'm having trouble right now. Please try rephrasing your question."


@app.route("/slide_notes/<session_id>/view_flashcards")
@login_required
def view_lecture_flashcards(session_id):
    """View flashcards for a specific lecture session."""
    try:
        session_doc = db.collection('annotated_slide_decks').document(session_id).get()
        if not session_doc.exists:
            flash("Session not found", "error")
            return redirect(url_for("dashboard"))
        
        session_data = session_doc.to_dict()
        if session_data.get('user_id') != current_user.id:
            flash("You do not have permission to view this.", "error")
            return redirect(url_for('dashboard'))
        
        flashcards_data = session_data.get('flashcards_data', [])
        if not flashcards_data:
            flash("No flashcards available for this lecture.", "error")
            return redirect(url_for('slide_notes_workspace', session_id=session_id))
        
        # Create a temporary activity for the flashcard viewer
        temp_activity = Activity(
            id=f"temp_{session_id}",
            hub_id=session_data.get('hub_id'),
            type='Flashcards',
            title=f"Flashcards - {session_data.get('title', 'Untitled')}",
            data={'cards': flashcards_data},
            status='completed'
        )
        
        return render_template('edit_flashcards.html', activity=temp_activity, is_lecture_flashcards=True, session_id=session_id)
        
    except Exception as e:
        print(f"Error viewing lecture flashcards: {e}")
        flash("An error occurred while loading flashcards.", "error")
        return redirect(url_for("dashboard"))


# ==============================================================================
# 5. STRIPE & SUBSCRIPTION ROUTES
# ==============================================================================

@app.route('/create-checkout-session', methods=['POST'])
@login_required
def create_checkout_session():
    price_id = request.form.get('price_id')
    print(f"Creating checkout session with price_id: {price_id}")
    
    try:
        # Check if user was referred (has a referred_by value)
        user_doc = db.collection('users').document(current_user.id).get()
        user_data = user_doc.to_dict() if user_doc.exists else {}
        was_referred = user_data.get('referred_by') is not None
        
        # Create line items with discount if user was referred
        line_items = [{
            'price': price_id,
            'quantity': 1,
        }]
        
        # If user was referred, add a 50% discount
        discounts = []
        if was_referred:
            # Create a 50% discount coupon
            try:
                coupon = stripe.Coupon.create(
                    percent_off=50,
                    duration='once',
                    name='Referral Discount',
                    id=f'referral_{current_user.id}_{int(datetime.now().timestamp())}'
                )
                discounts.append({'coupon': coupon.id})
                print(f"🎯 Created 50% discount coupon for referred user {current_user.email}")
            except Exception as e:
                print(f"❌ Failed to create discount coupon: {e}")
        
        checkout_session = stripe.checkout.Session.create(
            line_items=line_items,
            mode='subscription',
            success_url=YOUR_DOMAIN + '/dashboard?session_id={CHECKOUT_SESSION_ID}',
            cancel_url=YOUR_DOMAIN + '/',
            customer_email=current_user.email,
            discounts=discounts if discounts else None,
            # Pass user ID to identify the user in webhook
            metadata={'user_id': current_user.id, 'was_referred': str(was_referred)}
        )
    except Exception as e:
        print(f"❌ Error creating checkout session: {e}")
        return str(e)

    return redirect(checkout_session.url, code=303)

@app.route('/stripe-webhook', methods=['POST'])
def stripe_webhook():
    print("🔔 Stripe webhook received!")
    payload = request.data
    sig_header = request.headers.get('Stripe-Signature')
    endpoint_secret = os.getenv("STRIPE_ENDPOINT_SECRET")
    event = None

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, endpoint_secret
        )
        print(f"✅ Webhook event verified: {event['type']}")
    except ValueError as e:
        print(f"❌ Invalid payload: {e}")
        return 'Invalid payload', 400
    except stripe.error.SignatureVerificationError as e:
        print(f"❌ Invalid signature: {e}")
        return 'Invalid signature', 400

    # Handle the checkout.session.completed event
    if event['type'] == 'checkout.session.completed':
        print("🎉 Processing checkout.session.completed event")
        session = event['data']['object']
        user_id = session.get('metadata', {}).get('user_id')
        if user_id:
            user_ref = db.collection('users').document(user_id)
            user_ref.update({
                'subscription_tier': 'pro',
                'subscription_active': True,
                'stripe_customer_id': session.customer,
                'stripe_subscription_id': session.subscription
            })
            print(f"User {user_id} successfully subscribed to Pro plan.")
            
            # Verify the update worked
            updated_user_doc = db.collection('users').document(user_id).get()
            if updated_user_doc.exists:
                updated_data = updated_user_doc.to_dict()
                print(f"✅ User subscription updated - Tier: {updated_data.get('subscription_tier')}, Active: {updated_data.get('subscription_active')}")
            else:
                print(f"❌ Failed to verify user subscription update")
            
            # --- NEW: Process Referral Rewards ---
            try:
                print(f"🎯 Starting referral rewards processing for user {user_id}")
                process_referral_rewards(user_id)
                print(f"✅ Completed referral rewards processing for user {user_id}")
            except Exception as e:
                print(f"❌ Error processing referral rewards for user {user_id}: {e}")
                import traceback
                traceback.print_exc()

    # Handle other subscription events like cancellations
    elif event['type'] == 'customer.subscription.deleted':
        subscription = event['data']['object']
        stripe_customer_id = subscription.customer
        users_ref = db.collection('users').where('stripe_customer_id', '==', stripe_customer_id).stream()
        for user_doc in users_ref:
            user_doc.reference.update({
                'subscription_tier': 'free',
                'subscription_active': False
            })
            print(f"User {user_doc.id}'s subscription was cancelled.")

    return 'Success', 200

@app.route('/debug/subscription-status')
@login_required
def debug_subscription_status():
    """Debug route to check user's subscription status"""
    user_doc = db.collection('users').document(current_user.id).get()
    if user_doc.exists:
        user_data = user_doc.to_dict()
        return jsonify({
            "user_id": current_user.id,
            "email": current_user.email,
            "subscription_tier": user_data.get('subscription_tier'),
            "subscription_active": user_data.get('subscription_active'),
            "stripe_customer_id": user_data.get('stripe_customer_id'),
            "stripe_subscription_id": user_data.get('stripe_subscription_id'),
            "referred_by": user_data.get('referred_by'),
            "pro_referral_count": user_data.get('pro_referral_count', 0)
        })
    return jsonify({"error": "User not found"}), 404

@app.route('/debug/webhook-test')
def webhook_test():
    """Test route to verify webhook endpoint is accessible"""
    return jsonify({
        "status": "Webhook endpoint is accessible",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint_secret_configured": bool(os.getenv("STRIPE_ENDPOINT_SECRET"))
    })

@app.route('/api/referral-discount-status', methods=['GET'])
@login_required
def get_referral_discount_status():
    """Check if user is eligible for referral discount"""
    try:
        user_doc = db.collection('users').document(current_user.id).get()
        user_data = user_doc.to_dict() if user_doc.exists else {}
        was_referred = user_data.get('referred_by') is not None
        
        return jsonify({
            "success": True,
            "eligible_for_discount": was_referred,
            "original_price": 4.49,
            "discounted_price": 2.25 if was_referred else 4.49
        })
    except Exception as e:
        print(f"Error checking referral discount status: {e}")
        return jsonify({"success": False, "message": "Error checking discount status"}), 500

# ==============================================================================
# 6. CORE APP & HUB ROUTES
# ==============================================================================

@app.route("/hub/<hub_id>/delete")
@login_required
def delete_hub(hub_id):
    try:
        hub_doc = db.collection('hubs').document(hub_id).get()
        if not hub_doc.exists or hub_doc.to_dict().get('user_id') != current_user.id:
            flash("Hub not found or you don't have permission.", "error")
            return redirect(url_for('dashboard'))

        blobs = bucket.list_blobs(prefix=f"hubs/{hub_id}/")
        for blob in blobs: blob.delete()
        collections_to_delete = ['notes', 'activities', 'lectures', 'notifications', 'annotated_slide_decks', 'sessions', 'folders']
        for coll in collections_to_delete:
            docs = db.collection(coll).where('hub_id', '==', hub_id).stream()
            for doc in docs: doc.reference.delete()
        db.collection('hubs').document(hub_id).delete()
        flash(f"Hub and all its data have been successfully deleted.", "success")
    except Exception as e:
        print(f"Error deleting hub {hub_id}: {e}")
        flash("An error occurred while trying to delete the hub.", "error")
    return redirect(url_for('dashboard'))

@app.route("/flashcards/<activity_id>/export")
@login_required
def export_flashcards(activity_id):
    # 1. Get the format and fetch the flashcard data from Firestore
    export_format = request.args.get('format')
    activity_doc = db.collection('activities').document(activity_id).get()

    if not activity_doc.exists:
        return "Flashcard set not found.", 404

    activity = Activity.from_dict(activity_doc.to_dict())
    cards = activity.data.get('cards', [])
    
    string_buffer = io.StringIO()

    if export_format in ['quizlet', 'anki']:
        for card in cards:
            front = card.get('front', '').replace('\n', ' ')
            back = card.get('back', '').replace('\n', ' ')
            string_buffer.write(f"{front}\t{back}\n")
        
        mimetype = 'text/plain'
        filename = f"{activity.title or 'flashcards'}.txt"

    elif export_format == 'notion':
        writer = csv.writer(string_buffer)
        writer.writerow(['Front', 'Back'])
        for card in cards:
            writer.writerow([card.get('front', ''), card.get('back', '')])

        mimetype = 'text/csv'
        filename = f"{activity.title or 'flashcards'}.csv"
    
    else:
        return "Invalid export format specified.", 400

    response_data = string_buffer.getvalue()
    return Response(
        response_data,
        mimetype=mimetype,
        headers={"Content-Disposition": f"attachment;filename={filename}"}
    )


@app.route("/note/<note_id>")
@login_required
def view_note(note_id):
    # First try to find it as a regular note
    note_doc = db.collection('notes').document(note_id).get()
    if note_doc.exists:
        note = Note.from_dict(note_doc.to_dict())
        hub_doc = db.collection('hubs').document(note.hub_id).get()
        hub_name = hub_doc.to_dict().get('name') if hub_doc.exists else "Hub"
    else:
        # If not found as a note, try to find it as an activity with note content
        activity_doc = db.collection('activities').document(note_id).get()
        if not activity_doc.exists:
            return "Note not found", 404
        
        activity = Activity.from_dict(activity_doc.to_dict())
        hub_doc = db.collection('hubs').document(activity.hub_id).get()
        hub_name = hub_doc.to_dict().get('name') if hub_doc.exists else "Hub"
        
        # Check if this activity contains note content
        if activity.data and activity.data.get('type') == 'note' and activity.data.get('content_html'):
            # Create a temporary note object for the template
            note = Note(
                id=activity.id,
                hub_id=activity.hub_id,
                title=activity.title,
                content_html=activity.data['content_html']
            )
        else:
            return "Note not found", 404

    # Check for special note types based on title
    if "Mind Map for" in note.title:
        try:
            # Try to load the content as JSON for the mind map
            mind_map_data = json.loads(note.content_html)
            return render_template("mind_map.html", note=note, hub_name=hub_name, mind_map_data=mind_map_data)
        except json.JSONDecodeError:
            # If it fails, it's just text, render normally
            pass
    elif "Cheat Sheet for" in note.title:
        try:
            # The content should be a JSON object with the cheat sheet structure
            cheat_sheet_data = json.loads(note.content_html)
            return render_template("cheat_sheet.html", note=note, hub_name=hub_name, cheat_sheet_data=cheat_sheet_data)
        except json.JSONDecodeError:
            pass

    # Fallback to the standard note view
    return render_template("note.html", note=note, hub_name=hub_name)

@app.route("/hub/<hub_id>/setup_session", methods=["POST"])
@login_required
def setup_session(hub_id):
    if current_user.subscription_tier not in ['pro', 'admin']:
        flash("Interactive Sessions are a Pro feature. Please upgrade your plan.", "warning")
        return redirect(url_for('hub_page', hub_id=hub_id))

    selected_files = request.form.getlist('selected_files')
    if not selected_files:
        flash("Please select at least one file to start a session.", "error")
        return redirect(url_for('hub_page', hub_id=hub_id))
    
    return render_template("setup_session.html", hub_id=hub_id, selected_files=selected_files)

@app.route("/hub/<hub_id>/create_session", methods=["POST"])
@login_required
def create_session(hub_id):
    if current_user.subscription_tier not in ['pro', 'admin']:
        flash("Interactive Sessions are a Pro feature. Please upgrade your plan.", "warning")
        return redirect(url_for('hub_page', hub_id=hub_id))

    form = request.form
    duration = form.get('duration', '30')
    focus = form.get('focus', 'understanding')
    selected_files = form.getlist('selected_files')
    hub_text = get_text_from_hub_files(selected_files)

    if not hub_text:
        flash("Could not extract text from the selected files.", "error")
        return redirect(url_for('hub_page', hub_id=hub_id))

    try:
        plan_json_str = generate_full_study_session(hub_text, duration, focus)
        study_plan = safe_load_json(plan_json_str)
        
        if not study_plan.get('topics') or not isinstance(study_plan['topics'], list) or len(study_plan['topics']) == 0:
            raise ValueError("AI failed to generate a valid study plan with topics.")

        full_notes_html = ""
        all_quizzes_data = []
        for topic in study_plan.get('topics', []):
            full_notes_html += f"<h2>{topic.get('title', 'Untitled Topic')}</h2>"
            full_notes_html += topic.get('content_html', '<p>No content was generated for this section.</p>')
            
            for interaction in topic.get('interactions', []):
                if interaction.get('type') == 'checkpoint_quiz':
                    quiz_questions = interaction.get('questions', [])
                    if quiz_questions:
                        all_quizzes_data.append(quiz_questions)
        
        flashcards_raw = generate_flashcards_from_text(hub_text)
        all_flashcards = parse_flashcards(flashcards_raw)
        
        first_file_name = os.path.basename(selected_files[0]).replace('.pdf', '')
        
        batch = db.batch()
        session_ref = db.collection('sessions').document()
        
        note_ref = db.collection('notes').document()
        new_note = Note(id=note_ref.id, hub_id=hub_id, title=f"Notes for Session: {first_file_name}", content_html=full_notes_html)
        batch.set(note_ref, new_note.to_dict())

        flashcard_ref = db.collection('activities').document()
        new_flashcards = Activity(id=flashcard_ref.id, hub_id=hub_id, type='Flashcards', title=f"Flashcards for Session: {first_file_name}", data={'cards': all_flashcards}, status='completed')
        batch.set(flashcard_ref, new_flashcards.to_dict())

        quiz_ids = []
        for i, questions in enumerate(all_quizzes_data):
            quiz_ref = db.collection('activities').document()
            new_quiz = Activity(id=quiz_ref.id, hub_id=hub_id, type='Quiz', title=f"Checkpoint Quiz {i+1} for Session", data={'questions': questions})
            batch.set(quiz_ref, new_quiz.to_dict())
            quiz_ids.append(quiz_ref.id)

        new_session = StudySession(
            id=session_ref.id, hub_id=hub_id, title=f"Interactive Session: {first_file_name}",
            source_files=selected_files, session_plan=study_plan, note_id=note_ref.id,
            flashcard_activity_id=flashcard_ref.id, quiz_activity_ids=quiz_ids
        )
        batch.set(session_ref, new_session.to_dict())

        # --- NEW: Create Notification ---
        notification_ref = db.collection('notifications').document()
        message = f"Your interactive session '{new_session.title}' is ready to start."
        link = url_for('start_study_session', session_id=new_session.id)
        new_notification = Notification(id=notification_ref.id, hub_id=hub_id, message=message, link=link)
        batch.set(notification_ref, new_notification.to_dict())
        
        batch.commit()

        return redirect(url_for('start_study_session', session_id=session_ref.id))

    except Exception as e:
        print(f"Error creating session: {e}")
        flash(f"An error occurred while creating your session: {e}", "error")
        return redirect(url_for('hub_page', hub_id=hub_id))

@app.route("/session/<session_id>")
@login_required
def start_study_session(session_id):
    session_doc = db.collection('sessions').document(session_id).get()
    if not session_doc.exists:
        return "Study session not found.", 404
    session = StudySession.from_dict(session_doc.to_dict())
    return render_template("study_session.html", session=session)


@app.route("/session/evaluate_teach_back", methods=["POST"])
@login_required
def evaluate_teach_back():
    data = request.get_json()
    model_answer = data.get('model_answer')
    student_answer = data.get('student_answer')

    if not model_answer or not student_answer:
        return jsonify({"error": "Missing data"}), 400

    feedback_json_str = grade_answer_with_ai(
        question="The student was asked to explain a concept. Evaluate their explanation against the model answer.",
        model_answer=model_answer,
        student_answer=student_answer
    )
    feedback_data = safe_load_json(feedback_json_str)
    return jsonify(feedback_data)


@app.route("/session/<session_id>/complete", methods=["POST"])
@login_required
def complete_session(session_id):
    results_data = request.json.get('results')
    session_ref = db.collection('sessions').document(session_id)
    session_ref.update({
        'status': 'completed',
        'results': results_data
    })
    return jsonify({"success": True, "report_url": url_for('session_report', session_id=session_id)})


@app.route("/session/<session_id>/report")
@login_required
def session_report(session_id):
    session_doc = db.collection('sessions').document(session_id).get()
    if not session_doc.exists: return "Session report not found.", 404
    session = StudySession.from_dict(session_doc.to_dict())
    hub_text = get_text_from_hub_files(session.source_files)
    
    topic_scores = {}
    for result in session.results:
        topic_title = result.get('topic_title', 'Unknown Topic')
        if topic_title not in topic_scores:
            topic_scores[topic_title] = {'score': 0, 'total': 0}
        
        is_correct = result.get('is_correct', False) or result.get('score', 0) >= 7
        topic_scores[topic_title]['score'] += 1 if is_correct else 0
        topic_scores[topic_title]['total'] += 1

    weak_topics = [
        topic for topic, data in topic_scores.items() 
        if data['total'] > 0 and (data['score'] / data['total']) < 0.7
    ]
    
    remedial_links = {}
    if weak_topics and hub_text:
        batch = db.batch()
        for topic in weak_topics:
            try:
                materials_json_str = generate_remedial_materials(hub_text, topic)
                materials = safe_load_json(materials_json_str)
                
                quiz_data = materials.get("questions", [])

                remedial_note_ref = db.collection('notes').document()
                remedial_note = Note(id=remedial_note_ref.id, hub_id=session.hub_id, title=f"Review Notes: {topic}", content_html=materials.get("notes_html", ""))
                batch.set(remedial_note_ref, remedial_note.to_dict())
                
                remedial_quiz_ref = db.collection('activities').document()
                remedial_quiz = Activity(id=remedial_quiz_ref.id, hub_id=session.hub_id, type='Quiz', title=f"Review Quiz: {topic}", data={'questions': quiz_data})
                batch.set(remedial_quiz_ref, remedial_quiz.to_dict())
                
                remedial_links[topic] = {
                    "note_id": remedial_note_ref.id,
                    "quiz_id": remedial_quiz_ref.id
                }
            except Exception as e:
                print(f"Failed to generate remedial content for topic '{topic}': {e}")
        batch.commit()
    
    total_questions = len(session.results)
    correct_answers = sum(1 for r in session.results if r.get('is_correct') or r.get('score', 0) >= 7)
    final_score = int((correct_answers / total_questions) * 100) if total_questions > 0 else 0

    return render_template("session_report.html", session=session, final_score=final_score, weak_topics=weak_topics, remedial_links=remedial_links)

@app.route('/explain_selection', methods=['POST'])
@login_required
def explain_selection():
    data = request.get_json()
    selected_text = data.get('selected_text')
    context_text = data.get('context_text')
    explanation_type = data.get('explanation_type')

    if not selected_text or not explanation_type:
        return jsonify({"error": "Missing required data"}), 400

    prompts = {
        'simplify': f"""
            You are an expert educator. Rephrase the following **'Selected Text'** in clear, plain language.
            Use the **'Surrounding Context'** to understand the topic. Your explanation must be direct and concise.

            **Surrounding Context:** "{context_text}"
            **Selected Text:** "{selected_text}"
        """,
        'example': f"""
            You are a helpful tutor. Provide a concrete, real-world example that illustrates the core concept in the **'Selected Text'**.
            Use the **'Surrounding Context'** to inform the example. **Limit your response to a maximum of 4 sentences.**

            **Surrounding Context:** "{context_text}"
            **Selected Text:** "{selected_text}"
        """,
        'significance': f"""
            You are an academic expert. Explain the significance of the **'Selected Text'** ("Why It Matters").
            Focus on its relevance for an exam or its practical use. **Your explanation must be concise and no more than 4 sentences long.**

            **Surrounding Context:** "{context_text}"
            **Selected Text:** "{selected_text}"
        """
    }

    prompt = prompts.get(explanation_type)
    if not prompt:
        return jsonify({"error": "Invalid explanation type"}), 400

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        explanation = response.choices[0].message.content
        return jsonify({"explanation": markdown.markdown(explanation)})
    except Exception as e:
        print(f"Error in explain_selection: {e}")
        return jsonify({"error": "Failed to generate explanation."}), 500
    
@app.route("/hub/<hub_id>/stuck_on_question/clarify_step", methods=["POST"])
@login_required
def clarify_step(hub_id):
    data = request.get_json()
    stuck_step = data.get("stuck_step", "")
    clarification_request = data.get("clarification_request", "")
    original_question = data.get("original_question", "")
    solution_context = data.get("solution_context", [])

    if not clarification_request:
        return jsonify({"success": False, "message": "Missing clarification request"}), 400

    # Build a friendly tutor-style prompt
    prompt = f"""
    You are a friendly AI tutor helping a student through a worked solution step by step.

    The student’s original problem:
    "{original_question}"

    The solution so far:
    {" ".join(solution_context)}

    The current step they are stuck on:
    "{stuck_step}"

    The student’s clarification question:
    "{clarification_request}"

    Please provide a clear, beginner-friendly explanation that directly addresses their question.
    Use plain language, and include simple examples or analogies where possible.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a supportive tutor. Always explain concepts step by step in simple, encouraging language."},
                {"role": "user", "content": prompt}
            ]
        )
        clarification = response.choices[0].message.content
        return jsonify({"success": True, "clarification": clarification})
    except Exception as e:
        print(f"Error in clarify_step: {e}")
        return jsonify({"success": False, "message": "Failed to generate clarification."}), 500


@app.route('/generate_mini_quiz', methods=['POST'])
@login_required
def generate_mini_quiz():
    data = request.get_json()
    selected_text = data.get('selected_text')
    context_text = data.get('context_text')
    
    if not selected_text:
        return jsonify({"error": "Missing selected text"}), 400

    prompt = f"""
        You are a quiz generator. Create a 1-2 question knowledge check based ONLY on the following 'Selected Text'.
        Your response MUST be a single, valid JSON object with one key: "questions".
        Each object in the "questions" array must have: "question" (string), "options" (an array of 3-4 strings), and "correct_answer" (string, which must be one of the options).
        Use the 'Surrounding Context' only to understand the topic's domain.

        **Surrounding Context:** "{context_text}"
        **Selected Text:** "{selected_text}"
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        quiz_data = json.loads(response.choices[0].message.content)
        return jsonify(quiz_data)
    except Exception as e:
        print(f"Error generating mini quiz: {e}")
        return jsonify({"error": "Failed to generate quiz."}), 500

@app.route("/flashcards/<activity_id>")
@login_required
def view_flashcards(activity_id):
    activity_doc = db.collection('activities').document(activity_id).get()
    if not activity_doc.exists: return "Flashcard set not found.", 404
    activity = Activity.from_dict(activity_doc.to_dict())
    return render_template("flashcards.html", activity=activity)

@app.route("/flashcards/<activity_id>/edit_set")
@login_required
def edit_flashcard_set(activity_id):
    activity_doc = db.collection('activities').document(activity_id).get()
    if not activity_doc.exists:
        flash("Flashcard set not found.", "error")
        return redirect(url_for('dashboard'))
    activity = Activity.from_dict(activity_doc.to_dict())
    return render_template("edit_flashcards.html", activity=activity)

@app.route("/flashcards/<activity_id>/update_set", methods=["POST"])
@login_required
def update_flashcard_set(activity_id):
    form_data = request.form
    new_cards = []
    i = 0
    while f'front_{i}' in form_data:
        front = form_data.get(f'front_{i}')
        back = form_data.get(f'back_{i}')
        if front and back and front.strip() and back.strip():
            new_cards.append({'front': front.strip(), 'back': back.strip()})
        i += 1
    
    try:
        activity_ref = db.collection('activities').document(activity_id)
        activity_ref.update({'data.cards': new_cards})
        flash("Flashcards updated successfully!", "success")
    except Exception as e:
        flash(f"An error occurred: {e}", "error")

    return redirect(url_for('edit_flashcard_set', activity_id=activity_id))

@app.route('/explain-formula', methods=['POST'])
@login_required
def explain_formula():
    formula = request.json.get('formula')
    if not formula: return jsonify({"error": "No formula provided"}), 400
    prompt = f"Explain the formula `{formula}` step-by-step for a university student. Break down each component and its significance. Use Markdown for formatting, including bullet points for clarity."
    response = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}])
    explanation_markdown = response.choices[0].message.content
    explanation_html = markdown.markdown(explanation_markdown)
    return jsonify({"explanation": explanation_html})

@app.route("/quiz/<activity_id>/delete", methods=["POST"])
@login_required
def delete_quiz(activity_id):
    try:
        db.collection('activities').document(activity_id).delete()
        return jsonify({"success": True, "message": "Quiz deleted successfully."})
    except Exception as e:
        print(f"Error deleting activity {activity_id}: {e}")
        return jsonify({"success": False, "message": "An error occurred."}), 500

@app.route("/quiz/<activity_id>/edit", methods=["POST"])
@login_required
def edit_quiz(activity_id):
    data = request.get_json()
    new_title = data.get('new_title')
    if not new_title: return jsonify({"success": False, "message": "New title is required."}), 400
    try:
        db.collection('activities').document(activity_id).update({'title': new_title})
        return jsonify({"success": True, "message": "Quiz title updated."})
    except Exception as e:
        print(f"Error updating activity {activity_id}: {e}")
        return jsonify({"success": False, "message": "An error occurred."}), 500

@app.route("/hub/<hub_id>/delete_file")
@login_required
def delete_file(hub_id):
    file_path = request.args.get('file_path')
    if not file_path:
        flash("Missing file information.", "error")
        return redirect(url_for('hub_page', hub_id=hub_id))
    blob = bucket.blob(file_path)
    if blob.exists(): blob.delete()
    hub_ref = db.collection('hubs').document(hub_id)
    hub_doc = hub_ref.get()
    if hub_doc.exists:
        current_files = hub_doc.to_dict().get('files', [])
        updated_files = [f for f in current_files if f.get('path') != file_path]
        hub_ref.update({'files': updated_files})
        flash("File deleted successfully.", "success")
    return redirect(url_for('hub_page', hub_id=hub_id))

@app.route("/hub/<hub_id>/upload", methods=["POST"])
@login_required
def upload_file(hub_id):
    if 'pdf' not in request.files:
        return jsonify({"success": False, "message": "No file part in the request."}), 400
    file = request.files['pdf']
    if file.filename == '':
        return jsonify({"success": False, "message": "No file selected."}), 400
    if not file or not file.filename.lower().endswith(".pdf"):
        return jsonify({"success": False, "message": "Invalid file type. Please upload a PDF."}), 400

    try:
        filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', file.filename)
        file_path = f"hubs/{hub_id}/{filename}"
        blob = bucket.blob(file_path)
        
        file.seek(0)
        blob.upload_from_file(file, content_type='application/pdf')
        
        file.seek(0, os.SEEK_END)
        file_info = {'name': filename, 'path': file_path, 'size': file.tell()}
        db.collection('hubs').document(hub_id).update({'files': firestore.ArrayUnion([file_info])})
        
        # When a file is uploaded, invalidate the tutor's vector store cache for this hub
        if hub_id in vector_store_cache:
            del vector_store_cache[hub_id]

        return jsonify({
            "success": True,
            "message": "File uploaded successfully.",
            "file_info": file_info
        })

    except Exception as e:
        print(f"Error in upload_file: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

# --- NEW: Route to generate preview data for a specific file and pipeline ---
@app.route("/hub/<hub_id>/generate_preview", methods=["POST"])
@login_required
def generate_preview(hub_id):
    data = request.get_json()
    file_path = data.get('file_path')
    preview_type = data.get('preview_type')

    if not file_path or not preview_type:
        return jsonify({"success": False, "message": "Missing file path or preview type."}), 400

    text = get_text_from_hub_files([file_path])
    if not text:
        return jsonify({"success": False, "message": "Could not extract text from the selected file."}), 500

    try:
        preview_json_str = ""
        if preview_type == 'revision':
            preview_json_str = generate_revision_preview(text)
        elif preview_type == 'exam':
            preview_json_str = generate_exam_preview(text)
        else:
            return jsonify({"success": False, "message": "Invalid preview type specified."}), 400
        
        preview_data = safe_load_json(preview_json_str)
        return jsonify({"success": True, "preview_data": preview_data})

    except Exception as e:
        print(f"Error generating preview for {file_path}: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

def generate_quiz_on_topic(text, topic):
    """Generates a targeted, 5-question quiz on a specific topic."""
    prompt = f"""
    You are an expert educator. Your task is to create a targeted, 5-question practice quiz about one specific topic, based on the provided text.

    The quiz MUST be about this topic ONLY: "{topic}"

    Your response must be a single, valid JSON object containing one key: "questions".
    Each question object in the "questions" array must have "topic", "type", "question", and other relevant keys ("options", "correct_answer", "model_answer").
    Ensure the "topic" key for every question is set to "{topic}".

    Use the following text to generate the quiz:
    ---
    {text}
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content

@app.route("/hub/<hub_id>/generate_weakness_quiz/<topic>")
@login_required
def generate_weakness_quiz(hub_id, topic):
    if current_user.subscription_tier not in ['pro', 'admin']:
        flash("Targeted weakness quizzes are a Pro feature. Please upgrade.", "warning")
        return redirect(url_for('hub_page', hub_id=hub_id))

    hub_doc = db.collection('hubs').document(hub_id).get()
    if not hub_doc.exists:
        flash("Study hub not found.", "error")
        return redirect(url_for('dashboard'))
    
    hub = Hub.from_dict(hub_doc.to_dict())
    hub_text = get_text_from_hub_files([f['path'] for f in hub.files])

    if not hub_text:
        flash("No documents in this hub to generate a quiz from.", "error")
        return redirect(url_for('hub_page', hub_id=hub_id))

    quiz_json = generate_quiz_on_topic(hub_text, topic)
    quiz_data = safe_load_json(quiz_json)
    
    batch = db.batch()
    activity_ref = db.collection('activities').document()
    activity_title = f"Targeted Quiz: {topic}"
    new_activity = Activity(id=activity_ref.id, hub_id=hub_id, type='Quiz', title=activity_title, data={'questions': quiz_data.get('questions', [])})
    batch.set(activity_ref, new_activity.to_dict())

    # --- NEW: Create Notification ---
    notification_ref = db.collection('notifications').document()
    message = f"Your targeted quiz for '{topic}' has been generated."
    link = url_for('take_lecture_quiz', activity_id=new_activity.id)
    new_notification = Notification(id=notification_ref.id, hub_id=hub_id, message=message, link=link)
    batch.set(notification_ref, new_notification.to_dict())
    batch.commit()

    flash(f"Generated a special quiz to help you with {topic}!", "success")
    return redirect(url_for('take_lecture_quiz', activity_id=new_activity.id))

# ==============================================================================
# 5. UNIFIED AI TOOL & DOWNLOAD ROUTES
# ==============================================================================
@app.route("/hub/<hub_id>/one_click_study", methods=["POST"])
@login_required
def one_click_study_tool(hub_id):
    if current_user.subscription_tier not in ['pro', 'admin']:
        return jsonify({"success": False, "message": "One-Click Study Kits are a Pro feature. Please upgrade."}), 403

    selected_files = request.json.get('selected_files')
    if not selected_files:
        return jsonify({"success": False, "message": "No files were selected."}), 400

    hub_text = get_text_from_hub_files(selected_files)
    if not hub_text or len(hub_text) < 100:
        return jsonify({"success": False, "message": "Could not extract enough text from the selected document(s)."}), 500

    try:
        first_file_name = os.path.basename(selected_files[0]).replace('.pdf', '')
        lecture_title = f"Lecture for {first_file_name}"
        
        interactive_html = generate_interactive_notes_html(hub_text)
        if not interactive_html or len(interactive_html) < 50:
            raise ValueError("The AI failed to generate study notes.")
        
        note_ref = db.collection('notes').document()
        new_note = Note(id=note_ref.id, hub_id=hub_id, title=lecture_title, content_html=interactive_html)

        flashcards_raw = generate_flashcards_from_text(hub_text)
        flashcards_parsed = parse_flashcards(flashcards_raw)
        if not flashcards_parsed:
            raise ValueError("The AI failed to generate flashcards.")
        
        flashcard_title = f"Flashcards for {first_file_name}"
        activity_ref_fc = db.collection('activities').document()
        new_activity_fc = Activity(id=activity_ref_fc.id, hub_id=hub_id, type='Flashcards', data={'cards': flashcards_parsed}, status='completed', title=flashcard_title)
        
        quiz_ids = []
        new_quizzes = []
        for i in range(1, 4):
            quiz_json = generate_quiz_from_text(hub_text)
            quiz_data = safe_load_json(quiz_json)
            if not quiz_data.get('questions'):
                print(f"Warning: AI failed to generate questions for Quiz {i}.")
                continue

            quiz_title = f"Practice Quiz {i} for {first_file_name}"
            activity_ref_quiz = db.collection('activities').document()
            new_activity_quiz = Activity(
                id=activity_ref_quiz.id, 
                hub_id=hub_id, 
                type=f'Quiz',
                title=quiz_title, 
                data={'questions': quiz_data.get('questions', [])}
            )
            quiz_ids.append(new_activity_quiz.id)
            new_quizzes.append(new_activity_quiz)

        if not new_quizzes:
            raise ValueError("The AI failed to generate any practice quizzes.")

        batch = db.batch()
        batch.set(note_ref, new_note.to_dict())
        batch.set(activity_ref_fc, new_activity_fc.to_dict())

        for quiz_activity in new_quizzes:
            quiz_ref = db.collection('activities').document(quiz_activity.id)
            batch.set(quiz_ref, quiz_activity.to_dict())
        
        lecture_ref = db.collection('lectures').document()
        new_lecture = Lecture(
            id=lecture_ref.id, 
            hub_id=hub_id, 
            title=lecture_title, 
            note_id=new_note.id, 
            flashcard_id=new_activity_fc.id, 
            quiz_ids=quiz_ids, 
            source_files=selected_files
        )
        batch.set(lecture_ref, new_lecture.to_dict())
        
        batch.commit()

        return jsonify({"success": True, "message": "Study materials created successfully!", "lecture": new_lecture.to_dict()})

    except Exception as e:
        print(f"Error in one_click_study_tool: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/note/<note_id>/download_pdf")
@login_required
def download_note_pdf(note_id):
    note_doc = db.collection('notes').document(note_id).get()
    if not note_doc.exists: return "Note not found", 404
    note = Note.from_dict(note_doc.to_dict())
    pdf = NotesPDF()
    pdf.add_fonts()
    pdf.add_page()
    pdf.write_html_to_pdf(note.content_html)
    pdf_buffer = io.BytesIO(pdf.output()) 
    pdf_buffer.seek(0)
    return send_file(pdf_buffer, as_attachment=True, download_name=f"{note.title}.pdf", mimetype='application/pdf')

@app.route("/hub/<hub_id>/create_exam", methods=["POST"])
@login_required
def create_exam(hub_id):
    if current_user.subscription_tier not in ['pro', 'admin']:
        flash("The Mock Exam generator is a Pro feature. Please upgrade.", "warning")
        return redirect(url_for('hub_page', hub_id=hub_id))

    form_data = request.form
    selected_files = form_data.getlist('selected_files')
    activity_id = form_data.get('activity_id')
    difficulty = form_data.get('difficulty', 'Medium')
    num_mcq = form_data.get('num_mcq', 10, type=int)
    num_short = form_data.get('num_short', 5, type=int)
    num_essay = form_data.get('num_essay', 1, type=int)
    time_limit = form_data.get('time_limit', 90, type=int)
    hub_text = get_text_from_hub_files(selected_files)
    if not hub_text: return "Could not extract text from the selected files.", 400
    mcq_questions = generate_exam_questions(hub_text, difficulty, "mcq", num_mcq)
    short_questions = generate_exam_questions(hub_text, difficulty, "short", num_short)
    essay_questions = generate_exam_questions(hub_text, difficulty, "essay", num_essay)
    exam_data = {"section_a_mcq": mcq_questions, "section_b_short_answer": short_questions, "section_c_essay": essay_questions}
    return render_template("exam_interface.html", exam_data=exam_data, time_limit=time_limit, hub_id=hub_id, activity_id=activity_id)

@app.route("/hub/<hub_id>/grade_exam", methods=["POST"])
@login_required
def grade_exam_route(hub_id):
    form_data = request.form
    activity_id = form_data.get('activity_id')
    exam_data = {"section_a_mcq": safe_load_json(form_data.get("section_a_mcq")),"section_b_short_answer": safe_load_json(form_data.get("section_b_short_answer")),"section_c_essay": safe_load_json(form_data.get("section_c_essay"))}
    graded_results = {}
    total_mcq_score = 0
    mcq_section = exam_data.get('section_a_mcq', [])
    total_mcq_possible = len(mcq_section)
    if total_mcq_possible > 0:
        for i, q in enumerate(mcq_section):
            user_answer = form_data.get(f'mcq_answer_{i}')
            is_correct = (user_answer and user_answer.strip() == q['correct_answer'].strip())
            if is_correct: total_mcq_score += 1
            graded_results[f'mcq_{i}'] = {"user_answer": user_answer, "is_correct": is_correct}
    total_open_ended_score = 0
    total_open_ended_possible = 0
    sections_to_grade = {"short_answer": exam_data.get("section_b_short_answer", []), "essay": exam_data.get("section_c_essay", [])}
    for section_prefix, questions in sections_to_grade.items():
        if questions:
            total_open_ended_possible += len(questions) * 10
            for i, q in enumerate(questions):
                student_answer = form_data.get(f'{section_prefix}_answer_{i}')
                result_key = f'{section_prefix}_{i}'
                try:
                    feedback_json_str = grade_answer_with_ai(q.get('question'), q.get('model_answer'), student_answer)
                    feedback_data = json.loads(feedback_json_str)
                    score = feedback_data.get('score', 0)
                    total_open_ended_score += score
                    graded_results[result_key] = {"user_answer": student_answer, "feedback": feedback_data.get('feedback'), "score": score}
                except Exception as e:
                    print(f"Error grading answer for {result_key}: {e}")
                    graded_results[result_key] = {"user_answer": student_answer, "feedback": "Could not be graded automatically.", "score": 0}
    total_possible_score = total_mcq_possible + total_open_ended_possible
    total_achieved_score = total_mcq_score + total_open_ended_score
    final_percentage = int((total_achieved_score / total_possible_score) * 100) if total_possible_score > 0 else 0
    if activity_id:
        activity_ref = db.collection('activities').document(activity_id)
        activity_ref.update({'status': 'graded', 'score': final_percentage, 'exam_data': exam_data, 'graded_results': graded_results, 'mcq_score': total_mcq_score})
    flash(f"Exam graded! You scored {final_percentage}%.", "success")
    return redirect(url_for('hub_page', hub_id=hub_id))

# ==============================================================================
# 6. LECTURE & QUIZ ROUTES
# ==============================================================================
@app.route("/lecture/<lecture_id>")
@login_required
def lecture_page(lecture_id):
    lecture_doc = db.collection('lectures').document(lecture_id).get()
    if not lecture_doc.exists: return "Lecture not found.", 404
    lecture = Lecture.from_dict(lecture_doc.to_dict())
    note_doc = db.collection('notes').document(lecture.note_id).get()
    note = Note.from_dict(note_doc.to_dict()) if note_doc.exists else None
    flashcards_doc = db.collection('activities').document(lecture.flashcard_id).get()
    flashcards_data = flashcards_doc.to_dict().get('data', {}).get('cards', []) if flashcards_doc.exists else []
    quizzes = []
    for quiz_id in lecture.quiz_ids:
        quiz_doc = db.collection('activities').document(quiz_id).get()
        if quiz_doc.exists: quizzes.append(Activity.from_dict(quiz_doc.to_dict()))
    return render_template("lecture_page.html", lecture=lecture, note=note, flashcards=flashcards_data, quizzes=quizzes)

@app.route("/quiz/<activity_id>")
@login_required
def take_lecture_quiz(activity_id):
    quiz_doc = db.collection('activities').document(activity_id).get()
    if not quiz_doc.exists: return "Quiz not found.", 404
    quiz_activity = Activity.from_dict(quiz_doc.to_dict())
    questions = quiz_activity.data.get('questions', [])
    return render_template("quiz.html", questions=questions, time_limit=600, activity_id=activity_id)

@app.route("/quiz/<activity_id>/submit", methods=["POST"])
@login_required
def submit_quiz(activity_id):
    activity_ref = db.collection('activities').document(activity_id)
    activity_doc = activity_ref.get()
    if not activity_doc.exists:
        return "Quiz not found.", 404

    activity = Activity.from_dict(activity_doc.to_dict())
    questions = activity.data.get('questions', [])
    
    graded_answers = []
    total_achieved_score = 0
    total_possible_score = len(questions) * 10  # Each question is now worth 10 points

    for i, q in enumerate(questions):
        user_answer = request.form.get(f'question-{i}')
        question_topic = q.get('topic', 'General')
        
        # Default values for each graded answer
        score = 0
        is_correct = False
        feedback = "No feedback available."

        if q.get('type') == 'multiple_choice':
            correct_answer_text = q.get('correct_answer', '').strip()
            user_answer_text = user_answer.strip() if user_answer else ''
            
            is_correct = (user_answer and user_answer_text.lower() == correct_answer_text.lower())
            
            if is_correct:
                score = 10
                feedback = "Correct!"
            else:
                score = 0
                feedback = f"Incorrect. The correct answer was: {correct_answer_text}"
            
            total_achieved_score += score
            graded_answers.append({
                "user_answer": user_answer, 
                "correct": is_correct, 
                "score": score,
                "feedback": feedback,
                "topic": question_topic
            })
        
        elif q.get('type') in ['short_answer', 'explanation']:
            try:
                feedback_json_str = grade_answer_with_ai(q.get('question'), q.get('model_answer'), user_answer)
                feedback_data = json.loads(feedback_json_str)
                score = feedback_data.get('score', 0)
                feedback = feedback_data.get('feedback', 'Could not retrieve AI feedback.')
                is_correct = score >= 7 # Consider it "correct" if the score is 7/10 or higher
            except Exception as e:
                print(f"Error grading open-ended answer for quiz {activity_id}: {e}")
                score = 0
                feedback = "This answer could not be graded automatically due to an error."
                is_correct = False
            
            total_achieved_score += score
            graded_answers.append({
                "user_answer": user_answer, 
                "feedback": feedback, 
                "score": score, 
                "correct": is_correct,
                "topic": question_topic
            })
        
        # --- FIX: Add a catch-all 'else' to prevent crashes from unknown question types ---
        else:
            # Handle unknown or malformed question types gracefully
            graded_answers.append({
                "user_answer": user_answer or "No answer provided",
                "feedback": f"This question (type: {q.get('type', 'unknown')}) could not be graded automatically.",
                "score": 0,
                "correct": False,
                "topic": question_topic
            })


    final_percentage = int((total_achieved_score / total_possible_score) * 100) if total_possible_score > 0 else 0
    
    activity_ref.update({
        "status": "graded",
        "score": final_percentage,
        "graded_results": {
            "final_score": final_percentage,
            "questions": questions,
            "graded_answers": graded_answers
        }
    })

    # --- NEW: Update Hub Progress ---
    update_hub_progress(activity.hub_id, 15)

    return redirect(url_for('quiz_results', activity_id=activity_id))


@app.route("/quiz/<activity_id>/results")
@login_required
def quiz_results(activity_id):
    activity_doc = db.collection('activities').document(activity_id).get()
    if not activity_doc.exists: return "Quiz results not found.", 404
    activity = Activity.from_dict(activity_doc.to_dict())
    results = activity.graded_results
    return render_template("quiz_results.html", activity=activity, results=results)

@app.route("/flashcards/<activity_id>/save_game", methods=["POST"])
@login_required
def save_flashcard_game(activity_id):
    data = request.get_json()
    correct_indices = data.get('correct', [])
    incorrect_indices = data.get('incorrect', [])
    
    try:
        activity_ref = db.collection('activities').document(activity_id)
        activity_doc = activity_ref.get()
        if not activity_doc.exists:
            return jsonify({"success": False, "message": "Activity not found."}), 404

        activity = Activity.from_dict(activity_doc.to_dict())
        all_cards = activity.data.get('cards', [])

        incorrect_cards = [all_cards[i] for i in incorrect_indices if i < len(all_cards)]

        activity_ref.update({
            "status": "completed",
            "game_results": {
                "correct_count": len(correct_indices),
                "incorrect_count": len(incorrect_indices),
                "total_cards": len(all_cards),
                "last_played": datetime.now(timezone.utc),
                "incorrect_cards": incorrect_cards 
            }
        })
        
        # --- NEW: Update Hub Progress ---
        update_hub_progress(activity.hub_id, 5)

        return jsonify({"success": True, "message": "Game results saved."})
    except Exception as e:
        print(f"Error saving flashcard game for {activity_id}: {e}")
        return jsonify({"success": False, "message": "An error occurred."}), 500

@app.route("/lecture/<lecture_id>/delete")
@login_required
def delete_lecture(lecture_id):
    lecture_ref = db.collection('lectures').document(lecture_id)
    lecture_doc = lecture_ref.get()
    if not lecture_doc.exists: return "Lecture not found.", 404
    lecture = Lecture.from_dict(lecture_doc.to_dict())
    db.collection('notes').document(lecture.note_id).delete()
    db.collection('activities').document(lecture.flashcard_id).delete()
    for quiz_id in lecture.quiz_ids: db.collection('activities').document(quiz_id).delete()
    lecture_ref.delete()
    flash("Lecture and all associated data deleted successfully.", "success")
    return redirect(url_for('hub_page', hub_id=lecture.hub_id))

# ==============================================================================
# 6B. CALENDAR & SCHEDULING ROUTES (REPLACES AI TUTOR)
# ==============================================================================

@app.route("/hub/<hub_id>/events")
@login_required
def get_hub_events(hub_id):
    """API endpoint to fetch calendar events for a hub for FullCalendar."""
    try:
        start_str = request.args.get('start')
        end_str = request.args.get('end')

        start_dt = datetime.fromisoformat(start_str.replace('Z', '+00:00'))
        end_dt = datetime.fromisoformat(end_str.replace('Z', '+00:00'))

        events_query = db.collection('calendar_events').where('hub_id', '==', hub_id) \
            .where('start_time', '>=', start_dt) \
            .where('start_time', '<=', end_dt) \
            .stream()
        
        events_list = []
        for doc in events_query:
            event = CalendarEvent.from_dict(doc.to_dict())
            events_list.append({
                "id": event.id,
                "title": event.title,
                "start": event.start_time.isoformat(),
                "end": event.end_time.isoformat(),
                "allDay": event.all_day,
                "color": event.color,
                "extendedProps": {
                    "eventType": event.event_type,
                    "focus": event.focus or "No details provided.",
                    "exportUrl": url_for('export_ics', event_id=event.id)
                }
            })
            
        return jsonify(events_list)
    except Exception as e:
        print(f"Error fetching hub events: {e}")
        return jsonify({"success": False, "message": "An error occurred."}), 500


@app.route("/hub/<hub_id>/create_calendar_event", methods=["POST"])
@login_required
def create_calendar_event(hub_id):
    """Creates a new calendar event in Firestore."""
    data = request.get_json()
    try:
        event_title = data.get('title')
        event_type = data.get('event_type')
        start_str = data.get('start_time')
        duration_minutes = int(data.get('duration'))
        
        start_time = datetime.fromisoformat(start_str).replace(tzinfo=timezone.utc)
        end_time = start_time + timedelta(minutes=duration_minutes)

        event_ref = db.collection('calendar_events').document()
        new_event = CalendarEvent(
            id=event_ref.id,
            hub_id=hub_id,
            title=event_title,
            event_type=event_type,
            start_time=start_time,
            end_time=end_time,
            source_files=data.get('source_files', []),
            focus=data.get('focus')
        )
        event_ref.set(new_event.to_dict())

        return jsonify({"success": True, "message": "Event scheduled successfully!"})
    except Exception as e:
        print(f"Error creating calendar event: {e}")
        return jsonify({"success": False, "message": str(e)}), 500
        
@app.route("/event/<event_id>/export_ics")
@login_required
def export_ics(event_id):
    """Generates a downloadable .ics file for a calendar event."""
    event_doc = db.collection('calendar_events').document(event_id).get()
    if not event_doc.exists:
        return "Event not found", 404
    
    event_data = event_doc.to_dict()
    
    # Ensure datetime objects are timezone-aware (UTC)
    start_time = event_data['start_time'].replace(tzinfo=timezone.utc)
    end_time = event_data['end_time'].replace(tzinfo=timezone.utc)
    created_time = event_data['created_at'].replace(tzinfo=timezone.utc)

    start_utc = start_time.strftime('%Y%m%dT%H%M%SZ')
    end_utc = end_time.strftime('%Y%m%dT%H%M%SZ')
    created_utc = created_time.strftime('%Y%m%dT%H%M%SZ')
    
    description = f"Focus: {event_data.get('focus', 'N/A')}\\n"
    if event_data.get('source_files'):
        files_str = ", ".join([os.path.basename(f) for f in event_data['source_files']])
        description += f"Source Files: {files_str}"

    ics_content = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//MyStudyHub//NONSGML v1.0//EN",
        "BEGIN:VEVENT",
        f"UID:{event_doc.id}@mystudyhub.app",
        f"DTSTAMP:{created_utc}",
        f"DTSTART:{start_utc}",
        f"DTEND:{end_utc}",
        f"SUMMARY:{event_data['title']}",
        f"DESCRIPTION:{description}",
        "END:VEVENT",
        "END:VCALENDAR"
    ]
    
    response = Response("\r\n".join(ics_content), mimetype="text/calendar")
    response.headers["Content-Disposition"] = f"attachment; filename={secure_filename(event_data['title'])}.ics"
    return response


@app.route("/hub/<hub_id>/import_ics", methods=["POST"])
@login_required
def import_ics(hub_id):
    """Parses an uploaded .ics file and adds events to the hub's calendar."""
    if 'ics_file' not in request.files:
        flash("No file part in the request.", "error")
        return redirect(url_for('hub_page', hub_id=hub_id, _anchor='calendar'))

    file = request.files['ics_file']
    if file.filename == '':
        flash("No file selected.", "error")
        return redirect(url_for('hub_page', hub_id=hub_id, _anchor='calendar'))

    if file and file.filename.lower().endswith('.ics'):
        try:
            cal = Calendar.from_ical(file.read())
            batch = db.batch()
            event_count = 0

            for component in cal.walk():
                if component.name == "VEVENT":
                    event_count += 1
                    summary = str(component.get('summary', 'Untitled Event'))
                    start_dt = component.get('dtstart').dt
                    end_dt = component.get('dtend').dt

                    # Ensure datetime objects are timezone-aware (UTC)
                    if isinstance(start_dt, datetime) and start_dt.tzinfo is None:
                        start_dt = start_dt.replace(tzinfo=timezone.utc)
                    if isinstance(end_dt, datetime) and end_dt.tzinfo is None:
                        end_dt = end_dt.replace(tzinfo=timezone.utc)
                    
                    # Handle all-day events (which are of type date, not datetime)
                    all_day = not isinstance(start_dt, datetime)
                    if all_day:
                        start_dt = datetime.combine(start_dt, datetime.min.time(), tzinfo=timezone.utc)
                        end_dt = datetime.combine(end_dt, datetime.min.time(), tzinfo=timezone.utc)

                    event_ref = db.collection('calendar_events').document(str(uuid.uuid4()))
                    new_event = CalendarEvent(
                        id=event_ref.id,
                        hub_id=hub_id,
                        title=summary,
                        event_type="Imported",
                        start_time=start_dt,
                        end_time=end_dt,
                        all_day=all_day,
                        focus=str(component.get('description', ''))
                    )
                    batch.set(event_ref, new_event.to_dict())
            
            batch.commit()
            flash(f"Successfully imported {event_count} events from your calendar!", "success")

        except Exception as e:
            print(f"Error importing ICS file: {e}")
            flash("An error occurred while parsing the calendar file. It might be invalid.", "error")
    else:
        flash("Invalid file type. Please upload a .ics file.", "error")

    return redirect(url_for('hub_page', hub_id=hub_id, _anchor='calendar'))

# ==============================================================================
# 6A. ASSET MANAGEMENT & NOTIFICATION ROUTES
# ==============================================================================
@app.route("/notifications/mark_read", methods=["POST"])
@login_required
def mark_notifications_read():
    data = request.get_json()
    notification_ids = data.get('ids', [])
    if not notification_ids:
        return jsonify({"success": False, "message": "No notification IDs provided."}), 400
    try:
        batch = db.batch()
        for notif_id in notification_ids:
            notif_ref = db.collection('notifications').document(notif_id)
            batch.update(notif_ref, {'read': True})
        batch.commit()
        return jsonify({"success": True, "message": "Notifications marked as read."})
    except Exception as e:
        print(f"Error marking notifications as read: {e}")
        return jsonify({"success": False, "message": "An internal error occurred."}), 500
        
@app.route("/note/<note_id>/delete", methods=["POST"])
@login_required
def delete_note(note_id):
    try:
        # Get the note to verify ownership
        note_doc = db.collection('notes').document(note_id).get()
        if not note_doc.exists:
            return jsonify({"success": False, "message": "Note not found"}), 404
        
        note = Note.from_dict(note_doc.to_dict())
        
        # Verify hub ownership
        hub_doc = db.collection('hubs').document(note.hub_id).get()
        if not hub_doc.exists:
            return jsonify({"success": False, "message": "Hub not found"}), 404
        
        hub = Hub.from_dict(hub_doc.to_dict())
        if hub.user_id != current_user.id:
            return jsonify({"success": False, "message": "Permission denied"}), 403
        
        # Delete the note
        db.collection('notes').document(note_id).delete()
        return jsonify({"success": True, "message": "Note deleted successfully."})
    except Exception as e:
        print(f"Error deleting note {note_id}: {e}")
        return jsonify({"success": False, "message": "An error occurred."}), 500
    
@app.route("/hub/batch_delete", methods=["POST"])
@login_required
def batch_delete_assets():
    items_to_delete = request.json.get('items', [])
    if not items_to_delete:
        return jsonify({"success": False, "message": "No items provided."}), 400

    try:
        batch = db.batch()

        for item in items_to_delete:
            item_id = item.get('id')
            item_type = item.get('type')

            if not item_id or not item_type:
                continue

            if item_type == 'note':
                ref = db.collection('notes').document(item_id)
                batch.delete(ref)
            elif item_type in ['flashcards', 'quiz']:
                ref = db.collection('activities').document(item_id)
                batch.delete(ref)
            elif item_type == 'session':
                session_ref = db.collection('sessions').document(item_id)
                session_doc = session_ref.get()

                if session_doc.exists:
                    session_data = session_doc.to_dict()
                    
                    if session_data.get('note_id'):
                        batch.delete(db.collection('notes').document(session_data['note_id']))
                    
                    if session_data.get('flashcard_activity_id'):
                        batch.delete(db.collection('activities').document(session_data['flashcard_activity_id']))

                    for quiz_id in session_data.get('quiz_activity_ids', []):
                        batch.delete(db.collection('activities').document(quiz_id))
                    
                    batch.delete(session_ref)

        batch.commit()
        return jsonify({"success": True, "message": "Items deleted successfully."})

    except Exception as e:
        print(f"Error during batch delete: {e}")
        return jsonify({"success": False, "message": "An internal error occurred."}), 500

@app.route("/note/<note_id>/edit", methods=["POST"])
@login_required
def edit_note(note_id):
    data = request.get_json()
    new_title = data.get('new_title')
    if not new_title: return jsonify({"success": False, "message": "New title is required."}), 400
    try:
        db.collection('notes').document(note_id).update({'title': new_title})
        return jsonify({"success": True, "message": "Note title updated."})
    except Exception as e:
        print(f"Error updating note {note_id}: {e}")
        return jsonify({"success": False, "message": "An error occurred."}), 500

@app.route("/flashcards/<activity_id>/delete", methods=["POST"])
@login_required
def delete_flashcards(activity_id):
    try:
        db.collection('activities').document(activity_id).delete()
        return jsonify({"success": True, "message": "Flashcard set deleted successfully."})
    except Exception as e:
        print(f"Error deleting flashcard set {activity_id}: {e}")
        return jsonify({"success": False, "message": "An error occurred."}), 500

@app.route("/flashcards/<activity_id>/edit", methods=["POST"])
@login_required
def edit_flashcards(activity_id):
    data = request.get_json()
    new_title = data.get('new_title')
    if not new_title: return jsonify({"success": False, "message": "New title is required."}), 400
    try:
        db.collection('activities').document(activity_id).update({'title': new_title})
        return jsonify({"success": True, "message": "Flashcard set title updated."})
    except Exception as e:
        print(f"Error updating flashcard set {activity_id}: {e}")
        return jsonify({"success": False, "message": "An error occurred."}), 500

@app.route("/delete_all_hubs")
@login_required
def delete_all_hubs():
    try:
        hubs_ref = db.collection('hubs').where('user_id', '==', current_user.id).stream()
        hubs_to_delete = list(hubs_ref)
        if not hubs_to_delete:
            flash("There are no hubs to delete.", "info")
            return redirect(url_for('dashboard'))
        for hub_doc in hubs_to_delete:
            hub_id = hub_doc.id
            blobs = bucket.list_blobs(prefix=f"hubs/{hub_id}/")
            for blob in blobs: blob.delete()
            collections_to_delete = ['notes', 'activities', 'lectures', 'notifications']
            for coll in collections_to_delete:
                docs = db.collection(coll).where('hub_id', '==', hub_id).stream()
                for doc in docs: doc.reference.delete()
            hub_doc.reference.delete()
        flash(f"Successfully deleted all hubs and their data.", "success")
    except Exception as e:
        print(f"Error deleting all hubs: {e}")
        flash("An error occurred while trying to delete all hubs.", "error")
    return redirect(url_for('dashboard'))

# ==============================================================================
# 7. GUIDED WORKFLOW & PIPELINE ROUTES (RESTRUCTURED)
# ==============================================================================

@app.route("/hub/<hub_id>/build_revision_pack", methods=["POST"])
@login_required
def build_revision_pack(hub_id):
    data = request.get_json()
    selected_files = data.get('selected_files', [])
    num_flashcards = int(data.get('num_flashcards', 20))
    num_quiz_questions = int(data.get('num_quiz_questions', 10))
    include_notes = data.get('include_notes', False)

    if not selected_files:
        return jsonify({"success": False, "message": "No files were selected."}), 400

    hub_text = get_text_from_hub_files(selected_files)
    if not hub_text:
        return jsonify({"success": False, "message": "Could not extract text from files."}), 500

    try:
        first_file_name = os.path.basename(selected_files[0]).replace('.pdf', '')
        folder_name = f"Revision Pack for {first_file_name}"

        # Create activities with pending status for task queue
        activities_to_create = []

        if include_notes:
            note_ref = db.collection('activities').document()
            new_note_activity = Activity(
                id=note_ref.id, 
                hub_id=hub_id, 
                type='Notes', 
                title=f"Revision Notes for {first_file_name}", 
                status='pending',
                data={'selected_files': selected_files, 'text_length': len(hub_text)}
            )
            activities_to_create.append((note_ref, new_note_activity))

        if num_flashcards > 0:
            fc_ref = db.collection('activities').document()
            new_fc_activity = Activity(
                id=fc_ref.id, 
                hub_id=hub_id, 
                type='Flashcards', 
                title=f"Flashcards for {first_file_name}", 
                status='pending',
                data={'selected_files': selected_files, 'num_cards': num_flashcards, 'text_length': len(hub_text)}
            )
            activities_to_create.append((fc_ref, new_fc_activity))
        
        if num_quiz_questions > 0:
            quiz_ref = db.collection('activities').document()
            new_quiz_activity = Activity(
                id=quiz_ref.id, 
                hub_id=hub_id, 
                type='Quiz', 
                title=f"Practice Quiz for {first_file_name}", 
                status='pending',
                data={'selected_files': selected_files, 'num_questions': num_quiz_questions, 'text_length': len(hub_text)}
            )
            activities_to_create.append((quiz_ref, new_quiz_activity))
        
        # Create all activities with pending status
        batch = db.batch()
        folder_items = []
        
        for activity_ref, activity in activities_to_create:
            batch.set(activity_ref, activity.to_dict())
            # Map activity types to folder item types
            item_type = 'notes' if activity.type == 'Notes' else activity.type.lower()
            folder_items.append({'id': activity_ref.id, 'type': item_type})
        
        # Create the folder
        folder_ref = db.collection('folders').document()
        new_folder = Folder(id=folder_ref.id, hub_id=hub_id, name=folder_name, items=folder_items)
        batch.set(folder_ref, new_folder.to_dict())
        
        batch.commit()
        
        # Start background processing
        import threading
        threading.Thread(target=process_revision_pack_activities, args=(activities_to_create, hub_text, first_file_name)).start()
        
        return jsonify({"success": True, "message": f"Revision Pack '{folder_name}' is being generated. Check the task queue for progress."})

    except Exception as e:
        print(f"Error in build_revision_pack: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

def process_revision_pack_activities(activities_to_create, hub_text, first_file_name):
    """Background processing function for revision pack activities."""
    try:
        # Limit text length to prevent timeouts
        limited_text = hub_text[:8000] if len(hub_text) > 8000 else hub_text
        
        for activity_ref, activity in activities_to_create:
            try:
                print(f"Processing {activity.type} activity: {activity.title}")
                
                # Update status to processing
                activity_ref.update({'status': 'processing'})
                
                if activity.type == 'Notes':
                    # Generate notes
                    interactive_html = generate_interactive_notes_html(limited_text)
                    
                    # Update activity with the generated notes content
                    activity_ref.update({
                        'status': 'completed',
                        'data': {'content_html': interactive_html, 'type': 'note'}
                    })
                    
                elif activity.type == 'Flashcards':
                    # Generate flashcards
                    num_cards = activity.data.get('num_cards', 20)
                    flashcards_raw = generate_flashcards_from_text(limited_text, num_cards)
                    flashcards_parsed = parse_flashcards(flashcards_raw)
                    
                    if not flashcards_parsed:
                        # Fallback flashcards
                        flashcards_parsed = [
                    {"front": f"What is the main topic of {first_file_name}?", "back": "Review the document to understand the key concepts."},
                    {"front": f"List 3 key points from {first_file_name}", "back": "Identify the most important information from the document."},
                    {"front": f"How would you summarize {first_file_name}?", "back": "Create a brief overview of the main ideas and concepts."}
                ]
                
                    activity_ref.update({
                        'status': 'completed',
                        'data': {'cards': flashcards_parsed}
                    })
                    
                elif activity.type == 'Quiz':
                    # Generate quiz
                    num_questions = activity.data.get('num_questions', 10)
                    quiz_json = generate_quiz_from_text(limited_text, num_questions)
                    quiz_data = safe_load_json(quiz_json)
                    
                    if not quiz_data.get('questions'):
                        # Fallback quiz
                        quiz_data = {
                    "questions": [
                        {
                            "question": f"What is the main topic covered in {first_file_name}?",
                            "options": ["A) Introduction", "B) Main concepts", "C) Conclusion", "D) All of the above"],
                            "correct_answer": "D",
                            "explanation": "The document covers all aspects from introduction to conclusion."
                        },
                        {
                            "question": f"How would you best study {first_file_name}?",
                            "options": ["A) Read once", "B) Take notes and review", "C) Skip difficult parts", "D) Only read the summary"],
                            "correct_answer": "B",
                            "explanation": "Taking notes and reviewing helps with retention and understanding."
                        }
                    ]
                }
                
                    activity_ref.update({
                        'status': 'completed',
                        'data': quiz_data
                    })
                
                print(f"Completed {activity.type} activity: {activity.title}")
                
            except Exception as e:
                print(f"Error processing {activity.type} activity: {e}")
                activity_ref.update({'status': 'failed', 'error': str(e)})
        
        print(f"Completed processing revision pack for {first_file_name}")
        
    except Exception as e:
        print(f"Error in background processing: {e}")

@app.route("/hub/<hub_id>/get_activities_status")
@login_required
def get_activities_status(hub_id):
    """Get status of all activities for task queue updates."""
    try:
        # Get all activities for this hub
        activities_query = db.collection('activities').where(filter=firestore.FieldFilter('hub_id', '==', hub_id)).stream()
        activities = []
        
        for doc in activities_query:
            activity_data = doc.to_dict()
            # Only include pending/processing activities
            if activity_data.get('status') in ['pending', 'processing', 'completed']:
                activities.append({
                    'id': activity_data.get('id'),
                    'status': activity_data.get('status'),
                    'type': activity_data.get('type'),
                    'title': activity_data.get('title')
                })
        
        return jsonify({"success": True, "activities": activities})

    except Exception as e:
        print(f"Error getting activities status: {e}")
        return jsonify({"success": False, "message": str(e)}), 500



@app.route("/hub/<hub_id>/delete_activity/<activity_id>", methods=["POST"])
@login_required
def delete_activity(hub_id, activity_id):
    """Delete an activity (flashcards, quiz, etc.)."""
    try:
        # Verify hub ownership
        hub_doc = db.collection('hubs').document(hub_id).get()
        if not hub_doc.exists:
            return jsonify({"success": False, "message": "Hub not found"}), 404
        
        hub = Hub.from_dict(hub_doc.to_dict())
        if hub.user_id != current_user.id:
            return jsonify({"success": False, "message": "Permission denied"}), 403
        
        # Delete the activity
        activity_doc = db.collection('activities').document(activity_id).get()
        if not activity_doc.exists:
            return jsonify({"success": False, "message": "Activity not found"}), 404
        
        activity_doc.reference.delete()
        
        return jsonify({"success": True, "message": "Activity deleted successfully"})
    
    except Exception as e:
        print(f"Error deleting activity: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/hub/<hub_id>/delete_slide_note/<slide_id>", methods=["POST"])
@login_required
def delete_slide_note(hub_id, slide_id):
    """Delete a slide note (lecture notes)."""
    try:
        # Verify hub ownership
        hub_doc = db.collection('hubs').document(hub_id).get()
        if not hub_doc.exists:
            return jsonify({"success": False, "message": "Hub not found"}), 404
        
        hub = Hub.from_dict(hub_doc.to_dict())
        if hub.user_id != current_user.id:
            return jsonify({"success": False, "message": "Permission denied"}), 403
        
        # Delete the slide note
        slide_doc = db.collection('annotated_slide_decks').document(slide_id).get()
        if not slide_doc.exists:
            return jsonify({"success": False, "message": "Slide note not found"}), 404
        
        slide_doc.reference.delete()
        
        return jsonify({"success": True, "message": "Slide note deleted successfully"})
    
    except Exception as e:
        print(f"Error deleting slide note: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/hub/<hub_id>/generate_individual_notes", methods=["POST"])
@login_required
def generate_individual_notes(hub_id):
    """Generate individual interactive notes from selected files."""
    data = request.get_json()
    selected_files = data.get('selected_files', [])

    if not selected_files:
        return jsonify({"success": False, "message": "No files were selected."}), 400

    hub_text = get_text_from_hub_files(selected_files)
    if not hub_text:
        return jsonify({"success": False, "message": "Could not extract text from files."}), 500

    try:
        # Limit text length to prevent timeouts
        limited_text = hub_text[:8000] if len(hub_text) > 8000 else hub_text
        print(f"Generating individual notes with {len(limited_text)} characters of text")

        # Generate interactive notes
        interactive_html = generate_interactive_notes_html(limited_text)
        
        # Create note in database
        note_ref = db.collection('notes').document()
        first_file_name = os.path.basename(selected_files[0]).replace('.pdf', '')
        new_note = Note(id=note_ref.id, hub_id=hub_id, title=f"Interactive Notes: {first_file_name}", content_html=interactive_html)
        note_ref.set(new_note.to_dict())
        
        print("Individual notes generated successfully")
        return jsonify({"success": True, "redirect_url": url_for('view_note', note_id=note_ref.id)})

    except Exception as e:
        print(f"Error generating individual notes: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/hub/<hub_id>/generate_individual_flashcards", methods=["POST"])
@login_required
def generate_individual_flashcards(hub_id):
    """Generate individual flashcards from selected files."""
    data = request.get_json()
    selected_files = data.get('selected_files', [])
    num_flashcards = int(data.get('num_flashcards', 20))

    if not selected_files:
        return jsonify({"success": False, "message": "No files were selected."}), 400

    hub_text = get_text_from_hub_files(selected_files)
    if not hub_text:
        return jsonify({"success": False, "message": "Could not extract text from files."}), 500

    try:
        # Limit text length to prevent timeouts
        limited_text = hub_text[:8000] if len(hub_text) > 8000 else hub_text
        print(f"Generating individual flashcards with {len(limited_text)} characters of text")

        # Generate flashcards
        flashcards_raw = generate_flashcards_from_text(limited_text, num_flashcards)
        flashcards_parsed = parse_flashcards(flashcards_raw)
        
        if not flashcards_parsed:
            # Fallback flashcards
            first_file_name = os.path.basename(selected_files[0]).replace('.pdf', '')
            flashcards_parsed = [
                {"front": f"What is the main topic of {first_file_name}?", "back": "Review the document to understand the key concepts."},
                {"front": f"List 3 key points from {first_file_name}", "back": "Identify the most important information from the document."},
                {"front": f"How would you summarize {first_file_name}?", "back": "Create a brief overview of the main ideas and concepts."}
            ]
        
        # Create flashcards in database
        fc_ref = db.collection('activities').document()
        first_file_name = os.path.basename(selected_files[0]).replace('.pdf', '')
        new_fc = Activity(id=fc_ref.id, hub_id=hub_id, type='Flashcards', title=f"Flashcards: {first_file_name}", data={'cards': flashcards_parsed}, status='completed')
        fc_ref.set(new_fc.to_dict())
        
        # Automatically migrate flashcards to spaced repetition system
        try:
            migrate_flashcards_to_spaced_repetition(fc_ref.id, flashcards_parsed)
        except Exception as e:
            print(f"Warning: Failed to migrate flashcards to spaced repetition: {e}")
            # Continue anyway - flashcards are still created in activities
        
        print("Individual flashcards generated successfully")
        return jsonify({"success": True, "redirect_url": url_for('view_flashcards', activity_id=fc_ref.id)})

    except Exception as e:
        print(f"Error generating individual flashcards: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/hub/<hub_id>/generate_individual_quiz", methods=["POST"])
@login_required
def generate_individual_quiz(hub_id):
    """Generate individual quiz from selected files."""
    data = request.get_json()
    selected_files = data.get('selected_files', [])
    num_questions = int(data.get('num_questions', 10))

    if not selected_files:
        return jsonify({"success": False, "message": "No files were selected."}), 400

    hub_text = get_text_from_hub_files(selected_files)
    if not hub_text:
        return jsonify({"success": False, "message": "Could not extract text from files."}), 500

    try:
        # Limit text length to prevent timeouts
        limited_text = hub_text[:8000] if len(hub_text) > 8000 else hub_text
        print(f"Generating individual quiz with {len(limited_text)} characters of text")

        # Generate quiz
        quiz_json = generate_quiz_from_text(limited_text, num_questions)
        quiz_data = safe_load_json(quiz_json)
        
        if not quiz_data.get('questions'):
            # Fallback quiz
            first_file_name = os.path.basename(selected_files[0]).replace('.pdf', '')
            quiz_data = {
                "questions": [
                    {
                        "question": f"What is the main topic covered in {first_file_name}?",
                        "options": ["A) Introduction", "B) Main concepts", "C) Conclusion", "D) All of the above"],
                        "correct_answer": "D",
                        "explanation": "The document covers all aspects from introduction to conclusion."
                    },
                    {
                        "question": f"How would you best study {first_file_name}?",
                        "options": ["A) Read once", "B) Take notes and review", "C) Skip difficult parts", "D) Only read the summary"],
                        "correct_answer": "B",
                        "explanation": "Taking notes and reviewing helps with retention and understanding."
                    }
                ]
            }
        
        # Create quiz in database
        quiz_ref = db.collection('activities').document()
        first_file_name = os.path.basename(selected_files[0]).replace('.pdf', '')
        new_quiz = Activity(id=quiz_ref.id, hub_id=hub_id, type='Quiz', title=f"Practice Quiz: {first_file_name}", data=quiz_data)
        quiz_ref.set(new_quiz.to_dict())
        
        print("Individual quiz generated successfully")
        return jsonify({"success": True, "redirect_url": url_for('take_lecture_quiz', activity_id=quiz_ref.id)})

    except Exception as e:
        print(f"Error generating individual quiz: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/hub/<hub_id>/generate_individual_cheatsheet", methods=["POST"])
@login_required
def generate_individual_cheatsheet(hub_id):
    """Generate individual cheat sheet from selected files."""
    data = request.get_json()
    selected_files = data.get('selected_files', [])

    if not selected_files:
        return jsonify({"success": False, "message": "No files were selected."}), 400

    hub_text = get_text_from_hub_files(selected_files)
    if not hub_text:
        return jsonify({"success": False, "message": "Could not extract text from files."}), 500

    try:
        # Limit text length to prevent timeouts
        limited_text = hub_text[:8000] if len(hub_text) > 8000 else hub_text
        print(f"Generating individual cheat sheet with {len(limited_text)} characters of text")

        # Generate cheat sheet
        cheat_sheet_json = generate_cheat_sheet_json(limited_text)
        cheat_sheet_data = safe_load_json(cheat_sheet_json)
        
        if not cheat_sheet_data:
            # Fallback cheat sheet with correct JSON structure
            first_file_name = os.path.basename(selected_files[0]).replace('.pdf', '')
            cheat_sheet_data = {
                "title": f"Quick Reference: {first_file_name}",
                "columns": [
                    {
                        "blocks": [
                            {
                                "title": "Key Concepts",
                                "content_html": "<p>Review the document for main concepts, identify important definitions, and note key formulas or equations.</p>"
                            }
                        ]
                    },
                    {
                        "blocks": [
                            {
                                "title": "Important Points", 
                                "content_html": "<p>Focus on highlighted sections, review examples and case studies, and understand the main arguments.</p>"
                            }
                        ]
                    },
                    {
                        "blocks": [
                            {
                                "title": "Study Tips",
                                "content_html": "<p>Create flashcards for key terms, practice with examples, and review regularly for retention.</p>"
                            }
                        ]
                    }
                ]
            }
            # Convert to JSON string for storage
            cheat_sheet_json = json.dumps(cheat_sheet_data)
        
        # Create cheat sheet as a note
        note_ref = db.collection('notes').document()
        first_file_name = os.path.basename(selected_files[0]).replace('.pdf', '')
        
        # Store the JSON directly (same as exam pack version)
        new_note = Note(id=note_ref.id, hub_id=hub_id, title=f"Cheat Sheet for {first_file_name}", content_html=cheat_sheet_json)
        note_ref.set(new_note.to_dict())
        
        print("Individual cheat sheet generated successfully")
        return jsonify({"success": True, "redirect_url": url_for('view_note', note_id=note_ref.id)})

    except Exception as e:
        print(f"Error generating individual cheat sheet: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/hub/<hub_id>/build_exam_kit", methods=["POST"])
@login_required
def build_exam_kit(hub_id):
    if current_user.subscription_tier not in ['pro', 'admin']:
        return jsonify({"success": False, "message": "Exam Readiness Kits are a Pro feature."}), 403

    data = request.get_json()
    lecture_files = data.get('lecture_files', [])
    past_paper_files = data.get('past_paper_files', [])
    
    if not lecture_files:
        return jsonify({"success": False, "message": "Please select at least one lecture document."}), 400

    hub_text = get_text_from_hub_files(lecture_files)
    past_papers_text = get_text_from_hub_files(past_paper_files) if past_paper_files else None

    try:
        first_file_name = os.path.basename(lecture_files[0]).replace('.pdf', '')
        folder_name = f"Exam Kit for {first_file_name}"
        
        # Create activities with pending status for task queue
        activities_to_create = []
        
        # 1. Cheat Sheet
        cheat_sheet_ref = db.collection('activities').document()
        new_cheat_sheet_activity = Activity(
            id=cheat_sheet_ref.id, 
            hub_id=hub_id, 
            type='Notes', 
            title=f"Cheat Sheet for {first_file_name}", 
            status='pending',
            data={'lecture_files': lecture_files, 'text_length': len(hub_text), 'type': 'cheat_sheet'}
        )
        activities_to_create.append((cheat_sheet_ref, new_cheat_sheet_activity))
        
        # 2. Mock Exam
        mock_exam_ref = db.collection('activities').document()
        new_mock_exam_activity = Activity(
            id=mock_exam_ref.id, 
            hub_id=hub_id, 
            type='Quiz', 
            title=f"Mock Exam for {first_file_name}", 
            status='pending',
            data={'lecture_files': lecture_files, 'text_length': len(hub_text), 'num_questions': 15, 'type': 'mock_exam'}
        )
        activities_to_create.append((mock_exam_ref, new_mock_exam_activity))
        
        # 3. Exam Analysis (if past papers provided)
        if past_papers_text:
            analysis_ref = db.collection('activities').document()
            new_analysis_activity = Activity(
                id=analysis_ref.id, 
                hub_id=hub_id, 
                type='Notes', 
                title=f"Exam Analysis for {first_file_name}", 
                status='pending',
                data={'past_paper_files': past_paper_files, 'text_length': len(past_papers_text), 'type': 'exam_analysis'}
            )
            activities_to_create.append((analysis_ref, new_analysis_activity))
        
        # Create all activities with pending status
        batch = db.batch()
        folder_items = []
        
        for activity_ref, activity in activities_to_create:
            batch.set(activity_ref, activity.to_dict())
            # Map activity types to folder item types
            item_type = 'notes' if activity.type == 'Notes' else activity.type.lower()
            folder_items.append({'id': activity_ref.id, 'type': item_type})
        
        # Create the folder
        folder_ref = db.collection('folders').document()
        new_folder = Folder(id=folder_ref.id, hub_id=hub_id, name=folder_name, items=folder_items)
        batch.set(folder_ref, new_folder.to_dict())

        batch.commit()
        
        # Start background processing
        import threading
        threading.Thread(target=process_exam_kit_activities, args=(activities_to_create, hub_text, past_papers_text, first_file_name)).start()
        
        return jsonify({"success": True, "message": f"Exam Kit '{folder_name}' is being generated. Check the task queue for progress."})

    except Exception as e:
        print(f"Error in build_exam_kit: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

def process_exam_kit_activities(activities_to_create, hub_text, past_papers_text, first_file_name):
    """Background processing function for exam kit activities."""
    try:
        # Limit text length to prevent timeouts
        limited_text = hub_text[:8000] if len(hub_text) > 8000 else hub_text
        
        for activity_ref, activity in activities_to_create:
            try:
                print(f"Processing {activity.type} activity: {activity.title}")
                
                # Update status to processing
                activity_ref.update({'status': 'processing'})
                
                activity_type = activity.data.get('type')
                
                if activity_type == 'cheat_sheet':
                    # Generate cheat sheet
                    cheat_sheet_json_str = generate_cheat_sheet_json(limited_text)
                    
                    # Update activity with the generated cheat sheet content
                    activity_ref.update({
                        'status': 'completed',
                        'data': {'content_html': cheat_sheet_json_str, 'type': 'note'}
                    })
                    
                elif activity_type == 'mock_exam':
                    # Generate mock exam
                    num_questions = activity.data.get('num_questions', 15)
                    mock_exam_json = generate_quiz_from_text(limited_text, num_questions)
                    mock_exam_data = safe_load_json(mock_exam_json)
                    
                    if not mock_exam_data.get('questions'):
                        # Fallback mock exam
                        mock_exam_data = {
                            "questions": [
                                {
                                    "question": f"What is the main topic covered in {first_file_name}?",
                                    "options": ["A) Introduction", "B) Main concepts", "C) Conclusion", "D) All of the above"],
                                    "correct_answer": "D",
                                    "explanation": "The document covers all aspects from introduction to conclusion."
                                },
                                {
                                    "question": f"How would you best prepare for an exam on {first_file_name}?",
                                    "options": ["A) Cram the night before", "B) Study regularly and practice", "C) Skip difficult topics", "D) Only read summaries"],
                                    "correct_answer": "B",
                                    "explanation": "Regular study and practice leads to better retention and understanding."
                                }
                            ]
                        }
                    
                    activity_ref.update({
                        'status': 'completed',
                        'data': mock_exam_data
                    })
                    
                elif activity_type == 'exam_analysis':
                    # Generate exam analysis
                    analysis_markdown = analyse_papers_with_ai(past_papers_text)
                    analysis_html = markdown.markdown(analysis_markdown)
                    
                    # Update activity with the generated analysis content
                    activity_ref.update({
                        'status': 'completed',
                        'data': {'content_html': analysis_html, 'type': 'note'}
                    })
                
                print(f"Completed {activity.type} activity: {activity.title}")
                
            except Exception as e:
                print(f"Error processing {activity.type} activity: {e}")
                activity_ref.update({'status': 'failed', 'error': str(e)})
        
        print(f"Completed processing exam kit for {first_file_name}")
        
    except Exception as e:
        print(f"Error in background processing: {e}")

@app.route("/hub/<hub_id>/create_study_plan", methods=["POST"])
@login_required
def create_study_plan(hub_id):
    if current_user.subscription_tier not in ['pro', 'admin']:
        return jsonify({"success": False, "message": "The Smart Study Planner is a Pro feature."}), 403

    data = request.get_json()
    syllabus_files = data.get('syllabus_files', [])
    deadline = data.get('deadline')

    if not syllabus_files or not deadline:
        return jsonify({"success": False, "message": "Syllabus file and deadline are required."}), 400

    syllabus_text = get_text_from_hub_files(syllabus_files)
    if not syllabus_text:
        return jsonify({"success": False, "message": "Could not extract text from the syllabus."}), 500

    try:
        plan_json_str = generate_study_plan_from_syllabus(syllabus_text, deadline)
        plan_data = safe_load_json(plan_json_str)
        plan_weeks = plan_data.get('plan', [])

        if not plan_weeks:
            return jsonify({"success": False, "message": "AI could not generate a study plan from the provided syllabus."}), 500
        
        start_date = datetime.now(timezone.utc)
        deadline_date = datetime.fromisoformat(deadline).replace(tzinfo=timezone.utc)
        
        batch = db.batch()
        
        for week_plan in plan_weeks:
            # Simple scheduling: place event in the middle of the week
            event_date = start_date + timedelta(weeks=week_plan.get('week_number', 1) - 1, days=3)
            if event_date > deadline_date:
                continue

            event_ref = db.collection('calendar_events').document()
            new_event = CalendarEvent(
                id=event_ref.id,
                hub_id=hub_id,
                title=f"Study: {week_plan.get('topic', 'Weekly Topic')}",
                event_type="Study Session",
                start_time=event_date.replace(hour=10, minute=0, second=0), # Schedule for 10 AM
                end_time=event_date.replace(hour=11, minute=0, second=0), # 1 hour duration
                focus="\n".join(week_plan.get('tasks', [])),
                source_files=syllabus_files
            )
            batch.set(event_ref, new_event.to_dict())

        notification_ref = db.collection('notifications').document()
        message = "Your new smart study plan has been added to your calendar."
        link = url_for('hub_page', hub_id=hub_id, _anchor='calendar')
        new_notification = Notification(id=notification_ref.id, hub_id=hub_id, message=message, link=link)
        batch.set(notification_ref, new_notification.to_dict())

        batch.commit()
        return jsonify({"success": True, "message": "Study plan successfully created and added to your calendar!"})

    except Exception as e:
        print(f"Error in create_study_plan: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/hub/<hub_id>/run_async_tool", methods=["POST"])
@login_required
def run_async_tool(hub_id):
    data = request.get_json()
    tool = data.get('tool')
    selected_files = data.get('selected_files')

    if current_user.subscription_tier == 'free' and tool not in ['notes', 'flashcards']:
        return jsonify({"success": False, "message": f"The '{tool.capitalize()}' tool is a Pro feature."}), 403
    
    if not all([tool, selected_files]):
        return jsonify({"success": False, "message": "Missing tool or selected files."}), 400

    hub_text = get_text_from_hub_files(selected_files)
    if not hub_text:
        return jsonify({"success": False, "message": "Could not extract text from files."}), 500

    try:
        first_file_name = os.path.basename(selected_files[0]).replace('.pdf', '')
        
        if tool == 'notes':
            interactive_html = generate_interactive_notes_html(hub_text)
            note_ref = db.collection('notes').document()
            new_note = Note(id=note_ref.id, hub_id=hub_id, title=f"Quick Notes for {first_file_name}", content_html=interactive_html)
            note_ref.set(new_note.to_dict())
            redirect_url = url_for('view_note', note_id=note_ref.id)

        elif tool == 'flashcards':
            flashcards_raw = generate_flashcards_from_text(hub_text)
            flashcards_parsed = parse_flashcards(flashcards_raw)
            fc_ref = db.collection('activities').document()
            new_fc = Activity(id=fc_ref.id, hub_id=hub_id, type='Flashcards', title=f"Quick Flashcards for {first_file_name}", data={'cards': flashcards_parsed}, status='completed')
            fc_ref.set(new_fc.to_dict())
            redirect_url = url_for('edit_flashcard_set', activity_id=fc_ref.id)
            
        elif tool == 'quiz':
            quiz_json = generate_quiz_from_text(hub_text)
            quiz_data = safe_load_json(quiz_json)
            quiz_ref = db.collection('activities').document()
            new_quiz = Activity(id=quiz_ref.id, hub_id=hub_id, type='Quiz', title=f"Practice Quiz for {first_file_name}", data=quiz_data)
            quiz_ref.set(new_quiz.to_dict())
            redirect_url = url_for('take_lecture_quiz', activity_id=quiz_ref.id)

        elif tool == 'cheatsheet':
            cheat_sheet_json_str = generate_cheat_sheet_json(hub_text)
            note_ref = db.collection('notes').document()
            new_note = Note(id=note_ref.id, hub_id=hub_id, title=f"Cheat Sheet for {first_file_name}", content_html=cheat_sheet_json_str)
            note_ref.set(new_note.to_dict())
            redirect_url = url_for('view_note', note_id=note_ref.id)
            
        elif tool == 'analyse':
            analysis_markdown = analyse_papers_with_ai(hub_text)
            analysis_html = markdown.markdown(analysis_markdown)
            note_ref = db.collection('notes').document()
            new_note = Note(id=note_ref.id, hub_id=hub_id, title=f"Analysis of {first_file_name}", content_html=analysis_html)
            note_ref.set(new_note.to_dict())
            redirect_url = url_for('view_note', note_id=note_ref.id)

        else:
            return jsonify({"success": False, "message": f"Unknown tool: {tool}"}), 400

        return jsonify({"success": True, "redirect_url": redirect_url})

    except Exception as e:
        print(f"Error in run_async_tool for tool '{tool}': {e}")
        return jsonify({"success": False, "message": str(e)}), 500

# ==============================================================================
# 7. ASSIGNMENT WRITER ROUTES (NEW MAJOR FEATURE)
# ==============================================================================

def parse_brief_requirements_with_ai(brief_text, rubric_text):
    """Analyzes the brief and rubric to extract key assignment constraints."""
    prompt = f"""
    You are an expert academic assistant. Analyze the provided assignment brief and marking rubric to extract key requirements.
    Your response MUST be a single, valid JSON object with the following keys: "word_count", "sources_required", "prompt_type", "key_themes".
    - "word_count": An integer (e.g., 2500).
    - "sources_required": An integer representing the minimum number of sources.
    - "prompt_type": A short description (e.g., "Compare and contrast", "Critical analysis", "Case study").
    - "key_themes": An array of 3-5 strings representing the core topics to be covered.

    Assignment Brief:
    ---
    {brief_text}
    ---

    Marking Rubric (Optional):
    ---
    {rubric_text}
    ---
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content

def generate_assignment_outline_with_ai(assignment_details, context_text):
    """Generates a structured outline and evidence plan using RAG."""
    # This is a simplified RAG implementation for the response
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    documents = text_splitter.create_documents([context_text])
    vector_store = FAISS.from_documents(documents, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    # Retrieve relevant docs for the overall theme
    relevant_docs = retriever.get_relevant_documents(assignment_details['brief_text'])
    context_summary = "\n---\n".join([doc.page_content for doc in relevant_docs])

    prompt = f"""
    You are an expert academic planner creating a detailed essay outline.
    Based on the assignment brief and the provided source material context, generate a hierarchical outline as a JSON object.

    **Assignment Details:**
    - Title: {assignment_details['title']}
    - Brief: {assignment_details['brief_text']}
    - Word Count: {assignment_details['word_count_target']}
    - Referencing: {assignment_details['referencing_style']}
    - Voice: {assignment_details['voice']}

    **JSON Structure Rules:**
    1.  The root object must have two keys: "thesis_statement" (a string) and "sections" (an array of section objects).
    2.  Each section object MUST contain:
        - "title": A string for the section heading (e.g., "Introduction", "The Role of Fiscal Policy").
        - "word_count": An integer, a suggested word count for this section.
        - "plan": An array of strings, where each string is a key point or argument to be made in that section.
        - "evidence_suggestions": An array of strings, where each string is a brief note on what kind of evidence from the source material would be useful here (e.g., "Cite definitions from Document A", "Use statistics from Document B regarding inflation").

    **CRITICAL:** The total of all section word counts should roughly equal the target word count. Base all 'evidence_suggestions' ONLY on the provided context below.

    **Extracted Context from Source Materials:**
    ---
    {context_summary}
    ---
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content

def draft_assignment_section_with_ai(assignment_details, section_plan, context_text):
    """Drafts a single section of the assignment using RAG, incorporating human-like writing."""
    # This is a simplified RAG implementation for the response
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    documents = text_splitter.create_documents([context_text])
    vector_store = FAISS.from_documents(documents, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    # Retrieve relevant docs for the specific section
    query = f"{section_plan['title']}: {', '.join(section_plan['plan'])}"
    relevant_docs = retriever.get_relevant_documents(query)
    context_for_section = "\n---\n".join([doc.page_content for doc in relevant_docs])

    prompt = f"""
    You are an expert academic writer tasked with drafting a single essay section.
    
    **Overall Assignment Brief:** {assignment_details['brief_text']}
    **Target Voice & Style:** Write in a "{assignment_details['voice']}" voice, using {assignment_details['referencing_style']} for any citations.
    **Originality Level:** The user set an originality slider to "{assignment_details['originality_level']}".
        - If 'More Original': Rely heavily on synthesis and critical analysis, using sources mainly for support.
        - If 'More Evidenced': Use more direct quotes and close paraphrasing, ensuring every key point is backed by a citation.

    **Current Section to Draft:**
    - Title: "{section_plan['title']}"
    - Plan: {', '.join(section_plan['plan'])}
    - Target Word Count: {section_plan['word_count']} words

    **Instructions:**
    1.  Write a coherent, well-structured draft for this section only.
    2.  The draft must be approximately {section_plan['word_count']} words.
    3.  Strictly adhere to the plan and use the provided source context below to support all claims.
    4.  Where you use information from the context, insert an inline citation placeholder like `[CITATION: summary of source info]`. For example: `...which led to a significant economic downturn [CITATION: Document A statistics on GDP decline].`
    5.  Produce human-like prose. Vary sentence length, use appropriate discourse markers (e.g., 'however', 'consequently'), and maintain a formal academic tone.

    **Source Context for this Section:**
    ---
    {context_for_section}
    ---
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


@app.route("/hub/<hub_id>/start_assignment")
@login_required
def start_assignment(hub_id):
    if current_user.subscription_tier not in ['pro', 'admin']:
        flash("The Assignment Writer is a Pro feature. Please upgrade to access.", "warning")
        return redirect(url_for('hub_page', hub_id=hub_id))
    hub_doc = db.collection('hubs').document(hub_id).get()
    if not hub_doc.exists:
        return "Hub not found", 404
    hub = Hub.from_dict(hub_doc.to_dict())
    return render_template("create_assignment.html", hub=hub)

@app.route("/hub/<hub_id>/generate_assignment_plan", methods=["POST"])
@login_required
def generate_assignment_plan(hub_id):
    if current_user.subscription_tier not in ['pro', 'admin']:
        flash("The Assignment Writer is a Pro feature.", "warning")
        return redirect(url_for('hub_page', hub_id=hub_id))

    form = request.form
    
    # --- 1. Create and save the initial Assignment object ---
    assignment_ref = db.collection('assignments').document()
    new_assignment = Assignment(
        id=assignment_ref.id,
        hub_id=hub_id,
        title=form.get('title'),
        module=form.get('module'),
        word_count_target=form.get('word_count_target'),
        due_date=form.get('due_date'),
        referencing_style=form.get('referencing_style'),
        voice=form.get('voice'),
        originality_level=form.get('originality_level'),
        brief_text=form.get('brief_text'),
        rubric_text=form.get('rubric_text'),
        source_files=form.getlist('selected_files'),
        cite_only_uploaded='cite_only_uploaded' in form
    )
    assignment_ref.set(new_assignment.to_dict())

    try:
        # --- 2. Run AI to parse requirements and generate outline ---
        hub_text = get_text_from_hub_files(new_assignment.source_files)
        if not hub_text and new_assignment.cite_only_uploaded:
            assignment_ref.update({'status': 'error', 'parsed_requirements': {'error': 'No text could be extracted from source files.'}})
            return redirect(url_for('assignment_workspace', assignment_id=new_assignment.id))

        # requirements_json_str = parse_brief_requirements_with_ai(new_assignment.brief_text, new_assignment.rubric_text)
        # requirements_data = safe_load_json(requirements_json_str)
        
        outline_json_str = generate_assignment_outline_with_ai(new_assignment.to_dict(), hub_text)
        outline_data = safe_load_json(outline_json_str)

        # --- 3. Update the Assignment object with the results ---
        assignment_ref.update({
            # "parsed_requirements": requirements_data,
            "outline": outline_data,
            "status": 'outline_ready'
        })

        return redirect(url_for('assignment_workspace', assignment_id=new_assignment.id))

    except Exception as e:
        print(f"Error in generate_assignment_plan: {e}")
        assignment_ref.update({'status': 'error', 'outline': {'error': str(e)}})
        return redirect(url_for('assignment_workspace', assignment_id=new_assignment.id))

@app.route("/assignment/<assignment_id>")
@login_required
def assignment_workspace(assignment_id):
    assignment_doc = db.collection('assignments').document(assignment_id).get()
    if not assignment_doc.exists:
        return "Assignment not found", 404
    assignment = Assignment.from_dict(assignment_doc.to_dict())
    return render_template("assignment_workspace.html", assignment=assignment)

@app.route("/assignment/<assignment_id>/draft_section", methods=["POST"])
@login_required
def draft_section(assignment_id):
    assignment_doc = db.collection('assignments').document(assignment_id).get()
    if not assignment_doc.exists:
        return jsonify({"success": False, "message": "Assignment not found."}), 404
    
    assignment = Assignment.from_dict(assignment_doc.to_dict())
    section_index = request.json.get('section_index')
    
    if section_index is None or section_index >= len(assignment.outline.get('sections', [])):
        return jsonify({"success": False, "message": "Invalid section index."}), 400

    try:
        section_plan = assignment.outline['sections'][section_index]
        hub_text = get_text_from_hub_files(assignment.source_files)
        
        drafted_text = draft_assignment_section_with_ai(assignment.to_dict(), section_plan, hub_text)

        # Update the specific section in the draft_content map
        field_path = f'draft_content.section_{section_index}'
        db.collection('assignments').document(assignment_id).update({
            field_path: drafted_text
        })
        
        return jsonify({"success": True, "drafted_text": drafted_text})

    except Exception as e:
        print(f"Error drafting section {section_index} for assignment {assignment_id}: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

# ==============================================================================
# 8. FOLDER MANAGEMENT ROUTES
# ==============================================================================

@app.route("/hub/<hub_id>/create_folder", methods=["POST"])
@login_required
def create_folder(hub_id):
    folder_name = request.form.get('folder_name')
    if not folder_name:
        flash("Folder name cannot be empty.", "error")
        return redirect(url_for('hub_page', hub_id=hub_id))
    
    batch = db.batch()
    folder_ref = db.collection('folders').document()
    new_folder = Folder(id=folder_ref.id, hub_id=hub_id, name=folder_name)
    batch.set(folder_ref, new_folder.to_dict())
    
    # --- NEW: Create Notification ---
    notification_ref = db.collection('notifications').document()
    message = f"A new folder named '{folder_name}' was created."
    link = url_for('hub_page', hub_id=hub_id, _anchor='folders')
    new_notification = Notification(id=notification_ref.id, hub_id=hub_id, message=message, link=link)
    batch.set(notification_ref, new_notification.to_dict())
    batch.commit()

    flash(f"Folder '{folder_name}' created successfully!", "success")
    return redirect(url_for('hub_page', hub_id=hub_id, _anchor='folders'))


@app.route("/folder/<folder_id>")
@login_required
def view_folder(folder_id):
    folder_ref = db.collection('folders').document(folder_id)
    folder_doc = folder_ref.get()
    if not folder_doc.exists:
        return "Folder not found", 404
    folder = Folder.from_dict(folder_doc.to_dict())

    folder_items = []
    folder_item_ids = {item['id'] for item in folder.items}
    
    for item_ref in folder.items:
        doc_id = item_ref.get('id')
        doc_type = item_ref.get('type')
        
        if doc_type == 'note':
            doc = db.collection('notes').document(doc_id).get()
            if doc.exists: folder_items.append(Note.from_dict(doc.to_dict()))
        elif doc_type in ['quiz', 'flashcards']:
            doc = db.collection('activities').document(doc_id).get()
            if doc.exists: folder_items.append(Activity.from_dict(doc.to_dict()))

    all_notes = [n for n in db.collection('notes').where('hub_id', '==', folder.hub_id).stream() if n.id not in folder_item_ids]
    all_activities = [a for a in db.collection('activities').where('hub_id', '==', folder.hub_id).stream() if a.id not in folder_item_ids]
    
    available_assets = {
        "notes": [Note.from_dict(n.to_dict()) for n in all_notes],
        "quizzes": [Activity.from_dict(a.to_dict()) for a in all_activities if a.to_dict().get('type') == 'Quiz'],
        "flashcards": [Activity.from_dict(a.to_dict()) for a in all_activities if a.to_dict().get('type') == 'Flashcards']
    }
    
    return render_template("folder.html", folder=folder, folder_items=folder_items, available_assets=available_assets)

@app.route("/folder/<folder_id>/update_name", methods=["POST"])
@login_required
def update_folder_name(folder_id):
    new_name = request.json.get('new_name')
    if not new_name:
        return jsonify({"success": False, "message": "Name cannot be empty."}), 400
    try:
        db.collection('folders').document(folder_id).update({'name': new_name})
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/folder/<folder_id>/delete", methods=["POST"])
@login_required
def delete_folder(folder_id):
    try:
        # Get folder to verify ownership and find its items
        folder_doc = db.collection('folders').document(folder_id).get()
        if not folder_doc.exists:
            return jsonify({"success": False, "message": "Folder not found"}), 404
        
        folder = Folder.from_dict(folder_doc.to_dict())
        
        # Verify hub ownership
        hub_doc = db.collection('hubs').document(folder.hub_id).get()
        if not hub_doc.exists:
            return jsonify({"success": False, "message": "Hub not found"}), 404
        
        hub = Hub.from_dict(hub_doc.to_dict())
        if hub.user_id != current_user.id:
            return jsonify({"success": False, "message": "Permission denied"}), 403
        
        # Delete all items in the folder
        for item in folder.items:
            item_id = item.get('id')
            item_type = item.get('type')
            
            if item_type == 'note':
                # Delete from notes collection
                note_doc = db.collection('notes').document(item_id).get()
                if note_doc.exists:
                    note_doc.reference.delete()
            elif item_type in ['quiz', 'flashcards', 'notes']:
                # Delete from activities collection
                activity_doc = db.collection('activities').document(item_id).get()
                if activity_doc.exists:
                    activity_doc.reference.delete()
        
        # Delete the folder itself
        db.collection('folders').document(folder_id).delete()
        
        return jsonify({"success": True, "message": "Folder deleted successfully"})
    except Exception as e:
        print(f"Error deleting folder: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/folder/<folder_id>/add_items", methods=["POST"])
@login_required
def add_items_to_folder(folder_id):
    items_to_add = request.json.get('items', [])
    if not items_to_add:
        return jsonify({"success": False, "message": "No items to add."}), 400
    try:
        folder_ref = db.collection('folders').document(folder_id)
        folder_ref.update({'items': firestore.ArrayUnion(items_to_add)})
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/folder/<folder_id>/remove_item", methods=["POST"])
@login_required
def remove_item_from_folder(folder_id):
    item_to_remove = request.json.get('item')
    if not item_to_remove:
        return jsonify({"success": False, "message": "No item to remove."}), 400
    try:
        folder_ref = db.collection('folders').document(folder_id)
        folder_ref.update({'items': firestore.ArrayRemove([item_to_remove])})
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# ==============================================================================
# 9. LECTURE RECORDER ROUTES (MODIFIED)
# ==============================================================================
@app.route("/hub/<hub_id>/process_lecture_audio", methods=["POST"])
@login_required
def process_lecture_audio(hub_id):
    if 'audio_file' not in request.files:
        return jsonify({"success": False, "message": "No audio file found."}), 400

    audio_file = request.files['audio_file']
    title = request.form.get('title', f"Lecture Notes - {datetime.now().strftime('%Y-%m-%d')}")

    try:
        # Use AssemblyAI's much simpler file transcription API
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(audio_file)

        if transcript.status == aai.TranscriptStatus.error:
            return jsonify({"success": False, "message": transcript.error}), 500
        
        full_transcript_text = transcript.text
        if not full_transcript_text:
             return jsonify({"success": False, "message": "Audio could not be transcribed or was empty."}), 400

        # --- RE-USE YOUR EXISTING NOTE GENERATION LOGIC ---
        notes_html = generate_interactive_notes_html(full_transcript_text)

        # --- MODIFIED: Instead of saving, render a new review page ---
        return render_template(
            "lecture_review.html", 
            hub_id=hub_id,
            title=title,
            notes_html=notes_html,
            full_transcript_text=full_transcript_text
        )

    except Exception as e:
        print(f"Error processing lecture audio: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

# --- NEW: Route to save edited notes from the review page ---
@app.route("/hub/<hub_id>/save_lecture_notes", methods=["POST"])
@login_required
def save_lecture_notes(hub_id):
    title = request.form.get('title')
    notes_html = request.form.get('notes_html')

    if not title or not notes_html:
        flash("Title and notes content are required.", "error")
        return redirect(url_for('hub_page', hub_id=hub_id))

    try:
        # --- Save the note ---
        note_ref = db.collection('notes').document()
        new_note = Note(id=note_ref.id, hub_id=hub_id, title=title, content_html=notes_html)
        note_ref.set(new_note.to_dict())

        # --- Create a notification ---
        notification_ref = db.collection('notifications').document()
        message = f"Your lecture notes for '{title}' have been saved."
        link = url_for('view_note', note_id=note_ref.id)
        new_notification = Notification(id=notification_ref.id, hub_id=hub_id, message=message, link=link)
        notification_ref.set(new_notification.to_dict())
        
        # --- Update hub progress ---
        update_hub_progress(hub_id, 10) # 10 XP for a recorded lecture

        flash("Notes saved successfully!", "success")
        return redirect(url_for('hub_page', hub_id=hub_id, _anchor='notes'))

    except Exception as e:
        print(f"Error saving lecture notes: {e}")
        flash(f"An error occurred while saving your notes: {e}", "error")
        return redirect(url_for('hub_page', hub_id=hub_id))


# --- NEW: Route to create a full revision pack from the review page ---
@app.route("/hub/<hub_id>/create_revision_pack_from_lecture", methods=["POST"])
@login_required
def create_revision_pack_from_lecture(hub_id):
    title = request.form.get('title')
    notes_html = request.form.get('notes_html')
    # This is key for generating good flashcards/quizzes from the original source
    full_transcript_text = request.form.get('full_transcript_text')

    if not all([title, notes_html, full_transcript_text]):
        flash("Missing required data to create a revision pack.", "error")
        return redirect(url_for('hub_page', hub_id=hub_id))

    try:
        batch = db.batch()
        folder_items = []
        
        # --- 1. Save the edited notes ---
        note_ref = db.collection('notes').document()
        new_note = Note(id=note_ref.id, hub_id=hub_id, title=title, content_html=notes_html)
        batch.set(note_ref, new_note.to_dict())
        folder_items.append({'id': note_ref.id, 'type': 'note'})

        # --- 2. Generate flashcards from the original transcript ---
        flashcards_raw = generate_flashcards_from_text(full_transcript_text)
        flashcards_parsed = parse_flashcards(flashcards_raw)
        if flashcards_parsed:
            fc_ref = db.collection('activities').document()
            new_fc = Activity(
                id=fc_ref.id, 
                hub_id=hub_id, 
                type='Flashcards', 
                title=f"Flashcards for {title}", 
                data={'cards': flashcards_parsed}, 
                status='completed'
            )
            batch.set(fc_ref, new_fc.to_dict())
            folder_items.append({'id': fc_ref.id, 'type': 'flashcards'})

        # --- 3. Generate 3 quizzes from the original transcript ---
        for i in range(1, 4):
            quiz_json = generate_quiz_from_text(full_transcript_text)
            quiz_data = safe_load_json(quiz_json)
            if quiz_data.get('questions'):
                quiz_ref = db.collection('activities').document()
                new_quiz = Activity(
                    id=quiz_ref.id, 
                    hub_id=hub_id, 
                    type='Quiz', 
                    title=f"Practice Quiz {i} for {title}", 
                    data=quiz_data
                )
                batch.set(quiz_ref, new_quiz.to_dict())
                folder_items.append({'id': quiz_ref.id, 'type': 'quiz'})

        # --- 4. Create the folder ---
        folder_name = f"Revision Pack: {title}"
        folder_ref = db.collection('folders').document()
        new_folder = Folder(id=folder_ref.id, hub_id=hub_id, name=folder_name, items=folder_items)
        batch.set(folder_ref, new_folder.to_dict())
        
        # --- 5. Create a notification ---
        notification_ref = db.collection('notifications').document()
        message = f"Your new revision pack '{folder_name}' is ready."
        link = url_for('hub_page', hub_id=hub_id, _anchor='folders')
        new_notification = Notification(id=notification_ref.id, hub_id=hub_id, message=message, link=link)
        batch.set(notification_ref, new_notification.to_dict())
        
        # --- 6. Commit all changes to Firestore ---
        batch.commit()
        
        # --- 7. Update hub progress ---
        update_hub_progress(hub_id, 25) # Give more XP for creating a full pack

        flash(f"Successfully created revision pack '{folder_name}'!", "success")
        return redirect(url_for('hub_page', hub_id=hub_id, _anchor='folders'))

    except Exception as e:
        print(f"Error creating revision pack: {e}")
        flash(f"An error occurred while creating your revision pack: {e}", "error")
        return redirect(url_for('hub_page', hub_id=hub_id))

# ==============================================================================
# 10. COMMUNITY ROUTES (UPDATED)
# ==============================================================================
@app.route("/folder/share", methods=["POST"])
@login_required
def share_folder():
    if current_user.subscription_tier not in ['pro', 'admin']:
        flash("You must be a Pro member to share folders.", "error")
        return redirect(url_for('dashboard'))

    folder_id = request.form.get('folder_id')
    title = request.form.get('title') # NEW: Get editable title
    description = request.form.get('description')
    tags_raw = request.form.get('tags', '')
    
    if not all([folder_id, description, title]):
        flash("Folder, title, and description are required.", "error")
        return redirect(url_for('dashboard', _anchor='community'))

    try:
        folder_doc = db.collection('folders').document(folder_id).get()
        if not folder_doc.exists:
            flash("The selected folder could not be found.", "error")
            return redirect(url_for('dashboard', _anchor='community'))

        folder_data = folder_doc.to_dict()
        user_hubs = [h.id for h in _get_user_stats(current_user.id)['hubs']]
        if folder_data.get('hub_id') not in user_hubs:
             flash("You can only share folders that you own.", "error")
             return redirect(url_for('dashboard', _anchor='community'))

        tags = [tag.strip().lower() for tag in tags_raw.split(',') if tag.strip()]

        shared_folder_ref = db.collection('shared_folders').document()
        new_shared_folder = SharedFolder(
            id=shared_folder_ref.id,
            original_folder_id=folder_id,
            original_hub_id=folder_data.get('hub_id'),
            owner_id=current_user.id,
            title=title, # UPDATED: Use the new title from the form
            description=description,
            tags=tags
        )

        shared_folder_ref.set(new_shared_folder.to_dict())
        flash(f"Successfully shared '{new_shared_folder.title}' with the community!", "success")

    except Exception as e:
        print(f"Error sharing folder: {e}")
        flash("An unexpected error occurred while sharing.", "error")

    return redirect(url_for('dashboard', _anchor='community'))


@app.route("/folder/import/<shared_folder_id>", methods=["POST"])
@login_required
def import_folder(shared_folder_id):
    if current_user.subscription_tier not in ['pro', 'admin']:
        return jsonify({"success": False, "message": "You must be a Pro member to import folders."}), 403

    target_hub_id = request.json.get('hub_id')
    if not target_hub_id:
        return jsonify({"success": False, "message": "Target hub ID is required."}), 400

    try:
        shared_folder_ref = db.collection('shared_folders').document(shared_folder_id)
        shared_folder_doc = shared_folder_ref.get()
        if not shared_folder_doc.exists:
            return jsonify({"success": False, "message": "Shared folder not found."}), 404

        shared_folder = SharedFolder.from_dict(shared_folder_doc.to_dict())

        if shared_folder.owner_id == current_user.id:
            return jsonify({"success": False, "message": "You cannot import your own folder."}), 400

        original_folder_doc = db.collection('folders').document(shared_folder.original_folder_id).get()
        if not original_folder_doc.exists:
             return jsonify({"success": False, "message": "The original folder no longer exists."}), 404
        
        original_folder = Folder.from_dict(original_folder_doc.to_dict())
        
        batch = db.batch()
        new_items_for_folder = []

        # This logic correctly duplicates the items for the new user
        item_ids = [item['id'] for item in original_folder.items]
        if item_ids:
            notes_docs = db.collection('notes').where('id', 'in', item_ids).stream()
            activities_docs = db.collection('activities').where('id', 'in', item_ids).stream()
            
            # Process notes
            for doc in notes_docs:
                original_item_data = doc.to_dict()
                new_item_ref = db.collection('notes').document()
                new_item_data = original_item_data
                new_item_data['id'] = new_item_ref.id
                new_item_data['hub_id'] = target_hub_id
                batch.set(new_item_ref, new_item_data)
                new_items_for_folder.append({'id': new_item_ref.id, 'type': 'note'})

            # Process activities (quizzes, flashcards)
            for doc in activities_docs:
                original_item_data = doc.to_dict()
                new_item_ref = db.collection('activities').document()
                new_item_data = original_item_data
                new_item_data['id'] = new_item_ref.id
                new_item_data['hub_id'] = target_hub_id
                batch.set(new_item_ref, new_item_data)
                item_type = 'flashcards' if new_item_data.get('type') == 'Flashcards' else 'quiz'
                new_items_for_folder.append({'id': new_item_ref.id, 'type': item_type})

        new_folder_ref = db.collection('folders').document()
        new_folder = Folder(
            id=new_folder_ref.id,
            hub_id=target_hub_id,
            name=f"{shared_folder.title} (Imported)",
            items=new_items_for_folder
        )
        batch.set(new_folder_ref, new_folder.to_dict())

        # --- THIS IS THE KEY UPDATE ---
        # Increment import count AND add user to imported_by list
        batch.update(shared_folder_ref, {
            'imports': firestore.Increment(1),
            'imported_by': firestore.ArrayUnion([current_user.id])
        })
        # --- END OF UPDATE ---

        batch.commit()
        return jsonify({"success": True, "message": f"'{new_folder.name}' imported successfully!"})

    except Exception as e:
        print(f"Error importing folder: {e}")
        return jsonify({"success": False, "message": "An internal server error occurred."}), 500

# --- NEW: Route for Liking/Unliking a folder ---
@app.route("/folder/like/<shared_folder_id>", methods=["POST"])
@login_required
def like_folder(shared_folder_id):
    """Toggles a like on a shared folder for the current user."""
    shared_folder_ref = db.collection('shared_folders').document(shared_folder_id)
    shared_folder_doc = shared_folder_ref.get()

    if not shared_folder_doc.exists:
        return jsonify({"success": False, "message": "Folder not found."}), 404

    folder = SharedFolder.from_dict(shared_folder_doc.to_dict())

    # Users cannot like their own folders
    if folder.owner_id == current_user.id:
        return jsonify({"success": False, "message": "You cannot like your own folder."}), 403

    # Business Rule: User must have imported the folder to be able to like it.
    if current_user.id not in folder.imported_by:
        return jsonify({"success": False, "message": "You must import a folder before you can like it."}), 403

    user_has_liked = current_user.id in folder.liked_by

    try:
        if user_has_liked:
            # User is unliking the folder
            shared_folder_ref.update({
                'likes': firestore.Increment(-1),
                'liked_by': firestore.ArrayRemove([current_user.id])
            })
            new_like_count = folder.likes - 1
            return jsonify({"success": True, "action": "unliked", "count": new_like_count})
        else:
            # User is liking the folder
            shared_folder_ref.update({
                'likes': firestore.Increment(1),
                'liked_by': firestore.ArrayUnion([current_user.id])
            })
            new_like_count = folder.likes + 1
            return jsonify({"success": True, "action": "liked", "count": new_like_count})
    except Exception as e:
        print(f"Error liking folder {shared_folder_id}: {e}")
        return jsonify({"success": False, "message": "An internal error occurred."}), 500
    
# --- NEW: Route for deleting a shared folder ---
@app.route("/folder/share/delete/<shared_folder_id>", methods=["POST"])
@login_required
def delete_shared_folder(shared_folder_id):
    shared_folder_ref = db.collection('shared_folders').document(shared_folder_id)
    shared_folder_doc = shared_folder_ref.get()

    if not shared_folder_doc.exists:
        return jsonify({"success": False, "message": "Shared folder not found."}), 404

    folder_data = shared_folder_doc.to_dict()

    if folder_data.get('owner_id') != current_user.id:
        return jsonify({"success": False, "message": "You do not have permission to delete this folder."}), 403

    try:
        shared_folder_ref.delete()
        return jsonify({"success": True, "message": "Folder removed from community."})
    except Exception as e:
        return jsonify({"success": False, "message": f"An error occurred: {e}"}), 500

# ==============================================================================
# 11. AI TUTOR ROUTES (NEW)
# ==============================================================================
def get_or_create_vector_store(hub_id):
    """
    Retrieves a vector store from the cache or creates a new one 
    by processing all files in a given hub.
    """
    if hub_id in vector_store_cache:
        return vector_store_cache[hub_id]

    hub_doc = db.collection('hubs').document(hub_id).get()
    if not hub_doc.exists:
        return None
    
    hub = Hub.from_dict(hub_doc.to_dict())
    if not hub.files:
        return None
        
    all_file_paths = [f['path'] for f in hub.files]
    combined_text = get_text_from_hub_files(all_file_paths)

    if not combined_text:
        return None
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.create_documents([combined_text])
    
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store_cache[hub_id] = vector_store
    
    return vector_store

@app.route("/hub/<hub_id>/tutor_chat", methods=["POST"])
@login_required
def tutor_chat(hub_id):
    if current_user.subscription_tier not in ['pro', 'admin']:
        return jsonify({"answer": "The AI Tutor is a Pro feature. Please upgrade to chat."}), 403

    data = request.get_json()
    user_message = data.get('message')
    chat_history_json = data.get('history', [])
    
    chat_history = [HumanMessage(content=msg['content']) if msg['role'] == 'user' 
                    else AIMessage(content=msg['content']) 
                    for msg in chat_history_json]

    vector_store = get_or_create_vector_store(hub_id)
    if not vector_store:
        return jsonify({"answer": "I can't answer questions until you've uploaded at least one document to this hub."})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    retriever = vector_store.as_retriever()

    # 1. Create a prompt for rephrasing the question based on history
    rephrase_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, rephrase_prompt)

    # 2. Create the main prompt for answering the question
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert AI tutor. Answer the user's questions based on the provided context. Be concise and helpful. If you don't know the answer from the context, say so politely. Context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # 3. Combine them into a retrieval chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # 4. Invoke the chain
    response = rag_chain.invoke({
        "chat_history": chat_history,
        "input": user_message
    })

    return jsonify({"answer": response['answer']})

# ==============================================================================
# 12. STUCK ON A QUESTION TUTOR ROUTES (NEW FEATURE)
# ==============================================================================
import base64

def get_image_media_type(file_storage):
    """Determines the media type from the file extension for Vision API."""
    filename = file_storage.filename.lower()
    if filename.endswith('.png'):
        return 'image/png'
    elif filename.endswith(('.jpg', '.jpeg')):
        return 'image/jpeg'
    elif filename.endswith('.pdf'):
         # For simplicity, we'll handle the first page of a PDF as an image
         # In a real-world app, you might use a library like pdf2image to handle all pages
        return 'application/pdf' # GPT-4o can handle PDFs directly
    return None

@app.route("/hub/<hub_id>/stuck_on_question/start", methods=["POST"])
@login_required
def stuck_on_question_start(hub_id):
    """
    Handles the initial submission of a question, either as text or an image/PDF.
    Uses GPT-4o Vision to extract the question from the media.
    """
    question_text = request.form.get('question_text')
    question_file = request.files.get('question_file')

    if not question_text and not question_file:
        return jsonify({"success": False, "message": "Please provide a question as text or a file."}), 400

    try:
        if question_file:
            media_type = get_image_media_type(question_file)
            if not media_type:
                return jsonify({"success": False, "message": "Unsupported file type. Please use PNG, JPG, or PDF."}), 400

            encoded_file = base64.b64encode(question_file.read()).decode('utf-8')
            
            # Using GPT-4o's vision capabilities
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "You are an expert academic assistant. Your task is to analyze the provided image, which contains a question. Transcribe ALL text in the image and also describe any important visual elements like diagrams, graphs, or tables. Combine everything into a single, complete problem description in Markdown format. This full description will be used by another AI to solve the problem, so it must be comprehensive and accurate."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{encoded_file}"
                                }
                            }
                        ]
                    }
                ]
            )
            extracted_question = response.choices[0].message.content

        else: # It's a text submission
            extracted_question = question_text

        return jsonify({"success": True, "extracted_question": extracted_question})

    except Exception as e:
        print(f"Error in stuck_on_question_start: {e}")
        return jsonify({"success": False, "message": f"An error occurred while processing your question: {e}"}), 500

@app.route("/hub/<hub_id>/stuck_on_question/generate_solution", methods=["POST"])
@login_required
def stuck_on_question_generate_solution(hub_id):
    """Generates a step-by-step solution for a confirmed question."""
    question = request.json.get('question')
    if not question:
        return jsonify({"success": False, "message": "No question was provided."}), 400

    # THIS IS THE CRITICAL FIX: A much more detailed and demanding prompt.
    prompt = f"""
    You are an exceptionally patient and clear AI tutor, explaining a complex problem to a beginner who is feeling stuck. Your tone should be encouraging and simple.

    **CRITICAL TASK:** Your task is to solve the provided problem in a series of extremely granular, simple, step-by-step instructions.

    **RULES FOR STEP GENERATION (MUST FOLLOW):**
    1.  **One Action Per Step:** Each step in the output `steps` array must represent a single, small, logical action or thought. Do NOT combine multiple calculations or concepts into one step.
    2.  **Beginner-Friendly Language:** Use simple, direct language. Avoid jargon where possible, or explain it immediately in the next step.
    3.  **Conversational Flow:** Think of it as a conversation. After each small piece of information, the student needs to confirm they understand before you move on. Each confirmation point is a new step.
    4.  **Show, Don't Just Tell:** Start by identifying the relevant shapes, formulas, or concepts before you use them.

    **EXAMPLE of GOOD vs. BAD Steps for a geometry problem:**
    *   **BAD STEP (too complex):** "Using trigonometry in triangle OAB, we find the radius OA by calculating tan(30°) = OA/OB, so OA = 16 * tan(30°), which is 9.2."
    *   **GOOD STEPS (granular and simple):**
        *   "First, let's focus on the shape formed by points O, A, and B. This is a right-angled triangle because a tangent line (AB) always meets a radius (OA) at a perfect 90° angle."
        *   "In this triangle, we are given the angle at B is 30° and the length of the side next to it (OB) is 16."
        *   "We want to find the length of the side opposite the angle, which is OA. This is also the radius of our circle."
        *   "To connect an angle with its opposite and adjacent sides in a right-angled triangle, we use the tangent function: tan(angle) = Opposite / Adjacent."
        *   "Now, let's plug in the values we know: tan(30°) = OA / 16."
        *   "To get OA by itself, we can multiply both sides of the equation by 16. This gives us: OA = 16 * tan(30°)."
        *   "Calculating this gives us the length of the radius, OA. Let's do that next."

    **YOUR OUTPUT:**
    Your response MUST be a single, valid JSON object with one key: "steps". The "steps" value must be an array of simple, beginner-friendly strings, following all the rules above.

    **Here is the question to solve:**
    ---
    {question}
    ---
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        solution_data = json.loads(response.choices[0].message.content)
        # Add a check to ensure steps were actually generated
        if not solution_data.get("steps") or not isinstance(solution_data["steps"], list) or len(solution_data["steps"]) == 0:
            raise ValueError("AI returned a valid JSON but with no solution steps.")
            
        return jsonify({"success": True, "solution": solution_data})
    except Exception as e:
        print(f"Error generating solution: {e}")
        return jsonify({"success": False, "message": "The AI could not generate a solution for this question."}), 500


@app.route("/hub/<hub_id>/stuck_on_question/clarify_step", methods=["POST"])
@login_required
def stuck_on_question_clarify_step():
    """Provides a focused explanation for a specific step a user is stuck on."""
    data = request.json
    stuck_step = data.get('stuck_step')
    clarification_request = data.get('clarification_request')
    original_question = data.get('original_question')
    solution_context = data.get('solution_context') # The steps so far

    prompt = f"""
    You are a patient and helpful AI tutor. A student is stuck on a specific step of a problem.
    Your task is to explain ONLY the point of confusion. Do not move on to the next step.
    - **Original Question:** "{original_question}"
    - **The Solution Steps So Far:** {solution_context}
    - **The Step They Are Stuck On:** "{stuck_step}"
    - **Their Specific Question:** "{clarification_request}"

    Provide a clear, simple explanation in Markdown that directly addresses their question.
    Use analogies or simpler examples if helpful. Your explanation should be focused and concise.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        clarification_text = response.choices[0].message.content
        return jsonify({"success": True, "clarification": clarification_text})
    except Exception as e:
        print(f"Error clarifying step: {e}")
        return jsonify({"success": False, "message": "An error occurred while generating the explanation."}), 500

@app.route("/hub/<hub_id>/stuck_on_question/generate_practice", methods=["POST"])
@login_required
def stuck_on_question_generate_practice():
    """Generates a similar practice question based on the original problem."""
    original_question = request.json.get('question')
    solution_steps = request.json.get('solution_steps')
    
    prompt = f"""
    You are an expert question creator. Based on the following solved problem, create one new, similar practice question.
    The question should test the same core concept but use different numbers, scenarios, or wording.
    Your response MUST be a single, valid JSON object with one key: "practice_question". The value should be the question text in Markdown.

    **Original Solved Problem:**
    - Question: "{original_question}"
    - Solution: {solution_steps}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        practice_data = json.loads(response.choices[0].message.content)
        return jsonify({"success": True, "practice_question": practice_data.get("practice_question")})
    except Exception as e:
        print(f"Error generating practice question: {e}")
        return jsonify({"success": False, "message": "Could not generate a practice question."}), 500


@app.route("/hub/<hub_id>/stuck_on_question/evaluate_practice", methods=["POST"])
@login_required
def stuck_on_question_evaluate_practice():
    """Evaluates the user's answer to the practice question."""
    data = request.json
    practice_question = data.get('practice_question')
    student_answer = data.get('student_answer')
    original_question_context = data.get('original_question')

    prompt = f"""
    You are an expert AI evaluator. The student was given a practice question and has submitted an answer.
    - **Practice Question:** "{practice_question}"
    - **Student's Answer:** "{student_answer}"
    - **Context (Original Problem):** "{original_question_context}"

    Your task is to determine if the student's answer is correct. Your response MUST be a single, valid JSON object with two keys:
    1. 'is_correct' (boolean: true if their answer is correct, false otherwise).
    2. 'feedback' (string: A brief, encouraging message if correct, or a gentle correction like "Not quite. Let's walk through it." if incorrect).
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        evaluation_data = json.loads(response.choices[0].message.content)
        return jsonify({"success": True, "evaluation": evaluation_data})
    except Exception as e:
        print(f"Error evaluating practice answer: {e}")
        return jsonify({"success": False, "message": "Could not evaluate the answer."}), 500

# ==============================================================================
# 13. ONBOARDING ROUTES (NEW)
# ==============================================================================

@app.route('/onboarding/personalization', methods=['POST'])
@login_required
def save_personalization():
    """Saves user personalization data from onboarding."""
    try:
        data = request.get_json()
        user_ref = db.collection('users').document(current_user.id)
        
        # Map personalization data to user fields
        personalization_updates = {}
        
        if 'slide1' in data:
            goal_mapping = {
                'exam-prep': 'Exam preparation and test-taking',
                'understanding': 'Deep understanding of concepts',
                'efficiency': 'Study efficiency and time optimization'
            }
            personalization_updates['goals'] = goal_mapping.get(data['slide1'], data['slide1'])
        
        if 'slide2' in data:
            style_mapping = {
                'visual': 'Visual learning with diagrams and charts',
                'textual': 'Text-based learning with notes and summaries',
                'interactive': 'Interactive learning with quizzes and games'
            }
            personalization_updates['study_style'] = style_mapping.get(data['slide2'], data['slide2'])
        
        if 'slide3' in data:
            frequency_mapping = {
                'daily': 'Daily study sessions',
                'weekly': 'Weekly study sessions',
                'intensive': 'Intensive study before exams'
            }
            personalization_updates['study_frequency'] = frequency_mapping.get(data['slide3'], data['slide3'])
        
        # Update user with personalization data (only if there's data to update)
        if personalization_updates:
            user_ref.update(personalization_updates)
        else:
            print(f"No personalization data to save for user {current_user.id}")
            # Set default values if no data provided
            user_ref.update({
                'goals': 'Study efficiency and time optimization',
                'study_style': 'Interactive learning with quizzes and games',
                'study_frequency': 'Weekly study sessions'
            })
        
        return jsonify({"success": True, "message": "Personalization saved successfully."})
    except Exception as e:
        print(f"Error saving personalization for user {current_user.id}: {e}")
        return jsonify({"success": False, "message": "An error occurred while saving personalization."}), 500

@app.route('/onboarding/complete', methods=['POST'])
@login_required
def complete_onboarding():
    """Marks the user's onboarding as complete."""
    try:
        user_ref = db.collection('users').document(current_user.id)
        user_ref.update({'has_completed_onboarding': True})
        
        # Force refresh the current_user object from database
        user_doc = db.collection('users').document(current_user.id).get()
        if user_doc.exists:
            updated_user = User.from_dict(user_doc.to_dict())
            print(f"Onboarding complete: Database shows has_completed_onboarding = {updated_user.has_completed_onboarding}")
            
            # Clear the current session and re-login
            logout_user()
            login_user(updated_user, remember=True)
            print(f"Onboarding complete: After logout/login, current_user.has_completed_onboarding = {current_user.has_completed_onboarding}")
        
        return jsonify({"success": True, "redirect": url_for('dashboard')})
    except Exception as e:
        print(f"Error completing onboarding for user {current_user.id}: {e}")
        return jsonify({"success": False, "message": "An error occurred."}), 500

@app.route('/onboarding/restart', methods=['POST'])
@login_required
def restart_onboarding():
    """Restarts the onboarding process for testing purposes."""
    try:
        user_ref = db.collection('users').document(current_user.id)
        user_ref.update({'has_completed_onboarding': False})
        
        # Force refresh the current_user object from database
        user_doc = db.collection('users').document(current_user.id).get()
        if user_doc.exists:
            updated_user = User.from_dict(user_doc.to_dict())
            
            # Clear the current session and re-login
            logout_user()
            login_user(updated_user, remember=True)
        
        # Return a redirect response to force a fresh page load with updated user state
        return jsonify({
            "success": True, 
            "message": "Onboarding restarted successfully!",
            "redirect": "/dashboard"
        })
    except Exception as e:
        print(f"Error restarting onboarding for user {current_user.id}: {e}")
        return jsonify({"success": False, "message": "An error occurred while restarting onboarding."}), 500

@app.route('/onboarding/clear_demo_data', methods=['POST'])
@login_required
def clear_demo_data():
    """Clears demo hubs and content created during onboarding."""
    try:
        # Find and delete demo hubs (hubs with "Welcome" in the name)
        hubs_query = db.collection('hubs').where('user_id', '==', current_user.id).stream()
        demo_hub_ids = []
        
        for hub_doc in hubs_query:
            hub_data = hub_doc.to_dict()
            if 'Welcome' in hub_data.get('name', ''):
                demo_hub_ids.append(hub_doc.id)
        
        # Delete demo hubs and all associated content
        batch = db.batch()
        for hub_id in demo_hub_ids:
            # Delete hub
            batch.delete(db.collection('hubs').document(hub_id))
            
            # Delete notes
            notes_query = db.collection('notes').where('hub_id', '==', hub_id).stream()
            for note_doc in notes_query:
                batch.delete(note_doc.reference)
            
            # Delete activities (flashcards, quizzes)
            activities_query = db.collection('activities').where('hub_id', '==', hub_id).stream()
            for activity_doc in activities_query:
                batch.delete(activity_doc.reference)
            
            # Delete folders
            folders_query = db.collection('folders').where('hub_id', '==', hub_id).stream()
            for folder_doc in folders_query:
                batch.delete(folder_doc.reference)
            
            # Delete sessions
            sessions_query = db.collection('sessions').where('hub_id', '==', hub_id).stream()
            for session_doc in sessions_query:
                batch.delete(session_doc.reference)
        
        batch.commit()
        
        deleted_count = len(demo_hub_ids)
        return jsonify({
            "success": True, 
            "message": f"Cleared {deleted_count} demo hub(s) and all associated content."
        })
    except Exception as e:
        print(f"Error clearing demo data for user {current_user.id}: {e}")
        return jsonify({"success": False, "message": "An error occurred while clearing demo data."}), 500

# ==============================================================================
# 8. COMMUNITY FEATURES
# ==============================================================================

@app.route('/api/study-groups/create', methods=['POST'])
@login_required
def create_study_group():
    """Create a new study group with a 5-digit code"""
    try:
        data = request.get_json()
        name = data.get('name', '').strip()
        description = data.get('description', '').strip()
        
        if not name:
            return jsonify({"success": False, "message": "Study group name is required"}), 400
        
        # Generate unique 5-digit code
        import random
        while True:
            code = str(random.randint(10000, 99999))
            # Check if code already exists
            existing = db.collection('study_groups').where('code', '==', code).limit(1).get()
            if not existing:
                break
        
        # Create study group
        study_group_id = str(uuid.uuid4())
        study_group = StudyGroup(
            id=study_group_id,
            name=name,
            description=description,
            code=code,
            owner_id=current_user.id
        )
        
        # Save to database
        db.collection('study_groups').document(study_group_id).set(study_group.to_dict())
        
        # Add owner as first member
        member_id = str(uuid.uuid4())
        member = StudyGroupMember(
            id=member_id,
            study_group_id=study_group_id,
            user_id=current_user.id
        )
        db.collection('study_group_members').document(member_id).set(member.to_dict())
        
        return jsonify({
            "success": True, 
            "study_group": study_group.to_dict(),
            "code": code
        })
        
    except Exception as e:
        print(f"Error creating study group: {e}")
        return jsonify({"success": False, "message": "Failed to create study group"}), 500

@app.route('/api/study-groups/join', methods=['POST'])
@login_required
def join_study_group():
    """Join a study group using a 5-digit code"""
    try:
        data = request.get_json()
        code = data.get('code', '').strip()
        
        if not code or len(code) != 5:
            return jsonify({"success": False, "message": "Invalid study group code"}), 400
        
        # Find study group by code
        study_groups = db.collection('study_groups').where('code', '==', code).limit(1).get()
        
        if not study_groups:
            return jsonify({"success": False, "message": "Study group not found"}), 404
        
        study_group_doc = study_groups[0]
        study_group = StudyGroup.from_dict(study_group_doc.to_dict())
        
        # Check if user is already a member
        existing_members = db.collection('study_group_members').where(filter=firestore.FieldFilter('study_group_id', '==', study_group.id)).where(filter=firestore.FieldFilter('user_id', '==', current_user.id)).limit(1).get()
        
        if existing_members:
            return jsonify({"success": False, "message": "You are already a member of this study group"}), 400
        
        # Add user as member
        member_id = str(uuid.uuid4())
        member = StudyGroupMember(
            id=member_id,
            study_group_id=study_group.id,
            user_id=current_user.id
        )
        db.collection('study_group_members').document(member_id).set(member.to_dict())
        
        # Update member count
        study_group.member_count += 1
        db.collection('study_groups').document(study_group.id).update({'member_count': study_group.member_count})
        
        return jsonify({
            "success": True,
            "study_group": study_group.to_dict()
        })
        
    except Exception as e:
        print(f"Error joining study group: {e}")
        return jsonify({"success": False, "message": "Failed to join study group"}), 500

@app.route('/api/study-groups', methods=['GET'])
@login_required
def get_user_study_groups():
    """Get all study groups the user is a member of"""
    try:
        # Get all study groups where user is a member
        member_docs = db.collection('study_group_members').where('user_id', '==', current_user.id).get()
        
        study_groups = []
        for member_doc in member_docs:
            member = StudyGroupMember.from_dict(member_doc.to_dict())
            study_group_doc = db.collection('study_groups').document(member.study_group_id).get()
            
            if study_group_doc.exists:
                study_group = StudyGroup.from_dict(study_group_doc.to_dict())
                study_groups.append(study_group.to_dict())
        
        return jsonify({
            "success": True,
            "study_groups": study_groups
        })
        
    except Exception as e:
        print(f"Error getting study groups: {e}")
        return jsonify({"success": False, "message": "Failed to get study groups"}), 500

@app.route('/api/study-groups/<study_group_id>/resources', methods=['GET'])
@login_required
def get_study_group_resources(study_group_id):
    """Get all resources shared in a specific study group"""
    try:
        # Verify user is a member of the study group
        member_docs = db.collection('study_group_members').where(filter=firestore.FieldFilter('study_group_id', '==', study_group_id)).where(filter=firestore.FieldFilter('user_id', '==', current_user.id)).limit(1).get()
        
        if not member_docs:
            return jsonify({"success": False, "message": "You are not a member of this study group"}), 403
        
        # Get all resources shared in this study group
        resource_docs = db.collection('shared_resources').where('study_group_id', '==', study_group_id).order_by('created_at', direction=firestore.Query.DESCENDING).get()
        
        resources = []
        for doc in resource_docs:
            resource = SharedResource.from_dict(doc.to_dict())
            resources.append(resource.to_dict())
        
        return jsonify({
            "success": True,
            "resources": resources
        })
        
    except Exception as e:
        print(f"Error getting study group resources: {e}")
        return jsonify({"success": False, "message": "Failed to get study group resources"}), 500

@app.route('/api/resources/share', methods=['POST'])
@login_required
def share_resource():
    """Share a resource (folder, note, flashcard, quiz, cheatsheet) to global or study group"""
    try:
        data = request.get_json()
        resource_type = data.get('resource_type')  # 'folder', 'note', 'flashcard', 'quiz', 'cheatsheet'
        resource_id = data.get('resource_id')
        hub_id = data.get('hub_id')
        title = data.get('title', '').strip()
        description = data.get('description', '').strip()
        tags = data.get('tags', [])
        share_to = data.get('share_to')  # 'global' or study_group_id
        study_group_id = None if share_to == 'global' else share_to
        
        if not all([resource_type, resource_id, hub_id, title]):
            return jsonify({"success": False, "message": "Missing required fields"}), 400
        
        # If sharing to study group, verify user is a member
        if study_group_id:
            member_docs = db.collection('study_group_members').where(filter=firestore.FieldFilter('study_group_id', '==', study_group_id)).where(filter=firestore.FieldFilter('user_id', '==', current_user.id)).limit(1).get()
            if not member_docs:
                return jsonify({"success": False, "message": "You are not a member of this study group"}), 403
        
        # Check if resource is already shared
        query = db.collection('shared_resources').where(filter=firestore.FieldFilter('resource_type', '==', resource_type)).where(filter=firestore.FieldFilter('resource_id', '==', resource_id)).where(filter=firestore.FieldFilter('owner_id', '==', current_user.id))
        
        if study_group_id:
            query = query.where(filter=firestore.FieldFilter('study_group_id', '==', study_group_id))
        else:
            query = query.where(filter=firestore.FieldFilter('study_group_id', '==', None))
        
        existing = query.limit(1).get()
        
        if existing:
            return jsonify({"success": False, "message": "This resource is already shared"}), 400
        
        # Create shared resource
        shared_resource_id = str(uuid.uuid4())
        shared_resource = SharedResource(
            id=shared_resource_id,
            resource_type=resource_type,
            resource_id=resource_id,
            hub_id=hub_id,
            owner_id=current_user.id,
            study_group_id=study_group_id,
            title=title,
            description=description,
            tags=tags
        )
        
        # Save to database
        db.collection('shared_resources').document(shared_resource_id).set(shared_resource.to_dict())
        
        return jsonify({
            "success": True,
            "shared_resource": shared_resource.to_dict()
        })
        
    except Exception as e:
        print(f"Error sharing resource: {e}")
        return jsonify({"success": False, "message": "Failed to share resource"}), 500

@app.route('/api/resources/global', methods=['GET'])
@login_required
def get_global_resources():
    """Get all globally shared resources with user profile info and folder contents"""
    try:
        # Get all resources shared globally (study_group_id is None)
        resource_docs = db.collection('shared_resources').where('study_group_id', '==', None).order_by('created_at', direction=firestore.Query.DESCENDING).get()
        
        resources = []
        for doc in resource_docs:
            resource = SharedResource.from_dict(doc.to_dict())
            resource_dict = resource.to_dict()
            
            # Get user profile information
            user_doc = db.collection('users').document(resource.owner_id).get()
            user_profile = user_doc.to_dict() if user_doc.exists else {}
            
            # Get folder contents if it's a folder
            folder_contents = []
            if resource.resource_type == 'folder':
                folder_doc = db.collection('folders').document(resource.resource_id).get()
                if folder_doc.exists:
                    folder_data = folder_doc.to_dict()
                    items = folder_data.get('items', [])
                    
                    # Get details for each item in the folder
                    for item in items:
                        item_id = item.get('id')
                        item_type = item.get('type')
                        
                        if item_type == 'note':
                            note_doc = db.collection('notes').document(item_id).get()
                            if note_doc.exists:
                                note_data = note_doc.to_dict()
                                folder_contents.append({
                                    'id': item_id,
                                    'type': 'note',
                                    'title': note_data.get('title', 'Untitled Note'),
                                    'icon': '📝'
                                })
                        elif item_type in ['quiz', 'flashcards']:
                            activity_doc = db.collection('activities').document(item_id).get()
                            if activity_doc.exists:
                                activity_data = activity_doc.to_dict()
                                folder_contents.append({
                                    'id': item_id,
                                    'type': item_type,
                                    'title': activity_data.get('title', f'Untitled {item_type.title()}'),
                                    'icon': '🧠' if item_type == 'quiz' else '🃏'
                                })
            
            # Add user profile and folder contents to resource
            resource_dict['owner_name'] = user_profile.get('display_name', 'Unknown User')
            resource_dict['owner_avatar'] = user_profile.get('profile_picture_url', '/static/images/default-avatar.svg')
            resource_dict['folder_contents'] = folder_contents
            
            resources.append(resource_dict)
        
        return jsonify({
            "success": True,
            "resources": resources
        })
        
    except Exception as e:
        print(f"Error getting global resources: {e}")
        return jsonify({"success": False, "message": "Failed to get global resources"}), 500

@app.route('/api/resources/<resource_id>/import', methods=['POST'])
@login_required
def import_shared_resource(resource_id):
    """Import a shared resource to user's hub"""
    try:
        data = request.get_json()
        target_hub_id = data.get('hub_id')
        
        # Get the shared resource
        shared_resource_doc = db.collection('shared_resources').document(resource_id).get()
        
        if not shared_resource_doc.exists:
            return jsonify({"success": False, "message": "Resource not found"}), 404
        
        shared_resource = SharedResource.from_dict(shared_resource_doc.to_dict())
        
        # Check if user has already imported this resource
        if current_user.id in shared_resource.imported_by:
            return jsonify({"success": False, "message": "You have already imported this resource"}), 400
        
        # Get the target hub
        if target_hub_id:
            # Use specified hub
            target_hub_doc = db.collection('hubs').document(target_hub_id).get()
            if not target_hub_doc.exists:
                return jsonify({"success": False, "message": "Target hub not found"}), 404
            
            # Verify user owns the hub
            target_hub = Hub.from_dict(target_hub_doc.to_dict())
            if target_hub.user_id != current_user.id:
                return jsonify({"success": False, "message": "You don't have permission to import to this hub"}), 403
        else:
            # Get user's default hub (or first hub)
            user_hubs = db.collection('hubs').where('user_id', '==', current_user.id).limit(1).get()
            
            if not user_hubs:
                return jsonify({"success": False, "message": "No hub found to import resource to"}), 400
            
            target_hub = Hub.from_dict(user_hubs[0].to_dict())
        
        # Import the resource based on type
        if shared_resource.resource_type == 'folder':
            # Get the original folder
            original_folder_doc = db.collection('folders').document(shared_resource.resource_id).get()
            if not original_folder_doc.exists:
                return jsonify({"success": False, "message": "Original folder not found"}), 404
            
            original_folder = Folder.from_dict(original_folder_doc.to_dict())
            
            # Create new folder in user's hub
            new_folder_id = str(uuid.uuid4())
            new_folder_items = []
            
            # Copy each item from the original folder
            for item_ref in original_folder.items:
                item_id = item_ref.get('id')
                item_type = item_ref.get('type')
                
                if item_type == 'note':
                    # Copy the note
                    original_note_doc = db.collection('notes').document(item_id).get()
                    if original_note_doc.exists:
                        original_note = Note.from_dict(original_note_doc.to_dict())
                        new_note_id = str(uuid.uuid4())
                        new_note = Note(
                            id=new_note_id,
                            hub_id=target_hub.id,
                            title=f"{original_note.title} (Imported)",
                            content_html=original_note.content_html,
                            created_at=datetime.now(timezone.utc)
                        )
                        db.collection('notes').document(new_note_id).set(new_note.to_dict())
                        new_folder_items.append({'id': new_note_id, 'type': 'note'})
                
                elif item_type in ['quiz', 'flashcards']:
                    # Copy the activity
                    original_activity_doc = db.collection('activities').document(item_id).get()
                    if original_activity_doc.exists:
                        original_activity = Activity.from_dict(original_activity_doc.to_dict())
                        new_activity_id = str(uuid.uuid4())
                        new_activity = Activity(
                            id=new_activity_id,
                            hub_id=target_hub.id,
                            type=original_activity.type,
                            title=f"{original_activity.title} (Imported)",
                            data=original_activity.data,
                            status='completed',  # Mark as completed since it's imported
                            created_at=datetime.now(timezone.utc)
                        )
                        db.collection('activities').document(new_activity_id).set(new_activity.to_dict())
                        new_folder_items.append({'id': new_activity_id, 'type': item_type})
            
            # Create the new folder with copied items
            new_folder = Folder(
                id=new_folder_id,
                hub_id=target_hub.id,
                name=f"{original_folder.name} (Imported)",
                items=new_folder_items,
                created_at=datetime.now(timezone.utc)
            )
            db.collection('folders').document(new_folder_id).set(new_folder.to_dict())
            
        elif shared_resource.resource_type == 'note':
            # Get the original note
            original_note_doc = db.collection('notes').document(shared_resource.resource_id).get()
            if not original_note_doc.exists:
                return jsonify({"success": False, "message": "Original note not found"}), 404
            
            original_note = Note.from_dict(original_note_doc.to_dict())
            
            # Create new note in user's hub
            new_note_id = str(uuid.uuid4())
            new_note = Note(
                id=new_note_id,
                hub_id=target_hub.id,
                title=f"{original_note.title} (Imported)",
                content_html=original_note.content_html,
                created_at=datetime.now(timezone.utc)
            )
            db.collection('notes').document(new_note_id).set(new_note.to_dict())
            
        # Add similar logic for flashcards, quizzes, and cheatsheets...
        # (Implementation would be similar to above)
        
        # Update the shared resource to track the import
        shared_resource.imports += 1
        shared_resource.imported_by.append(current_user.id)
        db.collection('shared_resources').document(resource_id).update({
            'imports': shared_resource.imports,
            'imported_by': shared_resource.imported_by
        })
        
        return jsonify({
            "success": True,
            "message": "Resource imported successfully"
        })
        
    except Exception as e:
        print(f"Error importing resource: {e}")
        return jsonify({"success": False, "message": "Failed to import resource"}), 500

@app.route('/api/resources/<resource_id>', methods=['GET'])
@login_required
def get_shared_resource(resource_id):
    """Get a single shared resource by ID"""
    try:
        shared_resource_doc = db.collection('shared_resources').document(resource_id).get()
        if not shared_resource_doc.exists:
            return jsonify({"success": False, "message": "Resource not found"}), 404
        
        shared_resource = SharedResource.from_dict(shared_resource_doc.to_dict())
        return jsonify({"success": True, "resource": shared_resource.to_dict()})
        
    except Exception as e:
        print(f"Error getting resource: {e}")
        return jsonify({"success": False, "message": "Error getting resource"}), 500

@app.route('/api/resources/<resource_id>/like', methods=['POST'])
@login_required
def like_shared_resource(resource_id):
    """Like or unlike a shared resource"""
    try:
        # Get the shared resource
        shared_resource_doc = db.collection('shared_resources').document(resource_id).get()
        if not shared_resource_doc.exists:
            return jsonify({"success": False, "message": "Resource not found"}), 404
        
        shared_resource = SharedResource.from_dict(shared_resource_doc.to_dict())
        
        # Check if user has already liked this resource
        user_liked = current_user.id in shared_resource.liked_by
        
        if user_liked:
            # Unlike the resource
            shared_resource_doc.reference.update({
                'likes': shared_resource.likes - 1,
                'liked_by': firestore.ArrayRemove([current_user.id])
            })
            return jsonify({"success": True, "liked": False, "likes": shared_resource.likes - 1})
        else:
            # Like the resource
            shared_resource_doc.reference.update({
                'likes': shared_resource.likes + 1,
                'liked_by': firestore.ArrayUnion([current_user.id])
            })
            return jsonify({"success": True, "liked": True, "likes": shared_resource.likes + 1})
        
    except Exception as e:
        print(f"Error liking resource: {e}")
        return jsonify({"success": False, "message": "Error liking resource"}), 500

# Edit title API endpoints
@app.route('/api/folder/<folder_id>/update_title', methods=['POST'])
@login_required
def update_folder_title(folder_id):
    """Update folder title"""
    try:
        data = request.get_json()
        new_title = data.get('title', '').strip()
        
        if not new_title:
            return jsonify({"success": False, "message": "Title cannot be empty"}), 400
        
        # Get the folder and verify ownership
        folder_doc = db.collection('folders').document(folder_id).get()
        if not folder_doc.exists:
            return jsonify({"success": False, "message": "Folder not found"}), 404
        
        folder = Folder.from_dict(folder_doc.to_dict())
        
        # Verify user owns the hub that contains this folder
        hub_doc = db.collection('hubs').document(folder.hub_id).get()
        if not hub_doc.exists:
            return jsonify({"success": False, "message": "Hub not found"}), 404
        
        hub = Hub.from_dict(hub_doc.to_dict())
        if hub.user_id != current_user.id:
            return jsonify({"success": False, "message": "You don't have permission to edit this folder"}), 403
        
        # Update the folder title
        folder_doc.reference.update({'name': new_title})
        
        return jsonify({"success": True, "message": "Folder title updated successfully"})
        
    except Exception as e:
        print(f"Error updating folder title: {e}")
        return jsonify({"success": False, "message": "Error updating folder title"}), 500

@app.route('/api/note/<note_id>/update_title', methods=['POST'])
@login_required
def update_note_title(note_id):
    """Update note title"""
    try:
        data = request.get_json()
        new_title = data.get('title', '').strip()
        
        if not new_title:
            return jsonify({"success": False, "message": "Title cannot be empty"}), 400
        
        # Get the note and verify ownership
        note_doc = db.collection('notes').document(note_id).get()
        if not note_doc.exists:
            return jsonify({"success": False, "message": "Note not found"}), 404
        
        note = Note.from_dict(note_doc.to_dict())
        
        # Verify user owns the hub that contains this note
        hub_doc = db.collection('hubs').document(note.hub_id).get()
        if not hub_doc.exists:
            return jsonify({"success": False, "message": "Hub not found"}), 404
        
        hub = Hub.from_dict(hub_doc.to_dict())
        if hub.user_id != current_user.id:
            return jsonify({"success": False, "message": "You don't have permission to edit this note"}), 403
        
        # Update the note title
        note_doc.reference.update({'title': new_title})
        
        return jsonify({"success": True, "message": "Note title updated successfully"})
        
    except Exception as e:
        print(f"Error updating note title: {e}")
        return jsonify({"success": False, "message": "Error updating note title"}), 500

@app.route('/api/flashcards/<activity_id>/update_title', methods=['POST'])
@login_required
def update_flashcards_title(activity_id):
    """Update flashcards title"""
    try:
        data = request.get_json()
        new_title = data.get('title', '').strip()
        
        if not new_title:
            return jsonify({"success": False, "message": "Title cannot be empty"}), 400
        
        # Get the activity and verify ownership
        activity_doc = db.collection('activities').document(activity_id).get()
        if not activity_doc.exists:
            return jsonify({"success": False, "message": "Activity not found"}), 404
        
        activity = Activity.from_dict(activity_doc.to_dict())
        
        # Verify user owns the hub that contains this activity
        hub_doc = db.collection('hubs').document(activity.hub_id).get()
        if not hub_doc.exists:
            return jsonify({"success": False, "message": "Hub not found"}), 404
        
        hub = Hub.from_dict(hub_doc.to_dict())
        if hub.user_id != current_user.id:
            return jsonify({"success": False, "message": "You don't have permission to edit this activity"}), 403
        
        # Update the activity title
        activity_doc.reference.update({'title': new_title})
        
        return jsonify({"success": True, "message": "Flashcards title updated successfully"})
        
    except Exception as e:
        print(f"Error updating flashcards title: {e}")
        return jsonify({"success": False, "message": "Error updating flashcards title"}), 500

@app.route('/api/quiz/<activity_id>/update_title', methods=['POST'])
@login_required
def update_quiz_title(activity_id):
    """Update quiz title"""
    try:
        data = request.get_json()
        new_title = data.get('title', '').strip()
        
        if not new_title:
            return jsonify({"success": False, "message": "Title cannot be empty"}), 400
        
        # Get the activity and verify ownership
        activity_doc = db.collection('activities').document(activity_id).get()
        if not activity_doc.exists:
            return jsonify({"success": False, "message": "Activity not found"}), 404
        
        activity = Activity.from_dict(activity_doc.to_dict())
        
        # Verify user owns the hub that contains this activity
        hub_doc = db.collection('hubs').document(activity.hub_id).get()
        if not hub_doc.exists:
            return jsonify({"success": False, "message": "Hub not found"}), 404
        
        hub = Hub.from_dict(hub_doc.to_dict())
        if hub.user_id != current_user.id:
            return jsonify({"success": False, "message": "You don't have permission to edit this activity"}), 403
        
        # Update the activity title
        activity_doc.reference.update({'title': new_title})
        
        return jsonify({"success": True, "message": "Quiz title updated successfully"})
        
    except Exception as e:
        print(f"Error updating quiz title: {e}")
        return jsonify({"success": False, "message": "Error updating quiz title"}), 500

@app.route('/api/slide_notes/<session_id>/update_title', methods=['POST'])
@login_required
def update_slide_notes_title(session_id):
    """Update slide notes title"""
    try:
        data = request.get_json()
        new_title = data.get('title', '').strip()
        
        if not new_title:
            return jsonify({"success": False, "message": "Title cannot be empty"}), 400
        
        # Get the slide notes session and verify ownership
        session_doc = db.collection('slide_notes_sessions').document(session_id).get()
        if not session_doc.exists:
            return jsonify({"success": False, "message": "Session not found"}), 404
        
        session_data = session_doc.to_dict()
        
        # Verify user owns the hub that contains this session
        hub_doc = db.collection('hubs').document(session_data['hub_id']).get()
        if not hub_doc.exists:
            return jsonify({"success": False, "message": "Hub not found"}), 404
        
        hub = Hub.from_dict(hub_doc.to_dict())
        if hub.user_id != current_user.id:
            return jsonify({"success": False, "message": "You don't have permission to edit this session"}), 403
        
        # Update the session title
        session_doc.reference.update({'title': new_title})
        
        return jsonify({"success": True, "message": "Slide notes title updated successfully"})
        
    except Exception as e:
        print(f"Error updating slide notes title: {e}")
        return jsonify({"success": False, "message": "Error updating slide notes title"}), 500

@app.route('/api/resources/<resource_id>/delete', methods=['DELETE'])
@login_required
def delete_shared_resource(resource_id):
    """Delete a shared resource (only by owner)"""
    try:
        # Get the shared resource
        shared_resource_doc = db.collection('shared_resources').document(resource_id).get()
        if not shared_resource_doc.exists:
            return jsonify({"success": False, "message": "Resource not found"}), 404
        
        shared_resource = SharedResource.from_dict(shared_resource_doc.to_dict())
        
        # Check if user owns this resource
        if shared_resource.owner_id != current_user.id:
            return jsonify({"success": False, "message": "You don't have permission to delete this resource"}), 403
        
        # Delete the shared resource
        shared_resource_doc.reference.delete()
        
        return jsonify({"success": True, "message": "Resource deleted successfully"})
        
    except Exception as e:
        print(f"Error deleting resource: {e}")
        return jsonify({"success": False, "message": "Error deleting resource"}), 500

# ==============================================================================
# 8. REFERRAL SYSTEM API ENDPOINTS
# ==============================================================================

@app.route('/api/referrals/generate-code', methods=['POST'])
@login_required
def generate_user_referral_code():
    """Generate or get user's referral code"""
    try:
        # Check if user already has a referral code
        if current_user.referral_code:
            return jsonify({
                "success": True, 
                "referral_code": current_user.referral_code,
                "message": "Referral code retrieved"
            })
        
        # Generate new referral code for user
        new_code = generate_referral_code()
        
        # Update user in database
        user_ref = db.collection('users').document(current_user.id)
        user_ref.update({'referral_code': new_code})
        
        # Update current user object
        current_user.referral_code = new_code
        
        return jsonify({
            "success": True, 
            "referral_code": new_code,
            "message": "Referral code generated"
        })
        
    except Exception as e:
        print(f"Error generating referral code: {e}")
        return jsonify({"success": False, "message": "Error generating referral code"}), 500

@app.route('/api/referrals/validate-code', methods=['POST'])
@login_required
def validate_user_referral_code():
    """Validate a referral code"""
    try:
        data = request.get_json()
        code = data.get('code', '').strip()
        
        if not code:
            return jsonify({"success": False, "message": "Referral code is required"}), 400
        
        # Check if user is trying to use their own code
        if code == current_user.referral_code:
            return jsonify({"success": False, "message": "You cannot use your own referral code"}), 400
        
        # Validate the code
        referrer = validate_referral_code(code)
        if referrer:
            return jsonify({
                "success": True, 
                "valid": True,
                "referrer_name": referrer.display_name,
                "message": f"Valid referral code from {referrer.display_name}"
            })
        else:
            return jsonify({
                "success": True, 
                "valid": False,
                "message": "Invalid referral code"
            })
            
    except Exception as e:
        print(f"Error validating referral code: {e}")
        return jsonify({"success": False, "message": "Error validating referral code"}), 500

@app.route('/api/referrals/user-stats', methods=['GET'])
@login_required
def get_user_referral_stats():
    """Get user's referral statistics"""
    try:
        # Handle existing users who don't have referral fields yet
        user_ref = db.collection('users').document(current_user.id)
        user_doc = user_ref.get()
        
        if user_doc.exists:
            user_data = user_doc.to_dict()
            
            # Check if user needs referral fields initialized
            needs_update = False
            if 'referral_code' not in user_data:
                user_data['referral_code'] = generate_referral_code()
                needs_update = True
            if 'referred_by' not in user_data:
                user_data['referred_by'] = None
                needs_update = True
            if 'pro_referral_count' not in user_data:
                user_data['pro_referral_count'] = 0
                needs_update = True
            if 'referral_earnings' not in user_data:
                user_data['referral_earnings'] = 0.0
                needs_update = True
            
            # Update user record if needed
            if needs_update:
                user_ref.update({
                    'referral_code': user_data['referral_code'],
                    'referred_by': user_data['referred_by'],
                    'pro_referral_count': user_data['pro_referral_count'],
                    'referral_earnings': user_data['referral_earnings']
                })
                print(f"✅ Updated user {current_user.email} with referral fields")
        
        # Get user's current stats (with defaults for missing fields)
        stats = {
            "pro_referral_count": getattr(current_user, 'pro_referral_count', 0),
            "referral_earnings": getattr(current_user, 'referral_earnings', 0.0),
            "referral_code": getattr(current_user, 'referral_code', None) or user_data.get('referral_code', 'Generating...'),
            "referred_by": getattr(current_user, 'referred_by', None),
            "milestones": {
                "4": {"reached": getattr(current_user, 'pro_referral_count', 0) >= 4, "reward": "£10 Amazon giftcard"},
                "10": {"reached": getattr(current_user, 'pro_referral_count', 0) >= 10, "reward": "£20 Amazon giftcard"},
                "20": {"reached": getattr(current_user, 'pro_referral_count', 0) >= 20, "reward": "£50 Amazon giftcard"},
                "50": {"reached": getattr(current_user, 'pro_referral_count', 0) >= 50, "reward": "£100 Amazon giftcard"}
            }
        }
        
        return jsonify({"success": True, "stats": stats})
        
    except Exception as e:
        print(f"Error getting referral stats: {e}")
        return jsonify({"success": False, "message": "Error getting referral stats"}), 500

@app.route('/api/referrals/leaderboard', methods=['GET'])
@login_required
def get_referral_leaderboard():
    """Get monthly referral leaderboard"""
    try:
        # Get top users by pro referral count for current month
        current_month = datetime.now(timezone.utc).strftime('%Y-%m')
        
        # Query users sorted by pro_referral_count (descending)
        users_query = db.collection('users').order_by('pro_referral_count', direction=firestore.Query.DESCENDING).limit(10).stream()
        
        leaderboard = []
        for user_doc in users_query:
            user_data = user_doc.to_dict()
            if user_data.get('pro_referral_count', 0) > 0:  # Only include users with referrals
                leaderboard.append({
                    "display_name": user_data.get('display_name', 'Anonymous'),
                    "pro_referral_count": user_data.get('pro_referral_count', 0),
                    "avatar_url": user_data.get('avatar_url', '/static/images/default-avatar.svg')
                })
        
        return jsonify({"success": True, "leaderboard": leaderboard})
        
    except Exception as e:
        print(f"Error getting leaderboard: {e}")
        return jsonify({"success": False, "message": "Error getting leaderboard"}), 500

@app.route('/api/referrals/add-code', methods=['POST'])
@login_required
def add_referral_code():
    """Add a referral code for users who didn't use one during signup"""
    try:
        data = request.get_json()
        code = data.get('code', '').strip()
        
        if not code:
            return jsonify({"success": False, "message": "Referral code is required"}), 400
        
        # Check if user already has a referral (prevent multiple referral codes)
        if current_user.referred_by:
            return jsonify({"success": False, "message": "You have already used a referral code"}), 400
        
        # Validate the referral code
        referrer = validate_referral_code(code)
        if not referrer:
            return jsonify({"success": False, "message": "Invalid referral code"}), 400
        
        # Check if user is trying to use their own code
        if referrer.id == current_user.id:
            return jsonify({"success": False, "message": "You cannot use your own referral code"}), 400
        
        # Update user with referral information
        user_ref = db.collection('users').document(current_user.id)
        user_ref.update({
            'referred_by': referrer.id
        })
        
        # Create referral record
        referral_ref = db.collection('referrals').document()
        referral = Referral(
            id=referral_ref.id,
            referrer_id=referrer.id,
            referred_id=current_user.id,
            referral_code=code,
            status='pending'
        )
        referral_ref.set(referral.to_dict())
        
        print(f"✅ User {current_user.email} added referral code {code} from {referrer.email}")
        
        return jsonify({
            "success": True, 
            "message": f"Referral code added successfully! You're now supporting {referrer.display_name}"
        })
        
    except Exception as e:
        print(f"Error adding referral code: {e}")
        return jsonify({"success": False, "message": "Error adding referral code"}), 500


@app.route("/spaced_repetition/review")
@login_required
def spaced_repetition_review():
    """Serve the spaced repetition review interface"""
    return render_template("spaced_repetition_review.html")


@app.route("/api/spaced_repetition/migrate_flashcards/<activity_id>", methods=["POST"])
@login_required
def migrate_existing_flashcards(activity_id):
    """Migrate existing flashcards to spaced repetition system"""
    try:
        # Get the original activity
        activity_doc = db.collection('activities').document(activity_id).get()
        if not activity_doc.exists:
            return jsonify({"success": False, "message": "Activity not found"}), 404
        
        activity = Activity.from_dict(activity_doc.to_dict())
        
        # Check if user owns this activity
        hub_doc = db.collection('hubs').document(activity.hub_id).get()
        if not hub_doc.exists or hub_doc.to_dict().get('user_id') != current_user.id:
            return jsonify({"success": False, "message": "Unauthorized"}), 403
        
        # Check if already migrated
        existing_sr_cards = db.collection('spaced_repetition_cards').where('activity_id', '==', activity_id).stream()
        if list(existing_sr_cards):
            return jsonify({"success": True, "message": "Flashcards already migrated"})
        
        # Get flashcards data
        flashcards_data = activity.data.get('cards', [])
        if not flashcards_data:
            return jsonify({"success": False, "message": "No flashcards found in activity"}), 400
        
        # Migrate flashcards
        migrate_flashcards_to_spaced_repetition(activity_id, flashcards_data)
        
        return jsonify({
            "success": True, 
            "message": f"Successfully migrated {len(flashcards_data)} flashcards to spaced repetition system"
        })
        
    except Exception as e:
        print(f"Error migrating flashcards: {e}")
        return jsonify({"success": False, "message": "Failed to migrate flashcards"}), 500

@app.route("/api/spaced_repetition/enhanced_flashcards/<activity_id>")
@login_required
def get_enhanced_flashcards(activity_id):
    """Get flashcards with spaced repetition data for enhanced interface"""
    try:
        # Get the original activity
        activity_doc = db.collection('activities').document(activity_id).get()
        if not activity_doc.exists:
            return jsonify({"success": False, "message": "Activity not found"}), 404
        
        activity = Activity.from_dict(activity_doc.to_dict())
        
        # Check if user owns this activity
        hub_doc = db.collection('hubs').document(activity.hub_id).get()
        if not hub_doc.exists or hub_doc.to_dict().get('user_id') != current_user.id:
            return jsonify({"success": False, "message": "Unauthorized"}), 403
        
        # Get spaced repetition cards
        sr_cards_query = db.collection('spaced_repetition_cards').where('activity_id', '==', activity_id)
        sr_cards = list(sr_cards_query.stream())
        
        # If no spaced repetition cards exist, migrate them automatically
        if not sr_cards:
            flashcards_data = activity.data.get('cards', [])
            if flashcards_data:
                print(f"Auto-migrating {len(flashcards_data)} flashcards for activity {activity_id}")
                try:
                    migrate_flashcards_to_spaced_repetition(activity_id, flashcards_data)
                    # Re-query after migration
                    sr_cards = list(sr_cards_query.stream())
                except Exception as migration_error:
                    print(f"Auto-migration failed for activity {activity_id}: {migration_error}")
                    # Return empty cards instead of crashing
                    return jsonify({
                        "success": True,
                        "activity": {
                            'id': activity.id,
                            'title': activity.title,
                            'hub_id': activity.hub_id
                        },
                        "cards": [],
                        "total_cards": 0,
                        "due_cards": 0,
                        "migration_error": "Failed to migrate flashcards to spaced repetition system"
                    })
        
        enhanced_cards = []
        for sr_card_doc in sr_cards:
            sr_card_data = sr_card_doc.to_dict()
            sr_card = SpacedRepetitionCard.from_dict(sr_card_data)
            
            enhanced_cards.append({
                'id': sr_card.id,
                'front': sr_card.front,
                'back': sr_card.back,
                'ease_factor': sr_card.ease_factor,
                'interval_days': sr_card.interval_days,
                'repetitions': sr_card.repetitions,
                'next_review': sr_card.next_review.isoformat() if sr_card.next_review else None,
                'is_due': sr_card.is_due(),
                'difficulty': sr_card.difficulty
            })
        
        return jsonify({
            "success": True,
            "activity": {
                'id': activity.id,
                'title': activity.title,
                'hub_id': activity.hub_id
            },
            "cards": enhanced_cards,
            "total_cards": len(enhanced_cards),
            "due_cards": len([c for c in enhanced_cards if c['is_due']])
        })
        
    except Exception as e:
        print(f"Error getting enhanced flashcards for activity {activity_id}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "message": f"Failed to get enhanced flashcards: {str(e)}"}), 500


# Import debugging system
from debug_spaced_repetition import debugger, debug_logger, log_user_action, log_system_health

# ==============================================================================
# SPACED REPETITION SYSTEM - PHASE 2 IMPLEMENTATION
# ==============================================================================

@app.route("/api/spaced_repetition/create_session", methods=["POST"])
@login_required
@debug_logger
def create_review_session():
    """Create a new spaced repetition review session"""
    try:
        data = request.get_json()
        hub_id = data.get('hub_id')
        session_type = data.get('session_type', 'spaced_repetition')
        max_cards = data.get('max_cards', 20)
        
        # Enhanced debug logging
        print(f"🔍 DEBUG: Creating spaced repetition session for user {current_user.id}")
        print(f"🔍 DEBUG: Hub ID: {hub_id}, Max cards: {max_cards}, Session type: {session_type}")
        debugger.log_api_call('/api/spaced_repetition/create_session', 'POST', current_user.id, data)
        log_user_action(current_user.id, 'create_review_session', {'hub_id': hub_id, 'max_cards': max_cards})
        
        if not hub_id:
            print(f"❌ ERROR: User {current_user.id} attempted to create session without hub_id")
            debugger.logger.warning(f"User {current_user.id} attempted to create session without hub_id")
            return jsonify({"success": False, "message": "Hub ID is required"}), 400
        
        # Get due cards for this hub
        print(f"🔍 DEBUG: Getting due cards for hub {hub_id}")
        
        # Call the get_due_cards function directly (not the route)
        try:
            # Get all activities in this hub
            activities_query = db.collection('activities').where('hub_id', '==', hub_id).where('type', '==', 'Flashcards')
            activities = list(activities_query.stream())
            print(f"🔍 DEBUG: Found {len(activities)} flashcard activities in hub {hub_id}")
            
            due_cards = []
            total_cards_checked = 0
            
            for activity_doc in activities:
                activity_id = activity_doc.id
                print(f"🔍 DEBUG: Checking activity {activity_id}")
                
                # Get spaced repetition cards for this activity
                sr_cards_query = db.collection('spaced_repetition_cards').where('activity_id', '==', activity_id)
                sr_cards = list(sr_cards_query.stream())
                print(f"🔍 DEBUG: Found {len(sr_cards)} SR cards in activity {activity_id}")
                
                # If no spaced repetition cards exist, migrate them automatically
                if not sr_cards:
                    activity_data = activity_doc.to_dict()
                    flashcards_data = activity_data.get('data', {}).get('cards', [])
                    if flashcards_data:
                        print(f"🔍 DEBUG: Auto-migrating {len(flashcards_data)} flashcards for activity {activity_id}")
                        try:
                            migrate_flashcards_to_spaced_repetition(activity_id, flashcards_data)
                            # Re-query after migration
                            sr_cards = list(sr_cards_query.stream())
                            print(f"🔍 DEBUG: After migration, found {len(sr_cards)} SR cards")
                        except Exception as migration_error:
                            print(f"❌ ERROR: Auto-migration failed for activity {activity_id}: {migration_error}")
                            debugger.logger.error(f"Auto-migration failed for activity {activity_id}: {migration_error}")
                            continue
                
                # Check if activity needs recovery before processing cards
                activity_doc = db.collection('activities').document(activity_id).get()
                if activity_doc.exists:
                    activity_data = activity_doc.to_dict()
                    current_cards = activity_data.get('data', {}).get('cards', [])
                    
                    # If activity has no cards but SR cards exist, recover once for the entire activity
                    if len(current_cards) == 0 and len(sr_cards) > 0:
                        print(f"🔄 AUTO-RECOVERY: Activity {activity_id} has 0 cards, attempting recovery")
                        if recover_missing_flashcards(activity_id):
                            print(f"✅ AUTO-RECOVERY: Recovered cards for activity {activity_id}")
                            # Re-fetch the activity data after recovery
                            activity_doc = db.collection('activities').document(activity_id).get()
                            activity_data = activity_doc.to_dict()
                            current_cards = activity_data.get('data', {}).get('cards', [])
                            print(f"✅ AUTO-RECOVERY: Activity now has {len(current_cards)} cards")
                            
                            # CRITICAL: Re-query SR cards after recovery to get updated data
                            sr_cards_query = db.collection('spaced_repetition_cards').where('activity_id', '==', activity_id)
                            sr_cards = list(sr_cards_query.stream())
                            print(f"✅ AUTO-RECOVERY: Re-queried {len(sr_cards)} SR cards after recovery")
                        else:
                            print(f"❌ AUTO-RECOVERY: Failed to recover cards for activity {activity_id}")
                            # Skip this entire activity if recovery failed
                            continue

                for sr_card_doc in sr_cards:
                    sr_card_data = sr_card_doc.to_dict()
                    total_cards_checked += 1
                    
                    try:
                        sr_card = SpacedRepetitionCard.from_dict(sr_card_data)
                        print(f"🔍 DEBUG: Processing card {sr_card.id}, interval: {sr_card.interval_days}d, reps: {sr_card.repetitions}")
                        
                        # Use the same is_due() logic as get_due_cards
                        if sr_card.is_due():
                            print(f"🔍 DEBUG: Card {sr_card.id} is due for review")
                        # Get the original flashcard data (use the already fetched data)
                        try:
                            flashcard_doc = db.collection('activities').document(activity_id).get()
                            if flashcard_doc.exists:
                                flashcard_data = flashcard_doc.to_dict()
                                flashcards = flashcard_data.get('data', {}).get('cards', [])
                                
                                # Find the specific flashcard using robust matching
                                flashcard = None
                                card_index = sr_card.card_index
                                
                                # First try: Use stored index if valid
                                if 0 <= card_index < len(flashcards):
                                    flashcard = flashcards[card_index]
                                    print(f"✅ DEBUG: Found card by index {card_index}")
                                else:
                                    # Second try: Match by content (front/back text)
                                    print(f"🔍 DEBUG: Index {card_index} out of range, trying content matching")
                                    for idx, fc in enumerate(flashcards):
                                        if (fc.get('front', '').strip() == sr_card.front.strip() and 
                                            fc.get('back', '').strip() == sr_card.back.strip()):
                                            flashcard = fc
                                            card_index = idx
                                            print(f"✅ DEBUG: Found card by content matching at index {idx}")
                                            break
                                
                                if flashcard:
                                    due_cards.append({
                                        'id': sr_card.id,
                                        'front': flashcard.get('front', ''),
                                        'back': flashcard.get('back', ''),
                                        'activity_id': activity_id,
                                        'card_index': card_index,
                                        'next_review': sr_card.next_review.isoformat() if sr_card.next_review else None,
                                        'interval': sr_card.interval_days,
                                        'ease_factor': sr_card.ease_factor,
                                        'repetitions': sr_card.repetitions
                                    })
                                    print(f"✅ DEBUG: Added card {sr_card.id} to due cards")
                                else:
                                    print(f"❌ ERROR: Could not find matching flashcard for SR card {sr_card.id} (index: {sr_card.card_index}, activity: {activity_id})")
                                    print(f"🔍 DEBUG: Activity has {len(flashcards)} cards, SR card front: '{sr_card.front[:50]}...'")
                            else:
                                print(f"❌ ERROR: Activity {activity_id} not found")
                        except Exception as e:
                            print(f"❌ ERROR: Failed to get flashcard data for card {sr_card.id}: {e}")
                            debugger.logger.error(f"Failed to get flashcard data for card {sr_card.id}: {e}")
                        else:
                            next_review = sr_card.next_review.strftime("%Y-%m-%d %H:%M") if sr_card.next_review else "Never"
                            print(f"🔍 DEBUG: Card {sr_card.id} not due (next review: {next_review})")
                    except Exception as card_error:
                        print(f"❌ ERROR: Failed to process card {sr_card_doc.id}: {card_error}")
                        debugger.logger.error(f"Failed to process card {sr_card_doc.id}: {card_error}")
                        
                        # Fallback: Use the same logic as get_due_cards function
                        print(f"🔄 FALLBACK: Using manual due check for card {sr_card_doc.id}")
                        try:
                            # Manual due check (same as get_due_cards)
                            next_review = sr_card_data.get('next_review')
                            if next_review:
                                # Convert Firestore timestamp to datetime if needed
                                if hasattr(next_review, 'timestamp'):
                                    next_review_dt = datetime.fromtimestamp(next_review.timestamp(), tz=timezone.utc)
                                else:
                                    next_review_dt = next_review
                            
                            if next_review_dt <= datetime.now(timezone.utc):
                                print(f"🔍 DEBUG: Card {sr_card_doc.id} is due (manual check)")
                                # Get the original flashcard data
                                flashcard_doc = db.collection('activities').document(activity_id).get()
                                if flashcard_doc.exists:
                                    flashcard_data = flashcard_doc.to_dict()
                                    flashcards = flashcard_data.get('data', {}).get('cards', [])
                                    
                                    # If activity has no cards but SR cards exist, try to recover
                                    if len(flashcards) == 0:
                                        print(f"🔄 AUTO-RECOVERY: Activity {activity_id} has 0 cards, attempting recovery (fallback)")
                                        if recover_missing_flashcards(activity_id):
                                            # Re-fetch the activity data after recovery
                                            flashcard_doc = db.collection('activities').document(activity_id).get()
                                            flashcard_data = flashcard_doc.to_dict()
                                            flashcards = flashcard_data.get('data', {}).get('cards', [])
                                            print(f"✅ AUTO-RECOVERY: Recovered {len(flashcards)} cards for activity {activity_id} (fallback)")
                                        else:
                                            print(f"❌ AUTO-RECOVERY: Failed to recover cards for activity {activity_id} (fallback)")
                                            continue  # Skip this card if recovery failed
                                    
                                    # Find the specific flashcard using robust matching
                                    flashcard = None
                                    card_index = sr_card_data.get('card_index', 0)
                                    
                                    # First try: Use stored index if valid
                                    if 0 <= card_index < len(flashcards):
                                        flashcard = flashcards[card_index]
                                        print(f"✅ DEBUG: Found card by index {card_index} (fallback)")
                                    else:
                                        # Second try: Match by content (front/back text)
                                        print(f"🔍 DEBUG: Index {card_index} out of range, trying content matching (fallback)")
                                        sr_front = sr_card_data.get('front', '').strip()
                                        sr_back = sr_card_data.get('back', '').strip()
                                        for idx, fc in enumerate(flashcards):
                                            if (fc.get('front', '').strip() == sr_front and 
                                                fc.get('back', '').strip() == sr_back):
                                                flashcard = fc
                                                card_index = idx
                                                print(f"✅ DEBUG: Found card by content matching at index {idx} (fallback)")
                                                break
                                        
                                        if flashcard:
                                            due_cards.append({
                                                'id': sr_card_doc.id,
                                                'front': flashcard.get('front', ''),
                                                'back': flashcard.get('back', ''),
                                                'activity_id': activity_id,
                                                'card_index': card_index,
                                                'next_review': next_review_dt.isoformat(),
                                                'interval': sr_card_data.get('interval_days', 1),
                                                'ease_factor': sr_card_data.get('ease_factor', 2.5),
                                                'repetitions': sr_card_data.get('repetitions', 0)
                                            })
                                            print(f"✅ DEBUG: Added card {sr_card_doc.id} to due cards (fallback)")
                                        else:
                                            print(f"❌ ERROR: Could not find matching flashcard for SR card {sr_card_doc.id} (fallback)")
                        except Exception as fallback_error:
                            print(f"❌ ERROR: Fallback also failed for card {sr_card_doc.id}: {fallback_error}")
                            debugger.logger.error(f"Fallback failed for card {sr_card_doc.id}: {fallback_error}")
            
            print(f"🔍 DEBUG: Total cards checked: {total_cards_checked}, Due cards: {len(due_cards)}")
            debugger.logger.info(f"Found {len(due_cards)} due cards out of {total_cards_checked} total cards in hub {hub_id}")
            
        except Exception as e:
            print(f"❌ ERROR: Failed to get due cards for hub {hub_id}: {e}")
            debugger.logger.error(f"Error getting due cards for hub {hub_id}: {e}")
            return jsonify({"success": False, "message": f"Failed to get due cards: {str(e)}"}), 500
        
        # Limit cards for this session
        session_cards = due_cards[:max_cards]
        print(f"🔍 DEBUG: Limited to {len(session_cards)} cards for this session")
        
        if not session_cards:
            print(f"ℹ️ INFO: No cards due for review in hub {hub_id}")
            debugger.logger.info(f"No cards due for review in hub {hub_id}")
            return jsonify({
                "success": True,
                "message": "No cards due for review",
                "session_id": None,
                "cards": []
            })
        
        # Create review session
        print(f"🔍 DEBUG: Creating review session with {len(session_cards)} cards")
        session_ref = db.collection('review_sessions').document()
        session = ReviewSession(
            id=session_ref.id,
            user_id=current_user.id,
            hub_id=hub_id,
            session_type=session_type,
            cards_data=[card['id'] for card in session_cards]
        )
        
        session_ref.set(session.to_dict())
        print(f"✅ SUCCESS: Created session {session.id} with {len(session_cards)} cards")
        
        debugger.log_session_event(session.id, 'created', {
            'user_id': current_user.id,
            'hub_id': hub_id,
            'cards_count': len(session_cards)
        })
        
        response_data = {
            "success": True,
            "session_id": session.id,
            "cards": session_cards,
            "total_cards": len(session_cards)
        }
        
        print(f"🔍 DEBUG: Returning session data: {len(session_cards)} cards")
        debugger.log_api_call('/api/spaced_repetition/create_session', 'POST', current_user.id, data, response_data)
        
        return jsonify(response_data)
        
    except Exception as e:
        debugger.logger.error(f"Error creating review session: {e}")
        debugger.logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"success": False, "message": "Failed to create session"}), 500


@app.route("/api/spaced_repetition/complete_session", methods=["POST"])
@login_required
def complete_review_session():
    """Complete a review session and update statistics"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        cards_reviewed = data.get('cards_reviewed', 0)
        correct_count = data.get('correct_count', 0)
        incorrect_count = data.get('incorrect_count', 0)
        
        if not session_id:
            return jsonify({"success": False, "message": "Session ID is required"}), 400
        
        # Get and update session
        session_ref = db.collection('review_sessions').document(session_id)
        session_doc = session_ref.get()
        
        if not session_doc.exists:
            return jsonify({"success": False, "message": "Session not found"}), 404
        
        session = ReviewSession.from_dict(session_doc.to_dict())
        session.cards_reviewed = cards_reviewed
        session.correct_count = correct_count
        session.incorrect_count = incorrect_count
        session.complete_session()
        
        # Update session in database
        session_ref.update(session.to_dict())
        
        # Update hub progress (XP)
        accuracy = session.calculate_accuracy()
        xp_earned = int(cards_reviewed * (accuracy / 100) * 2)  # 2 XP per card with accuracy bonus
        update_hub_progress(session.hub_id, xp_earned)
        
        return jsonify({
            "success": True,
            "message": "Session completed successfully",
            "accuracy": accuracy,
            "xp_earned": xp_earned,
            "duration_minutes": session.session_duration_minutes
        })
        
    except Exception as e:
        print(f"Error completing session: {e}")
        return jsonify({"success": False, "message": "Failed to complete session"}), 500


@app.route("/api/spaced_repetition/user_settings", methods=["GET", "POST"])
@login_required
def manage_user_settings():
    """Get or update user spaced repetition settings"""
    try:
        if request.method == "GET":
            # Get user settings
            settings_query = db.collection('user_spaced_repetition_settings').where('user_id', '==', current_user.id).limit(1)
            settings_docs = list(settings_query.stream())
            
            if settings_docs:
                settings = UserSpacedRepetitionSettings.from_dict(settings_docs[0].to_dict())
                return jsonify({
                    "success": True,
                    "settings": settings.to_dict()
                })
            else:
                # Create default settings
                settings_ref = db.collection('user_spaced_repetition_settings').document()
                default_settings = UserSpacedRepetitionSettings(
                    id=settings_ref.id,
                    user_id=current_user.id
                )
                settings_ref.set(default_settings.to_dict())
                
                return jsonify({
                    "success": True,
                    "settings": default_settings.to_dict()
                })
        
        else:  # POST
            # Update user settings
            data = request.get_json()
            
            settings_query = db.collection('user_spaced_repetition_settings').where('user_id', '==', current_user.id).limit(1)
            settings_docs = list(settings_query.stream())
            
            if settings_docs:
                settings_ref = db.collection('user_spaced_repetition_settings').document(settings_docs[0].id)
                settings_ref.update(data)
            else:
                # Create new settings
                settings_ref = db.collection('user_spaced_repetition_settings').document()
                settings = UserSpacedRepetitionSettings(
                    id=settings_ref.id,
                    user_id=current_user.id,
                    **data
                )
                settings_ref.set(settings.to_dict())
            
            return jsonify({"success": True, "message": "Settings updated successfully"})
            
    except Exception as e:
        print(f"Error managing user settings: {e}")
        return jsonify({"success": False, "message": "Failed to manage settings"}), 500


@app.route("/api/spaced_repetition/recent_sessions/<hub_id>")
@login_required
def get_recent_sessions(hub_id):
    """Get recent review sessions for a hub"""
    try:
        sessions_query = db.collection('review_sessions').where('user_id', '==', current_user.id).where('hub_id', '==', hub_id).order_by('started_at', direction=firestore.Query.DESCENDING).limit(10)
        sessions = sessions_query.stream()
        
        recent_sessions = []
        for session_doc in sessions:
            session_data = session_doc.to_dict()
            session = ReviewSession.from_dict(session_data)
            
            recent_sessions.append({
                'id': session.id,
                'started_at': session.started_at.isoformat() if session.started_at else None,
                'completed_at': session.completed_at.isoformat() if session.completed_at else None,
                'cards_reviewed': session.cards_reviewed,
                'accuracy': session.calculate_accuracy(),
                'duration_minutes': session.session_duration_minutes
            })
        
        return jsonify({
            "success": True,
            "sessions": recent_sessions
        })
        
    except Exception as e:
        print(f"Error getting recent sessions: {e}")
        return jsonify({"success": False, "message": "Failed to get sessions"}), 500


@app.route("/api/spaced_repetition/learning_progress/<hub_id>")
@login_required
def get_learning_progress(hub_id):
    """Get detailed learning progress for a hub"""
    try:
        # Get all activities in this hub
        activities_query = db.collection('activities').where('hub_id', '==', hub_id).where('type', '==', 'Flashcards')
        activities = activities_query.stream()
        
        progress_data = {
            'total_cards': 0,
            'new_cards': 0,
            'learning_cards': 0,
            'review_cards': 0,
            'mature_cards': 0,
            'overdue_cards': 0,
            'cards_by_difficulty': {'easy': 0, 'medium': 0, 'hard': 0},
            'average_ease_factor': 0,
            'retention_rate': 0
        }
        
        total_ease_factor = 0
        total_reviews = 0
        successful_reviews = 0
        
        for activity_doc in activities:
            activity_id = activity_doc.id
            
            # Get spaced repetition cards for this activity
            sr_cards_query = db.collection('spaced_repetition_cards').where('activity_id', '==', activity_id)
            sr_cards = sr_cards_query.stream()
            
            for sr_card_doc in sr_cards:
                sr_card_data = sr_card_doc.to_dict()
                sr_card = SpacedRepetitionCard.from_dict(sr_card_data)
                
                progress_data['total_cards'] += 1
                total_ease_factor += sr_card.ease_factor
                
                # Categorize cards
                if sr_card.repetitions == 0:
                    progress_data['new_cards'] += 1
                elif sr_card.repetitions < 3:
                    progress_data['learning_cards'] += 1
                elif sr_card.interval_days >= 21:
                    progress_data['mature_cards'] += 1
                else:
                    progress_data['review_cards'] += 1
                
                # Check if overdue
                if sr_card.next_review and datetime.now(timezone.utc) > sr_card.next_review:
                    progress_data['overdue_cards'] += 1
                
                # Count by difficulty
                progress_data['cards_by_difficulty'][sr_card.difficulty] += 1
                
                # Calculate retention rate
                if sr_card.repetitions > 0:
                    total_reviews += sr_card.repetitions
                    successful_reviews += max(0, sr_card.repetitions - 1)  # Assume last review was successful
        
        # Calculate averages
        if progress_data['total_cards'] > 0:
            progress_data['average_ease_factor'] = total_ease_factor / progress_data['total_cards']
        
        if total_reviews > 0:
            progress_data['retention_rate'] = (successful_reviews / total_reviews) * 100
        
        return jsonify({
            "success": True,
            "progress": progress_data
        })
        
    except Exception as e:
        print(f"Error getting learning progress: {e}")
        return jsonify({"success": False, "message": "Failed to get progress"}), 500


# ==============================================================================
# SPACED REPETITION SYSTEM - PHASE 1 IMPLEMENTATION
# ==============================================================================

@app.route("/admin/repair_spaced_repetition_indices", methods=["POST"])
@login_required
def repair_all_spaced_repetition_indices():
    """
    Repair card indices for all activities by matching content.
    This fixes the "Card index out of range" errors.
    """
    try:
        print("🔧 REPAIR: Starting global index repair")
        
        # Get all flashcard activities
        activities_query = db.collection('activities').where('type', '==', 'Flashcards')
        activities = list(activities_query.stream())
        
        repaired_activities = 0
        recovered_activities = 0
        
        for activity_doc in activities:
            activity_id = activity_doc.id
            print(f"🔧 REPAIR: Processing activity {activity_id}")
            
            # First try to recover missing flashcards
            if recover_missing_flashcards(activity_id):
                recovered_activities += 1
            
            # Then repair indices
            if repair_spaced_repetition_indices(activity_id):
                repaired_activities += 1
        
        print(f"🔧 REPAIR: Completed global repair for {repaired_activities} activities")
        print(f"🔄 RECOVERY: Recovered flashcards for {recovered_activities} activities")
        
        return jsonify({
            "success": True,
            "message": f"Successfully repaired {repaired_activities} activities and recovered {recovered_activities} activities",
            "repaired_activities": repaired_activities,
            "recovered_activities": recovered_activities
        })
        
    except Exception as e:
        print(f"Error repairing spaced repetition indices: {e}")
        return jsonify({"success": False, "message": "Failed to repair indices"}), 500

@app.route("/admin/recover_missing_flashcards", methods=["POST"])
@login_required
def recover_all_missing_flashcards():
    """
    Recover missing flashcards from spaced repetition data.
    This fixes activities that have 0 cards but have SR cards.
    """
    try:
        print("🔄 RECOVERY: Starting global flashcard recovery")
        
        # Get all flashcard activities
        activities_query = db.collection('activities').where('type', '==', 'Flashcards')
        activities = list(activities_query.stream())
        
        recovered_activities = 0
        total_recovered_cards = 0
        
        for activity_doc in activities:
            activity_id = activity_doc.id
            activity_data = activity_doc.to_dict()
            current_cards = activity_data.get('data', {}).get('cards', [])
            
            # Only process activities with missing cards
            if len(current_cards) == 0:
                print(f"🔄 RECOVERY: Processing activity {activity_id} (0 cards)")
                
                # Get spaced repetition cards for this activity
                sr_cards_query = db.collection('spaced_repetition_cards').where('activity_id', '==', activity_id)
                sr_cards = list(sr_cards_query.stream())
                
                if sr_cards:
                    if recover_missing_flashcards(activity_id):
                        recovered_activities += 1
                        total_recovered_cards += len(sr_cards)
                        print(f"✅ RECOVERY: Recovered {len(sr_cards)} cards for activity {activity_id}")
                else:
                    print(f"ℹ️ RECOVERY: No SR cards found for activity {activity_id}")
        
        print(f"🔄 RECOVERY: Completed recovery for {recovered_activities} activities")
        print(f"🔄 RECOVERY: Total cards recovered: {total_recovered_cards}")
        
        return jsonify({
            "success": True,
            "message": f"Successfully recovered {total_recovered_cards} cards from {recovered_activities} activities",
            "recovered_activities": recovered_activities,
            "total_recovered_cards": total_recovered_cards
        })
        
    except Exception as e:
        print(f"Error recovering missing flashcards: {e}")
        return jsonify({"success": False, "message": "Failed to recover flashcards"}), 500

@app.route("/admin/migrate_flashcards_to_spaced_repetition", methods=["POST"])
@login_required
def migrate_all_flashcards_to_spaced_repetition():
    """
    Migration script to convert existing flashcards to spaced repetition system.
    This should be run once to migrate all existing flashcard activities.
    """
    try:
        # Get all flashcard activities
        activities_query = db.collection('activities').where('type', '==', 'Flashcards')
        activities = activities_query.stream()
        
        migrated_count = 0
        skipped_count = 0
        
        for activity_doc in activities:
            activity_data = activity_doc.to_dict()
            activity_id = activity_doc.id
            
            # Check if already migrated
            existing_sr_cards = db.collection('spaced_repetition_cards').where('activity_id', '==', activity_id).limit(1).stream()
            if list(existing_sr_cards):
                skipped_count += 1
                continue
            
            cards = activity_data.get('data', {}).get('cards', [])
            if not cards:
                skipped_count += 1
                continue
            
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
                    interval_days=1,  # Start with 1 day interval
                    repetitions=0,
                    difficulty='medium'
                )
                batch.set(sr_card_ref, sr_card.to_dict())
            
            batch.commit()
            migrated_count += 1
            print(f"Migrated activity {activity_id} with {len(cards)} cards")
        
        return jsonify({
            "success": True,
            "message": f"Migration completed. Migrated {migrated_count} activities, skipped {skipped_count} activities.",
            "migrated_count": migrated_count,
            "skipped_count": skipped_count
        })
        
    except Exception as e:
        print(f"Error during migration: {e}")
        return jsonify({"success": False, "message": f"Migration failed: {str(e)}"}), 500


@app.route("/api/spaced_repetition/due_cards/<hub_id>")
@login_required
def get_due_cards(hub_id):
    """Get cards that are due for review in a specific hub"""
    try:
        print(f"🔍 DEBUG: Getting due cards for hub {hub_id}")
        
        # Check if Firebase is available
        if not db:
            print(f"❌ ERROR: Firebase not available for hub {hub_id}")
            return jsonify({
                "success": False,
                "message": "Firebase not configured",
                "firebase_available": False,
                "total_due": 0,
                "due_cards": []
            })
        
        # Get all activities in this hub
        activities_query = db.collection('activities').where('hub_id', '==', hub_id).where('type', '==', 'Flashcards')
        activities = list(activities_query.stream())
        print(f"🔍 DEBUG: Found {len(activities)} flashcard activities in hub {hub_id}")
        
        due_cards = []
        total_cards_checked = 0
        
        for activity_doc in activities:
            activity_id = activity_doc.id
            print(f"🔍 DEBUG: Checking activity {activity_id}")
            
            # Get spaced repetition cards for this activity
            sr_cards_query = db.collection('spaced_repetition_cards').where('activity_id', '==', activity_id)
            sr_cards = list(sr_cards_query.stream())
            print(f"🔍 DEBUG: Found {len(sr_cards)} spaced repetition cards in activity {activity_id}")
            
            for sr_card_doc in sr_cards:
                sr_card_data = sr_card_doc.to_dict()
                sr_card = SpacedRepetitionCard.from_dict(sr_card_data)
                total_cards_checked += 1
                
                if sr_card.is_due():
                    due_cards.append({
                        'id': sr_card.id,
                        'front': sr_card.front,
                        'back': sr_card.back,
                        'activity_id': sr_card.activity_id,
                        'card_index': sr_card.card_index,
                        'ease_factor': sr_card.ease_factor,
                        'interval_days': sr_card.interval_days,
                        'repetitions': sr_card.repetitions,
                        'difficulty': sr_card.difficulty
                    })
                    print(f"🔍 DEBUG: Card {sr_card.id} is due (interval: {sr_card.interval_days}d, reps: {sr_card.repetitions})")
                else:
                    next_review = sr_card.next_review.strftime("%Y-%m-%d %H:%M") if sr_card.next_review else "Never"
                    print(f"🔍 DEBUG: Card {sr_card.id} not due (next review: {next_review})")
        
        print(f"✅ SUCCESS: Found {len(due_cards)} due cards out of {total_cards_checked} total cards")
        
        return jsonify({
            "success": True,
            "due_cards": due_cards,
            "total_due": len(due_cards)
        })
        
    except Exception as e:
        print(f"Error getting due cards: {e}")
        return jsonify({"success": False, "message": "Failed to get due cards"}), 500


@app.route("/api/spaced_repetition/review_card", methods=["POST"])
@login_required
@debug_logger
def review_card():
    """Process a card review and update its spaced repetition data"""
    try:
        data = request.get_json()
        card_id = data.get('card_id')
        quality_rating = data.get('quality_rating')  # 0=again, 1=hard, 2=good, 3=easy
        
        # Enhanced debug logging
        print(f"🔍 DEBUG: Processing card review for user {current_user.id}")
        print(f"🔍 DEBUG: Card ID: {card_id}, Quality rating: {quality_rating}")
        debugger.log_api_call('/api/spaced_repetition/review_card', 'POST', current_user.id, data)
        log_user_action(current_user.id, 'review_card', {'card_id': card_id, 'rating': quality_rating})
        
        if not card_id or quality_rating is None:
            print(f"❌ ERROR: User {current_user.id} attempted card review with missing data")
            debugger.logger.warning(f"User {current_user.id} attempted card review with missing data")
            return jsonify({"success": False, "message": "Missing card_id or quality_rating"}), 400
        
        # Get the card
        print(f"🔍 DEBUG: Fetching card {card_id} from database")
        card_ref = db.collection('spaced_repetition_cards').document(card_id)
        card_doc = card_ref.get()
        
        if not card_doc.exists:
            print(f"❌ ERROR: Card {card_id} not found for user {current_user.id}")
            debugger.logger.error(f"Card {card_id} not found for user {current_user.id}")
            return jsonify({"success": False, "message": "Card not found"}), 404
        
        # Update the card with new review data
        sr_card = SpacedRepetitionCard.from_dict(card_doc.to_dict())
        old_interval = sr_card.interval_days
        old_ease_factor = sr_card.ease_factor
        old_repetitions = sr_card.repetitions
        
        print(f"🔍 DEBUG: Card state before review: interval={old_interval}d, ease={old_ease_factor:.2f}, reps={old_repetitions}")
        
        sr_card.calculate_next_review(quality_rating)
        
        print(f"🔍 DEBUG: Card state after review: interval={sr_card.interval_days}d, ease={sr_card.ease_factor:.2f}, reps={sr_card.repetitions}")
        
        # Log algorithm calculation
        debugger.log_algorithm_calculation(
            card_id, 
            old_interval, 
            quality_rating, 
            sr_card.interval_days, 
            sr_card.ease_factor
        )
        
        # Save updated card
        card_ref.update(sr_card.to_dict())
        print(f"✅ SUCCESS: Card {card_id} updated: {old_interval}d -> {sr_card.interval_days}d")
        
        debugger.logger.info(f"Card {card_id} updated: {old_interval}d -> {sr_card.interval_days}d")
        
        response_data = {
            "success": True,
            "message": "Card reviewed successfully",
            "next_review": sr_card.next_review.isoformat() if sr_card.next_review else None,
            "interval_days": sr_card.interval_days,
            "ease_factor": sr_card.ease_factor
        }
        
        debugger.log_api_call('/api/spaced_repetition/review_card', 'POST', current_user.id, data, response_data)
        
        return jsonify(response_data)
        
    except Exception as e:
        debugger.logger.error(f"Error reviewing card: {e}")
        debugger.logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"success": False, "message": "Failed to review card"}), 500


@app.route("/api/spaced_repetition/stats/<hub_id>")
@login_required
def get_spaced_repetition_stats(hub_id):
    """Get spaced repetition statistics for a hub"""
    try:
        # Get all activities in this hub
        activities_query = db.collection('activities').where('hub_id', '==', hub_id).where('type', '==', 'Flashcards')
        activities = activities_query.stream()
        
        total_cards = 0
        due_cards = 0
        new_cards = 0
        mature_cards = 0
        
        for activity_doc in activities:
            activity_id = activity_doc.id
            
            # Get spaced repetition cards for this activity
            sr_cards_query = db.collection('spaced_repetition_cards').where('activity_id', '==', activity_id)
            sr_cards = sr_cards_query.stream()
            
            for sr_card_doc in sr_cards:
                sr_card_data = sr_card_doc.to_dict()
                sr_card = SpacedRepetitionCard.from_dict(sr_card_data)
                
                total_cards += 1
                
                if sr_card.is_due():
                    due_cards += 1
                
                if sr_card.repetitions == 0:
                    new_cards += 1
                elif sr_card.interval_days >= 21:  # Cards with 21+ day intervals are considered mature
                    mature_cards += 1
        
        return jsonify({
            "success": True,
            "stats": {
                "total_cards": total_cards,
                "due_cards": due_cards,
                "new_cards": new_cards,
                "mature_cards": mature_cards
            }
        })
        
    except Exception as e:
        print(f"Error getting stats: {e}")
        return jsonify({"success": False, "message": "Failed to get stats"}), 500


@app.route("/api/spaced_repetition/system_health")
@login_required
def get_system_health():
    """Get system health and debugging information"""
    try:
        # Check if Firebase is available
        if not db:
            return jsonify({
                "success": False,
                "message": "Firebase not configured",
                "firebase_available": False,
                "system_status": "development_mode",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        
        # Get system health metrics
        health_metrics = log_system_health()
        
        # Get recent errors
        recent_errors = debugger._get_recent_errors()
        
        # Get log file information
        log_files = debugger._get_log_file_info()
        
        # Get spaced repetition statistics
        try:
            # Get total cards across all hubs - more efficient approach
            total_cards = 0
            total_due_cards = 0
            
            # Get user's hubs first
            hubs_query = db.collection('hubs').where('user_id', '==', current_user.id)
            hubs = hubs_query.stream()
            
            hub_ids = [hub.id for hub in hubs]
            
            if hub_ids:
                # Get all activities for user's hubs
                activities_query = db.collection('activities').where('hub_id', 'in', hub_ids).where('type', '==', 'Flashcards')
                activities = list(activities_query.stream())
                activity_ids = [activity.id for activity in activities]
                
                if activity_ids:
                    # Count total cards for these activities
                    cards_query = db.collection('spaced_repetition_cards').where('activity_id', 'in', activity_ids)
                    total_cards = len(list(cards_query.stream()))
                    
                    # Count due cards for these activities
                    due_query = db.collection('spaced_repetition_cards').where('activity_id', 'in', activity_ids).where('next_review', '<=', datetime.now(timezone.utc))
                    total_due_cards = len(list(due_query.stream()))
        
        except Exception as e:
            debugger.logger.warning(f"Error getting spaced repetition stats: {e}")
            total_cards = 0
            total_due_cards = 0
        
        health_data = {
            "success": True,
            "firebase_available": True,
            "system_health": health_metrics,
            "spaced_repetition_stats": {
                "total_cards": total_cards,
                "due_cards": total_due_cards,
                "system_status": "operational" if total_cards > 0 else "no_cards"
            },
            "recent_errors": recent_errors,
            "log_files": log_files,
            "debugging_enabled": True,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        debugger.log_api_call('/api/spaced_repetition/system_health', 'GET', current_user.id, None, health_data)
        
        return jsonify(health_data)
        
    except Exception as e:
        debugger.logger.error(f"Error getting system health: {e}")
        return jsonify({
            "success": False,
            "message": "Failed to get system health",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), 500


@app.route("/api/spaced_repetition/debug_report")
@login_required
def generate_debug_report():
    """Generate comprehensive debug report"""
    try:
        report = debugger.create_debug_report()
        
        debugger.log_api_call('/api/spaced_repetition/debug_report', 'GET', current_user.id, None, report)
        
        return jsonify({
            "success": True,
            "report": report,
            "message": "Debug report generated successfully"
        })
        
    except Exception as e:
        debugger.logger.error(f"Error generating debug report: {e}")
        return jsonify({
            "success": False,
            "message": "Failed to generate debug report",
            "error": str(e)
        }), 500


@app.route("/spaced_repetition/settings")
@login_required
def spaced_repetition_settings():
    """Serve the spaced repetition settings page"""
    try:
        debugger.log_api_call('/spaced_repetition/settings', 'GET', current_user.id)
        return render_template('spaced_repetition_settings.html')
    except Exception as e:
        debugger.logger.error(f"Error serving settings page: {e}")
        return render_template('error.html', error="Failed to load settings page"), 500


if __name__ == '__main__':
    # Initialize debugging system
    print("Initializing Spaced Repetition Debugging System...")
    debugger.logger.info("Starting Flask application with spaced repetition system")
    
    # Log system startup
    startup_info = {
        'python_version': sys.version,
        'platform': sys.platform,
        'working_directory': os.getcwd(),
        'timestamp': datetime.now(timezone.utc).isoformat()
    }
    debugger.logger.info(f"Application startup: {json.dumps(startup_info)}")
    
    app.run(debug=True)