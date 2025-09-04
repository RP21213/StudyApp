# ==============================================================================
# 1. SETUP & IMPORTS
# ==============================================================================
import os
import io
import json
import markdown
import re
import random
from datetime import datetime, timezone
from flask import Flask, request, render_template, redirect, url_for, Response, send_file, flash, jsonify
from werkzeug.security import check_password_hash, generate_password_hash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from openai import OpenAI
import PyPDF2
import ast
from fpdf import FPDF
import csv
from flask import Response
from datetime import datetime, timedelta, timezone
import threading
import firebase_admin
from firebase_admin import credentials, firestore, storage
import stripe
import base64 # Make sure this import is at the top with your others
import json   # Make sure this import is at the top with your others
from models import Hub, Activity, Note, Lecture, StudySession, Folder, Notification, Assignment, CalendarEvent, User, SharedFolder

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
except Exception as e:
    print(f"Firebase initialization failed. Error: {e}")

db = firestore.client()
bucket = storage.bucket()

# --- Flask App and OpenAI Client Initialization ---
app = Flask(__name__)
# IMPORTANT: Set a secret key for session management in your environment variables
app.config['SECRET_KEY'] = os.getenv("FLASK_SECRET_KEY", "a-default-secret-key-for-development")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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

# ==============================================================================
# 2. CORE UTILITY & HELPER FUNCTIONS
# ==============================================================================

def _get_user_stats(user_id):
    """Helper function to calculate stats for a given user."""
    hubs_ref = db.collection('hubs').where('user_id', '==', user_id).stream()
    hubs_list = [Hub.from_dict(doc.to_dict()) for doc in hubs_ref]
    
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
    You are an expert AI creating an interactive HTML study guide from the provided text.
    Your response MUST be a single block of well-formed HTML.

    Follow these rules precisely:
    1.  **Structure:** Use standard HTML tags like `<h1>`, `<h2>`, `<p>`, `<ul>`, `<li>`, and `<strong>`.
    2.  **Keywords:** For every important keyword or key term, wrap it in a `<span>` with `class="keyword"` and a `title` attribute containing its concise definition.
        - **Example:** `<span class="keyword" title="A resource with economic value that is expected to provide a future benefit.">Asset</span>`
    3.  **Formulas:** For every mathematical or scientific formula, wrap it in a `<span>` with `class="formula"` and a `data-formula` attribute containing the exact formula as a string.
        - **Example:** `<span class="formula" data-formula="Assets = Liabilities + Equity">Assets = Liabilities + Equity</span>`

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

def generate_flashcards_from_text(text):
    """Generates flashcards from text using the AI."""
    prompt = f"""
    Based on the following raw text, create a set of flashcards. Identify key terms, concepts, and questions.
    
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
            {"role": "system", "content": "You are a helpful assistant that creates flashcards from study materials. Separate each flashcard with '---'."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def generate_quiz_from_text(text):
    """Generates a quiz in JSON format from text using the AI."""
    prompt = f"""
    You are an expert educator. Your task is to create a 10-question practice quiz from the text below.

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

# --- Homepage and Auth Routes ---
@app.route("/")
def index():
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

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        users_ref = db.collection('users').where('email', '==', email).limit(1).stream()
        if list(users_ref):
            flash('Email address already exists.')
            return redirect(url_for('signup'))
        
        password_hash = generate_password_hash(password, method='pbkdf2:sha256')
        
        user_ref = db.collection('users').document()
        new_user = User(
            id=user_ref.id, 
            email=email, 
            password_hash=password_hash
        )
        user_ref.set(new_user.to_dict())

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
        activities_query = db.collection('activities').where('hub_id', '==', hub.id).where('status', '==', 'graded').stream()
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
    trending_folders = []
    total_shared_count = 0
    if current_user.subscription_tier in ['pro', 'admin']:
        
        # --- Get total count for gamification ---
        all_shared_docs = db.collection('shared_folders').stream()
        total_shared_count = len(list(all_shared_docs))

        # --- Base query for main list ---
        query = db.collection('shared_folders')
        sort_by = request.args.get('sort', 'created_at')
        if sort_by == 'likes':
            query = query.order_by('likes', direction=firestore.Query.DESCENDING)
        else:
            query = query.order_by('created_at', direction=firestore.Query.DESCENDING)
        
        shared_folders_docs = query.limit(20).stream()
        shared_folders = [SharedFolder.from_dict(doc.to_dict()) for doc in shared_folders_docs]
        
        # --- Get trending folders (most liked) ---
        trending_query = db.collection('shared_folders').order_by('likes', direction=firestore.Query.DESCENDING).limit(3).stream()
        trending_folders_raw = [SharedFolder.from_dict(doc.to_dict()) for doc in trending_query]

        # --- Hydrate folder data (get owner info and item details) ---
        all_folders_to_hydrate = shared_folders + trending_folders_raw
        owner_ids = list(set(sf.owner_id for sf in all_folders_to_hydrate))
        original_folder_ids = list(set(sf.original_folder_id for sf in all_folders_to_hydrate))
        
        users_data = {}
        if owner_ids:
            user_docs = db.collection('users').where('id', 'in', owner_ids).stream()
            users_data = {user.to_dict()['id']: user.to_dict() for user in user_docs}

        original_folders_data = {}
        if original_folder_ids:
            folder_docs = db.collection('folders').where('id', 'in', original_folder_ids).stream()
            original_folders_data = {f.to_dict()['id']: f.to_dict() for f in folder_docs}

        hydrated_cache = {}

        def _hydrate_folder_info(sf):
            if sf.id in hydrated_cache:
                return hydrated_cache[sf.id]

            owner_info = users_data.get(sf.owner_id)
            original_folder_info = original_folders_data.get(sf.original_folder_id)
            
            item_count = 0
            folder_type = "Pack"
            if original_folder_info:
                items = original_folder_info.get('items', [])
                item_count = len(items)
                types = set(item['type'] for item in items)
                if len(types) == 1:
                    folder_type = types.pop().capitalize()
                elif 'note' in types:
                    folder_type = "Notes"
            
            hydrated_info = {
                'folder': sf,
                'owner_name': owner_info.get('display_name', 'Unknown User') if owner_info else 'Unknown User',
                'owner_avatar': owner_info.get('avatar_url', 'default_avatar.png') if owner_info else 'default_avatar.png',
                'item_count': item_count,
                'folder_type': folder_type,
            }
            hydrated_cache[sf.id] = hydrated_info
            return hydrated_info

        shared_folders_hydrated = [_hydrate_folder_info(sf) for sf in shared_folders]
        trending_folders = [_hydrate_folder_info(sf) for sf in trending_folders_raw]


    # --- NEW: Fetch all of the user's own folders for the share modal ---
    all_user_folders = []
    for hub in hubs_list:
        folders_query = db.collection('folders').where('hub_id', '==', hub.id).stream()
        for folder_doc in folders_query:
            folder = Folder.from_dict(folder_doc.to_dict())
            all_user_folders.append({
                "id": folder.id,
                "name": folder.name,
                "hub_name": hub.name
            })

    # --- FIX: Create a separate, JSON-serializable list of hubs for the JavaScript part ---
    hubs_for_json = [hub.to_dict() for hub in hubs_list]

    return render_template(
        "dashboard.html", 
        hubs=hubs_list,
        hubs_for_json=hubs_for_json,
        total_study_hours=total_study_hours,
        longest_streak=longest_streak,
        quiz_scores_json=json.dumps(all_quiz_scores),
        weak_topics=weak_topics,
        shared_folders=shared_folders_hydrated,
        trending_folders=trending_folders,
        total_shared_count=total_shared_count,
        all_user_folders=all_user_folders
    )


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
        return redirect(url_for('profile'))

    stats = _get_user_stats(current_user.id)
    return render_template('profile.html', stats=stats)

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

    activities_query = db.collection('activities').where('hub_id', '==', hub_id).stream()
    all_activities = [Activity.from_dict(doc.to_dict()) for doc in activities_query]
    
    notes_query = db.collection('notes').where('hub_id', '==', hub_id).stream()
    all_notes = [Note.from_dict(note.to_dict()) for note in notes_query]
    
    sessions_query = db.collection('sessions').where('hub_id', '==', hub_id).order_by('created_at', direction=firestore.Query.DESCENDING).stream()
    all_sessions = [StudySession.from_dict(doc.to_dict()) for doc in sessions_query]

    folders_query = db.collection('folders').where('hub_id', '==', hub_id).order_by('created_at', direction=firestore.Query.DESCENDING).stream()
    all_folders = [Folder.from_dict(doc.to_dict()) for doc in folders_query]

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
            elif doc_type in ['quiz', 'flashcards'] and doc_id in activities_map:
                item = activities_map[doc_id]
            
            if item:
                hydrated_items.append(item)
        
        folder.hydrated_items = hydrated_items

    all_flashcards = [activity for activity in all_activities if activity.type == 'Flashcards']
    graded_activities = [activity for activity in all_activities if activity.status == 'graded']
    
    # --- FIX: Create a list of all quizzes/exams, not just graded ones ---
    all_quizzes_and_exams = [
        activity for activity in all_activities 
        if 'Quiz' in activity.type or 'Exam' in activity.type
    ]

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

    return render_template(
        "hub.html", 
        hub=hub,
        # Pass the new persistent data to the template
        total_xp=total_xp,
        streak_days=streak_days,
        level_data=level_data,
        # Pass all other data as before
        all_notes=all_notes, 
        all_sessions=all_sessions, 
        all_flashcards=all_flashcards, 
        all_quizzes_and_exams=all_quizzes_and_exams,
        topic_mastery=topic_mastery,
        spotlight=spotlight,
        today_xp=today_xp,
        all_folders=all_folders,
        yesterday_activities=yesterday_activities,
        notifications=notifications,
        unread_notifications_count=unread_notifications_count
    )

# ==============================================================================
# 5. NEW STRIPE & SUBSCRIPTION ROUTES
# ==============================================================================

@app.route('/create-checkout-session', methods=['POST'])
@login_required
def create_checkout_session():
    price_id = request.form.get('price_id')
    try:
        checkout_session = stripe.checkout.Session.create(
            line_items=[
                {
                    'price': price_id,
                    'quantity': 1,
                },
            ],
            mode='subscription',
            success_url=YOUR_DOMAIN + '/dashboard?session_id={CHECKOUT_SESSION_ID}',
            cancel_url=YOUR_DOMAIN + '/',
            customer_email=current_user.email,
            # Pass user ID to identify the user in webhook
            metadata={'user_id': current_user.id}
        )
    except Exception as e:
        return str(e)

    return redirect(checkout_session.url, code=303)

@app.route('/stripe-webhook', methods=['POST'])
def stripe_webhook():
    payload = request.data
    sig_header = request.headers.get('Stripe-Signature')
    endpoint_secret = os.getenv("STRIPE_ENDPOINT_SECRET")
    event = None

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, endpoint_secret
        )
    except ValueError as e:
        # Invalid payload
        return 'Invalid payload', 400
    except stripe.error.SignatureVerificationError as e:
        # Invalid signature
        return 'Invalid signature', 400

    # Handle the checkout.session.completed event
    if event['type'] == 'checkout.session.completed':
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

# ==============================================================================
# 4. FLASK ROUTES
# ==============================================================================

@app.route("/")
def home():
    return redirect(url_for('dashboard'))

@app.route("/hub/<hub_id>/delete")
@login_required
def delete_hub(hub_id):
    try:
        blobs = bucket.list_blobs(prefix=f"hubs/{hub_id}/")
        for blob in blobs: blob.delete()
        collections_to_delete = ['notes', 'activities', 'lectures', 'notifications']
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
    note_doc = db.collection('notes').document(note_id).get()
    if not note_doc.exists:
        return "Note not found", 404
    note = Note.from_dict(note_doc.to_dict())
    hub_doc = db.collection('hubs').document(note.hub_id).get()
    hub_name = hub_doc.to_dict().get('name') if hub_doc.exists else "Hub"

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
        flash("No file part in the request.", "error")
        return redirect(url_for('hub_page', hub_id=hub_id))
    file = request.files['pdf']
    if file.filename == '':
        flash("No file selected.", "error")
        return redirect(url_for('hub_page', hub_id=hub_id))
    if file and file.filename.lower().endswith(".pdf"):
        filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', file.filename)
        file_path = f"hubs/{hub_id}/{filename}"
        blob = bucket.blob(file_path)
        blob.upload_from_file(file, content_type='application/pdf')
        file.seek(0, os.SEEK_END)
        file_info = {'name': filename, 'path': file_path, 'size': file.tell()}
        db.collection('hubs').document(hub_id).update({'files': firestore.ArrayUnion([file_info])})
        flash(f"File '{filename}' uploaded successfully!", "success")
    else:
        flash("Invalid file type. Please upload a PDF.", "error")
    return redirect(url_for('hub_page', hub_id=hub_id))

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
    link = url_for('take_lecture_quiz', activity_id=new_activity.id) # FIX: Changed quiz_id to activity_id
    new_notification = Notification(id=notification_ref.id, hub_id=hub_id, message=message, link=link)
    batch.set(notification_ref, new_notification.to_dict())
    batch.commit()

    flash(f"Generated a special quiz to help you with {topic}!", "success")
    return redirect(url_for('take_lecture_quiz', activity_id=new_activity.id)) # FIX: Changed quiz_id to activity_id

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

# --- FIX: Standardized route to use <activity_id> ---
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
        all_hubs = db.collection('hubs').stream()
        hubs_to_delete = list(all_hubs)
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
# 7. GUIDED WORKFLOW ROUTES (NEW SECTION)
# ==============================================================================

# --- NEW: This is the single route that handles the new workflow form ---
@app.route("/hub/<hub_id>/generate_workflow_folder", methods=["POST"])
@login_required
def generate_workflow_folder(hub_id):
    try:
        data = request.get_json()
        workflow_type = data.get('workflow_type')
        selected_tools = data.get('selected_tools', [])
        selected_files = data.get('selected_files', [])

        # --- ACCESS CONTROL ---
        if current_user.subscription_tier == 'free':
            pro_tools_requested = [tool for tool in selected_tools if tool not in ['notes', 'flashcards']]
            if pro_tools_requested:
                # Capitalize for display
                pro_tools_str = ', '.join([t.capitalize() for t in pro_tools_requested])
                return jsonify({"success": False, "message": f"Creating {pro_tools_str} is a Pro feature. Please upgrade to generate these resources."}), 403

        if not all([workflow_type, selected_tools, selected_files]):
            return jsonify({"success": False, "message": "Missing required data."}), 400

        hub_text = get_text_from_hub_files(selected_files)
        if not hub_text:
            return jsonify({"success": False, "message": "Could not extract text from files."}), 500

        first_file_name = os.path.basename(selected_files[0]).replace('.pdf', '')
        folder_name = f"Exam Pack for {first_file_name}" if workflow_type == 'exam' else f"Revision Pack for {first_file_name}"
        
        batch = db.batch()
        folder_items = []
        folder_ref = db.collection('folders').document()

        # --- Generate selected resources ---
        if 'notes' in selected_tools:
            interactive_html = generate_interactive_notes_html(hub_text)
            note_ref = db.collection('notes').document()
            new_note = Note(id=note_ref.id, hub_id=hub_id, title=f"Notes for {first_file_name}", content_html=interactive_html)
            batch.set(note_ref, new_note.to_dict())
            folder_items.append({'id': note_ref.id, 'type': 'note'})

        if 'flashcards' in selected_tools:
            flashcards_raw = generate_flashcards_from_text(hub_text)
            flashcards_parsed = parse_flashcards(flashcards_raw)
            if flashcards_parsed:
                fc_ref = db.collection('activities').document()
                new_fc = Activity(id=fc_ref.id, hub_id=hub_id, type='Flashcards', title=f"Flashcards for {first_file_name}", data={'cards': flashcards_parsed}, status='completed')
                batch.set(fc_ref, new_fc.to_dict())
                folder_items.append({'id': fc_ref.id, 'type': 'flashcards'})

        if 'quiz' in selected_tools or 'exam' in selected_tools:
            quiz_json = generate_quiz_from_text(hub_text) # Generates a 10-q quiz
            quiz_data = safe_load_json(quiz_json)
            if quiz_data.get('questions'):
                quiz_ref = db.collection('activities').document()
                quiz_title = "Mock Exam" if 'exam' in selected_tools else "Practice Quiz"
                new_quiz = Activity(id=quiz_ref.id, hub_id=hub_id, type='Quiz', title=f"{quiz_title} for {first_file_name}", data=quiz_data)
                batch.set(quiz_ref, new_quiz.to_dict())
                folder_items.append({'id': quiz_ref.id, 'type': 'quiz'})
        
        if 'cheatsheet' in selected_tools:
            cheat_sheet_json_str = generate_cheat_sheet_json(hub_text)
            note_ref = db.collection('notes').document()
            new_note = Note(id=note_ref.id, hub_id=hub_id, title=f"Cheat Sheet for {first_file_name}", content_html=cheat_sheet_json_str)
            batch.set(note_ref, new_note.to_dict())
            folder_items.append({'id': note_ref.id, 'type': 'note'})
            
        if 'heatmap' in selected_tools:
            # Note: Heatmap is a view, not a savable asset. We'll skip adding it to the folder.
            # You could potentially save the JSON data as a note if desired.
            pass

        if 'analyse' in selected_tools:
            # Note: Analysis is a view, not a savable asset.
            pass

        # --- Create the folder and notification ---
        new_folder = Folder(id=folder_ref.id, hub_id=hub_id, name=folder_name, items=folder_items)
        batch.set(folder_ref, new_folder.to_dict())
        
        notification_ref = db.collection('notifications').document()
        message = f"Your new resource folder '{folder_name}' is ready."
        link = url_for('hub_page', hub_id=hub_id, _anchor='folders')
        new_notification = Notification(id=notification_ref.id, hub_id=hub_id, message=message, link=link)
        batch.set(notification_ref, new_notification.to_dict())

        batch.commit()
        return jsonify({"success": True, "message": f"Successfully created folder '{folder_name}'!"})

    except Exception as e:
        print(f"Error in generate_workflow_folder: {e}")
        return jsonify({"success": False, "message": str(e)}), 500


# --- NEW: ROUTE TO HANDLE INDIVIDUAL TOOL GENERATION ---
@app.route("/hub/<hub_id>/run_async_tool", methods=["POST"])
@login_required
def run_async_tool(hub_id):
    data = request.get_json()
    tool = data.get('tool')
    selected_files = data.get('selected_files')

    # --- ACCESS CONTROL ---
    if current_user.subscription_tier == 'free' and tool not in ['notes', 'flashcards']:
        return jsonify({"success": False, "message": f"The '{tool.capitalize()}' tool is a Pro feature. Please upgrade to use it."}), 403
    
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
            redirect_url = url_for('take_lecture_quiz', activity_id=quiz_ref.id) # FIX: Changed quiz_id to activity_id

        elif tool == 'mindmap':
            mind_map_json_str = generate_mind_map_json(hub_text)
            note_ref = db.collection('notes').document()
            new_note = Note(id=note_ref.id, hub_id=hub_id, title=f"Mind Map for {first_file_name}", content_html=mind_map_json_str)
            note_ref.set(new_note.to_dict())
            redirect_url = url_for('view_note', note_id=note_ref.id)
            
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
        db.collection('folders').document(folder_id).delete()
        return jsonify({"success": True})
    except Exception as e:
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
# 10. COMMUNITY ROUTES (NEW SECTION)
# ==============================================================================
@app.route("/folder/share", methods=["POST"])
@login_required
def share_folder():
    if current_user.subscription_tier not in ['pro', 'admin']:
        flash("You must be a Pro member to share folders with the community.", "error")
        return redirect(url_for('dashboard'))

    folder_id = request.form.get('folder_id')
    description = request.form.get('description')
    tags_raw = request.form.get('tags', '')
    
    if not all([folder_id, description]):
        flash("Folder and description are required.", "error")
        return redirect(url_for('dashboard', _anchor='community'))

    try:
        # 1. Fetch the original folder to ensure it exists and belongs to the user
        folder_doc = db.collection('folders').document(folder_id).get()
        if not folder_doc.exists:
            flash("The selected folder could not be found.", "error")
            return redirect(url_for('dashboard', _anchor='community'))

        folder_data = folder_doc.to_dict()
        user_hubs = [h.id for h in _get_user_stats(current_user.id)['hubs']]
        if folder_data.get('hub_id') not in user_hubs:
             flash("You can only share folders that you own.", "error")
             return redirect(url_for('dashboard', _anchor='community'))

        # 2. Prepare tags
        tags = [tag.strip().lower() for tag in tags_raw.split(',') if tag.strip()]

        # 3. Create the SharedFolder object
        shared_folder_ref = db.collection('shared_folders').document()
        new_shared_folder = SharedFolder(
            id=shared_folder_ref.id,
            original_folder_id=folder_id,
            original_hub_id=folder_data.get('hub_id'),
            owner_id=current_user.id,
            title=folder_data.get('name'),
            description=description,
            tags=tags
        )

        shared_folder_ref.set(new_shared_folder.to_dict())
        flash(f"Successfully shared '{new_shared_folder.title}' with the community!", "success")

    except Exception as e:
        print(f"Error sharing folder: {e}")
        flash("An unexpected error occurred while sharing your folder.", "error")

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

        # Prevent users from importing their own folders
        if shared_folder.owner_id == current_user.id:
            return jsonify({"success": False, "message": "You cannot import your own folder."}), 400

        # Fetch original folder and its items
        original_folder_doc = db.collection('folders').document(shared_folder.original_folder_id).get()
        if not original_folder_doc.exists:
             return jsonify({"success": False, "message": "The original folder no longer exists."}), 404
        
        original_folder = Folder.from_dict(original_folder_doc.to_dict())
        
        batch = db.batch()
        new_items_for_folder = []

        # Deep copy all assets (notes, activities)
        for item_ref in original_folder.items:
            item_id = item_ref.get('id')
            item_type = item_ref.get('type')
            
            collection_name = 'notes' if item_type == 'note' else 'activities'
            original_item_doc = db.collection(collection_name).document(item_id).get()

            if original_item_doc.exists:
                original_item_data = original_item_doc.to_dict()
                new_item_ref = db.collection(collection_name).document()
                
                # Create new data, updating ID and hub_id
                new_item_data = original_item_data
                new_item_data['id'] = new_item_ref.id
                new_item_data['hub_id'] = target_hub_id
                
                batch.set(new_item_ref, new_item_data)
                new_items_for_folder.append({'id': new_item_ref.id, 'type': item_type})

        # Create the new folder for the current user
        new_folder_ref = db.collection('folders').document()
        new_folder = Folder(
            id=new_folder_ref.id,
            hub_id=target_hub_id,
            name=f"{shared_folder.title} (Imported)",
            items=new_items_for_folder
        )
        batch.set(new_folder_ref, new_folder.to_dict())

        # Increment the import count on the shared folder
        batch.update(shared_folder_ref, {'imports': firestore.Increment(1)})

        batch.commit()
        return jsonify({"success": True, "message": f"'{new_folder.name}' imported successfully!"})

    except Exception as e:
        print(f"Error importing folder: {e}")
        return jsonify({"success": False, "message": "An internal server error occurred."}), 500

# ==============================================================================
# 9. MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    app.run(debug=True)