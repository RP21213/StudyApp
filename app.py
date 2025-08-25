# app.py
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
    base_dir = os.path.dirname(os.path.abspath(__file__))
    key_path = os.path.join(base_dir, "firebase_key.json")
    cred = credentials.Certificate(key_path)
    BUCKET_NAME = "ai-study-hub-f3040.firebasestorage.app" # YOUR FIREBASE BUCKET NAME
    firebase_admin.initialize_app(cred, {'storageBucket': BUCKET_NAME})
    print("Firebase initialized successfully!")
except Exception as e:
    print(f"Firebase initialization failed. Error: {e}")

db = firestore.client()
bucket = storage.bucket()

# --- Flask App and OpenAI Client Initialization ---
app = Flask(__name__)
app.secret_key = os.urandom(24)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))


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
    # ✅ FIX: This prompt is re-engineered to be more direct and reliable for topic tagging.
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

# ==============================================================================
# 4. FLASK ROUTES
# ==============================================================================
@app.route("/")
def home():
    return redirect(url_for('dashboard'))

@app.route("/dashboard")
def dashboard():
    hubs_ref = db.collection('hubs').stream()
    hubs_list = []
    for hub_doc in hubs_ref:
        hub = Hub.from_dict(hub_doc.to_dict())
        activities_query = db.collection('activities').where('hub_id', '==', hub.id).stream()
        hub_activities = [Activity.from_dict(doc.to_dict()) for doc in activities_query]
        notes_query = db.collection('notes').where('hub_id', '==', hub.id).stream()
        hub.notes_count = len(list(notes_query))
        hub.flashcard_count = len([a for a in hub_activities if a.type == 'Flashcards'])
        hub.quizzes_taken = len([a for a in hub_activities if a.type in ['Quiz', 'Mock Exam'] and a.status == 'graded'])
        hubs_list.append(hub)
    return render_template("dashboard.html", hubs=hubs_list)

@app.route("/add_hub", methods=["POST"])
def add_hub():
    hub_name = request.form.get('hub_name')
    if hub_name:
        hub_ref = db.collection('hubs').document()
        hub_styles = [
            {"color": "#fdba74", "pattern": "data:image/svg+xml,%3Csvg width='20' height='20' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M10 0v20M0 10h20' stroke-width='1' stroke='rgba(0,0,0,0.1)'/%3E%3C/svg%3E"},
            {"color": "#6ee7b7", "pattern": "data:image/svg+xml,%3Csvg width='20' height='20' xmlns='http://www.w3.org/2000/svg'%3E%3Ccircle cx='10' cy='10' r='4' stroke-width='1.5' stroke='rgba(0,0,0,0.1)' fill='none'/%3E%3C/svg%3E"},
            {"color": "#93c5fd", "pattern": "data:image/svg+xml,%3Csvg width='20' height='20' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M2 10 L10 18 L18 10' stroke-width='1.5' stroke='rgba(0,0,0,0.1)' fill='none'/%3E%3C/svg%3E"},
            {"color": "#c4b5fd", "pattern": "data:image/svg+xml,%3Csvg width='20' height='20' xmlns='http://www.w3.org/2000/svg'%3E%3Crect x='5' y='5' width='10' height='10' rx='2' stroke-width='1.5' stroke='rgba(0,0,0,0.1)' fill='none'/%3E%3C/svg%3E"},
            {"color": "#f9a8d4", "pattern": "data:image/svg+xml,%3Csvg width='24' height='24' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M12 2 L18 22 L6 22 Z' stroke-width='1.5' stroke='rgba(0,0,0,0.1)' fill='none'/%3E%3C/svg%3E"},
            {"color": "#fde047", "pattern": "data:image/svg+xml,%3Csvg width='20' height='20' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M0 0L20 20M20 0L0 20' stroke-width='1' stroke='rgba(0,0,0,0.1)'/%3E%3C/svg%3E"}
        ]
        style = random.choice(hub_styles)
        new_hub = Hub(id=hub_ref.id, name=hub_name, header_color=style['color'], header_pattern_url=style['pattern'])
        hub_ref.set(new_hub.to_dict())
    return redirect(url_for('dashboard'))

@app.route("/hub/<hub_id>/delete")
def delete_hub(hub_id):
    try:
        blobs = bucket.list_blobs(prefix=f"hubs/{hub_id}/")
        for blob in blobs: blob.delete()
        collections_to_delete = ['notes', 'activities', 'lectures']
        for coll in collections_to_delete:
            docs = db.collection(coll).where('hub_id', '==', hub_id).stream()
            for doc in docs: doc.reference.delete()
        db.collection('hubs').document(hub_id).delete()
        flash(f"Hub and all its data have been successfully deleted.", "success")
    except Exception as e:
        print(f"Error deleting hub {hub_id}: {e}")
        flash("An error occurred while trying to delete the hub.", "error")
    return redirect(url_for('dashboard'))

# ✅ ADDED: New route to handle exporting flashcards in different formats
@app.route("/flashcards/<activity_id>/export")
def export_flashcards(activity_id):
    # 1. Get the format and fetch the flashcard data from Firestore
    export_format = request.args.get('format')
    activity_doc = db.collection('activities').document(activity_id).get()

    if not activity_doc.exists:
        return "Flashcard set not found.", 404

    activity = Activity.from_dict(activity_doc.to_dict())
    cards = activity.data.get('cards', [])
    
    # Use an in-memory text buffer to build the file
    string_buffer = io.StringIO()

    # 2. Generate the correct file format based on the user's choice
    if export_format in ['quizlet', 'anki']:
        # Quizlet and Anki both use a simple Tab-Separated Value (.txt) format
        for card in cards:
            front = card.get('front', '').replace('\n', ' ')
            back = card.get('back', '').replace('\n', ' ')
            string_buffer.write(f"{front}\t{back}\n")
        
        mimetype = 'text/plain'
        filename = f"{activity.title or 'flashcards'}.txt"

    elif export_format == 'notion':
        # Notion imports well from Comma-Separated Value (.csv) files
        writer = csv.writer(string_buffer)
        writer.writerow(['Front', 'Back']) # CSV Header
        for card in cards:
            writer.writerow([card.get('front', ''), card.get('back', '')])

        mimetype = 'text/csv'
        filename = f"{activity.title or 'flashcards'}.csv"
    
    else:
        return "Invalid export format specified.", 400

    # 3. Create and send the downloadable file as a response
    response_data = string_buffer.getvalue()
    return Response(
        response_data,
        mimetype=mimetype,
        headers={"Content-Disposition": f"attachment;filename={filename}"}
    )


# In app.py

@app.route("/hub/<hub_id>")
def hub_page(hub_id):
    hub_doc = db.collection('hubs').document(hub_id).get()
    if not hub_doc.exists: return "Hub not found.", 404
    hub = Hub.from_dict(hub_doc.to_dict())

    # --- 1. Fetch all assets ---
    activities_query = db.collection('activities').where('hub_id', '==', hub_id).order_by('created_at').stream()
    all_activities = [Activity.from_dict(doc.to_dict()) for doc in activities_query]
    
    notes_query = db.collection('notes').where('hub_id', '==', hub_id).order_by('created_at', direction=firestore.Query.DESCENDING).stream()
    all_notes = [Note.from_dict(note.to_dict()) for note in notes_query]
    
    lectures_query = db.collection('lectures').where('hub_id', '==', hub_id).order_by('created_at', direction=firestore.Query.DESCENDING).stream()
    all_lectures = [Lecture.from_dict(doc.to_dict()) for doc in lectures_query]

    all_flashcards = [activity for activity in all_activities if activity.type == 'Flashcards']
    graded_activities = [activity for activity in all_activities if activity.status == 'graded']
    all_quizzes = [activity for activity in graded_activities if 'Quiz' in activity.type or 'Exam' in activity.type]

    # --- 2. Process Data for Progress Dashboard ---
    topic_performance = {}
    topic_history = {} # For improvement calculation
    last_seen_topic = {} # For recommendation calculation

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
    
    # --- 3. Calculate Spotlight Card Variety ---
    potential_spotlights = []
    
    # A) Weakest Topic (always a candidate if it exists)
    if topic_mastery:
        weakest = min(topic_mastery, key=lambda x: x['score'])
        if weakest['score'] < 70: # Only show if mastery is below a threshold
             potential_spotlights.append({'type': 'weakest', 'topic': weakest['topic'], 'score': weakest['score']})

    # B) Most Improved Topic
    best_improvement = {'topic': None, 'change': 0}
    for topic, history in topic_history.items():
        if len(history) >= 4: # Need at least 4 data points to compare
            midpoint = len(history) // 2
            first_half_avg = sum(history[:midpoint]) / midpoint
            second_half_avg = sum(history[midpoint:]) / (len(history) - midpoint)
            improvement = (second_half_avg - first_half_avg) * 100
            if improvement > best_improvement['change']:
                best_improvement = {'topic': topic, 'change': int(improvement)}
    
    if best_improvement['topic'] and best_improvement['change'] > 10:
        potential_spotlights.append({'type': 'improved', 'topic': best_improvement['topic'], 'change': best_improvement['change']})
        
    # C) Next Recommended Topic (Spaced Repetition)
    review_candidates = [t for t in topic_mastery if t['score'] < 90] # Don't need to review mastered topics
    if review_candidates:
        oldest_first = sorted(review_candidates, key=lambda x: last_seen_topic.get(x['topic']))
        potential_spotlights.append({'type': 'recommend', 'topic': oldest_first[0]['topic']})

    spotlight = random.choice(potential_spotlights) if potential_spotlights else None

    # --- 4. Calculate Gamification & Recent Activity ---
    total_xp = 0
    study_dates = set()
    today = datetime.now(timezone.utc).date()
    yesterday = today - timedelta(days=1)
    today_xp = 0
    yesterday_activities = {'quizzes': 0, 'flashcards': 0}

    for activity in all_activities:
        activity_date = activity.created_at.date()
        xp_gain = 0
        if activity.status == 'graded': xp_gain = 15
        elif activity.type == 'Flashcards': xp_gain = 5
        
        total_xp += xp_gain
        study_dates.add(activity_date)
        
        if activity_date == today:
            today_xp += xp_gain
        elif activity_date == yesterday:
            if 'Quiz' in activity.type or 'Exam' in activity.type:
                yesterday_activities['quizzes'] += 1
            elif activity.type == 'Flashcards':
                yesterday_activities['flashcards'] += 1
    
    # Calculate Streak
    streak_days = 0
    current_day = today
    while current_day in study_dates or (today - current_day).days < 2:
        if current_day in study_dates: streak_days += 1
        current_day -= timedelta(days=1)
        if current_day not in study_dates and (today - current_day).days >= 2: break

    # Calculate Level
    level_data = {}
    current_level = 1
    xp_for_next = 100
    xp_cumulative = 0
    while total_xp >= xp_cumulative + xp_for_next:
        xp_cumulative += xp_for_next
        current_level += 1
        xp_for_next = int(100 * (current_level ** 1.5))
    level_data = {"current_level": current_level, "xp_in_level": total_xp - xp_cumulative, "xp_for_next_level": xp_for_next}

    # --- 5. Finalize Hub Stats & Render Template ---
    hub.notes_count = len(all_notes)
    hub.flashcard_count = len(all_flashcards)
    hub.quizzes_taken = len(all_quizzes)

    return render_template(
        "hub.html", 
        hub=hub, 
        all_notes=all_notes, 
        all_lectures=all_lectures, 
        all_flashcards=all_flashcards, 
        all_quizzes=all_quizzes,
        topic_mastery=topic_mastery,
        level_data=level_data,
        total_xp=total_xp,
        streak_days=streak_days,
        spotlight=spotlight,
        today_xp=today_xp,
        yesterday_activities=yesterday_activities
    )

@app.route("/note/<note_id>")
def view_note(note_id):
    note_doc = db.collection('notes').document(note_id).get()
    if not note_doc.exists: return "Note not found", 404
    note = Note.from_dict(note_doc.to_dict())
    hub_doc = db.collection('hubs').document(note.hub_id).get()
    hub_name = hub_doc.to_dict().get('name') if hub_doc.exists else "Hub"
    return render_template("note.html", note=note, hub_name=hub_name)

# In app.py

@app.route('/explain_selection', methods=['POST'])
def explain_selection():
    data = request.get_json()
    selected_text = data.get('selected_text')
    context_text = data.get('context_text')
    explanation_type = data.get('explanation_type')

    if not selected_text or not explanation_type:
        return jsonify({"error": "Missing required data"}), 400

    # --- UPDATED: Stricter prompts to enforce brevity ---
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

# --- NEW: Dedicated route for generating interactive quiz data ---
@app.route('/generate_mini_quiz', methods=['POST'])
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
            model="gpt-4o", # Using a more powerful model for reliable JSON generation
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        quiz_data = json.loads(response.choices[0].message.content)
        return jsonify(quiz_data)
    except Exception as e:
        print(f"Error generating mini quiz: {e}")
        return jsonify({"error": "Failed to generate quiz."}), 500

@app.route("/flashcards/<activity_id>")
def view_flashcards(activity_id):
    # This route for the game itself remains unchanged
    activity_doc = db.collection('activities').document(activity_id).get()
    if not activity_doc.exists: return "Flashcard set not found.", 404
    activity = Activity.from_dict(activity_doc.to_dict())
    return render_template("flashcards.html", activity=activity)

# ✅ ADDED: New route to render the edit_flashcards.html page
@app.route("/flashcards/<activity_id>/edit_set")
def edit_flashcard_set(activity_id):
    activity_doc = db.collection('activities').document(activity_id).get()
    if not activity_doc.exists:
        flash("Flashcard set not found.", "error")
        return redirect(url_for('dashboard'))
    activity = Activity.from_dict(activity_doc.to_dict())
    return render_template("edit_flashcards.html", activity=activity)

# ✅ ADDED: New route to handle the form submission from the edit page
@app.route("/flashcards/<activity_id>/update_set", methods=["POST"])
def update_flashcard_set(activity_id):
    form_data = request.form
    new_cards = []
    i = 0
    # Loop through form fields dynamically until we can't find the next one
    while f'front_{i}' in form_data:
        front = form_data.get(f'front_{i}')
        back = form_data.get(f'back_{i}')
        if front and back and front.strip() and back.strip():
            new_cards.append({'front': front.strip(), 'back': back.strip()})
        i += 1
    
    try:
        activity_ref = db.collection('activities').document(activity_id)
        # Update only the 'cards' part of the 'data' field in Firestore
        activity_ref.update({'data.cards': new_cards})
        flash("Flashcards updated successfully!", "success")
    except Exception as e:
        flash(f"An error occurred: {e}", "error")

    return redirect(url_for('edit_flashcard_set', activity_id=activity_id))

@app.route('/explain-formula', methods=['POST'])
def explain_formula():
    formula = request.json.get('formula')
    if not formula: return jsonify({"error": "No formula provided"}), 400
    prompt = f"Explain the formula `{formula}` step-by-step for a university student. Break down each component and its significance. Use Markdown for formatting, including bullet points for clarity."
    response = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}])
    explanation_markdown = response.choices[0].message.content
    explanation_html = markdown.markdown(explanation_markdown)
    return jsonify({"explanation": explanation_html})

@app.route("/quiz/<activity_id>/delete", methods=["POST"])
def delete_quiz(activity_id):
    try:
        db.collection('activities').document(activity_id).delete()
        return jsonify({"success": True, "message": "Quiz deleted successfully."})
    except Exception as e:
        print(f"Error deleting activity {activity_id}: {e}")
        return jsonify({"success": False, "message": "An error occurred."}), 500

@app.route("/quiz/<activity_id>/edit", methods=["POST"])
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

# ✅ ADDED: New route to generate a quiz focused on the user's weakest topic.
@app.route("/hub/<hub_id>/generate_weakness_quiz/<topic>")
def generate_weakness_quiz(hub_id, topic):
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
    
    activity_ref = db.collection('activities').document()
    activity_title = f"Targeted Quiz: {topic}"
    new_activity = Activity(id=activity_ref.id, hub_id=hub_id, type='Quiz', title=activity_title, data={'questions': quiz_data.get('questions', [])})
    activity_ref.set(new_activity.to_dict())

    flash(f"Generated a special quiz to help you with {topic}!", "success")
    return redirect(url_for('take_lecture_quiz', quiz_id=new_activity.id))

# ==============================================================================
# 5. UNIFIED AI TOOL & DOWNLOAD ROUTES
# ==============================================================================
@app.route("/hub/<hub_id>/one_click_study", methods=["POST"])
def one_click_study_tool(hub_id):
    selected_files = request.json.get('selected_files')
    if not selected_files:
        return jsonify({"success": False, "message": "No files were selected."}), 400

    hub_text = get_text_from_hub_files(selected_files)
    if not hub_text or len(hub_text) < 100: # Basic check for meaningful content
        return jsonify({"success": False, "message": "Could not extract enough text from the selected document(s)."}), 500

    try:
        # --- Step 1: Generate and Validate Notes ---
        first_file_name = os.path.basename(selected_files[0]).replace('.pdf', '')
        lecture_title = f"Lecture for {first_file_name}"
        
        interactive_html = generate_interactive_notes_html(hub_text)
        if not interactive_html or len(interactive_html) < 50:
            raise ValueError("The AI failed to generate study notes.")
        
        note_ref = db.collection('notes').document()
        new_note = Note(id=note_ref.id, hub_id=hub_id, title=lecture_title, content_html=interactive_html)

        # --- Step 2: Generate and Validate Flashcards ---
        flashcards_raw = generate_flashcards_from_text(hub_text)
        flashcards_parsed = parse_flashcards(flashcards_raw)
        if not flashcards_parsed:
            raise ValueError("The AI failed to generate flashcards.")
        
        flashcard_title = f"Flashcards for {first_file_name}"
        activity_ref_fc = db.collection('activities').document()
        new_activity_fc = Activity(id=activity_ref_fc.id, hub_id=hub_id, type='Flashcards', data={'cards': flashcards_parsed}, status='completed', title=flashcard_title)
        
        # --- Step 3: Generate and Validate 3 Quizzes ---
        quiz_ids = []
        new_quizzes = []
        for i in range(1, 4):
            quiz_json = generate_quiz_from_text(hub_text)
            quiz_data = safe_load_json(quiz_json)
            if not quiz_data.get('questions'):
                # If one quiz fails, we can proceed with fewer, but we'll log it.
                print(f"Warning: AI failed to generate questions for Quiz {i}.")
                continue

            # FIXED: Added a descriptive title to each quiz
            quiz_title = f"Practice Quiz {i} for {first_file_name}"
            activity_ref_quiz = db.collection('activities').document()
            new_activity_quiz = Activity(
                id=activity_ref_quiz.id, 
                hub_id=hub_id, 
                type=f'Quiz', # Simplified type for better grouping
                title=quiz_title, 
                data={'questions': quiz_data.get('questions', [])}
            )
            quiz_ids.append(new_activity_quiz.id)
            new_quizzes.append(new_activity_quiz)

        if not new_quizzes:
            raise ValueError("The AI failed to generate any practice quizzes.")

        # --- Step 4: Save Everything to Database in a Batch ---
        # This ensures that if one part fails, nothing is saved.
        batch = db.batch()
        
        # Save the Note
        batch.set(note_ref, new_note.to_dict())
        
        # Save the Flashcard Activity
        batch.set(activity_ref_fc, new_activity_fc.to_dict())

        # Save all successful Quiz Activities
        for quiz_activity in new_quizzes:
            quiz_ref = db.collection('activities').document(quiz_activity.id)
            batch.set(quiz_ref, quiz_activity.to_dict())
        
        # Finally, create and save the Lecture linking all assets
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
        
        batch.commit() # Execute all database operations

        return jsonify({"success": True, "message": "Study materials created successfully!", "lecture_id": new_lecture.id})

    except Exception as e:
        print(f"Error in one_click_study_tool: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/hub/<hub_id>/run_tool", methods=["POST"])
def run_hub_tool(hub_id):
    tool = request.form.get('tool')
    selected_files = request.form.getlist('selected_files')
    if not selected_files:
        flash("Please select at least one file to process.", "error")
        return redirect(url_for('hub_page', hub_id=hub_id))
    hub_text = get_text_from_hub_files(selected_files)
    if not hub_text:
        flash("Could not extract text from the selected file(s).", "error")
        return redirect(url_for('hub_page', hub_id=hub_id))
    if tool == 'notes':
        interactive_html = generate_interactive_notes_html(hub_text)
        note_ref = db.collection('notes').document()
        first_file_name = os.path.basename(selected_files[0]).replace('.pdf', '')
        note_title = f"Notes for {first_file_name}"
        new_note = Note(id=note_ref.id, hub_id=hub_id, title=note_title, content_html=interactive_html)
        note_ref.set(new_note.to_dict())
        flash("Your interactive notes have been created!", "success")
        return redirect(url_for('view_note', note_id=new_note.id))
    elif tool == 'flashcards':
        flashcards_raw = generate_flashcards_from_text(hub_text)
        flashcards_parsed = parse_flashcards(flashcards_raw)
        if not flashcards_parsed:
            flash("The AI could not generate any flashcards from the selected text. Please try again.", "warning")
            return redirect(url_for('hub_page', hub_id=hub_id))
        
        activity_ref = db.collection('activities').document()
        first_file_name = os.path.basename(selected_files[0]).replace('.pdf', '')
        activity_title = f"Flashcards for {first_file_name}"
        
        new_activity = Activity(id=activity_ref.id, hub_id=hub_id, type='Flashcards', title=activity_title, data={'cards': flashcards_parsed}, status='completed')
        activity_ref.set(new_activity.to_dict())
        
        flash(f"{len(flashcards_parsed)} flashcards were generated! You can now edit them.", "success")
        # ✅ CHANGED: Redirect to the new edit page instead of the game
        return redirect(url_for('edit_flashcard_set', activity_id=new_activity.id))
    elif tool == 'quiz':
        quiz_json = generate_quiz_from_text(hub_text)
        quiz_data = safe_load_json(quiz_json)
        activity_ref = db.collection('activities').document()
        first_file_name = os.path.basename(selected_files[0]).replace('.pdf', '')
        activity_title = f"Quiz for {first_file_name}"
        new_activity = Activity(id=activity_ref.id, hub_id=hub_id, type='Quiz', title=activity_title, data={'questions': quiz_data.get('questions', [])})
        activity_ref.set(new_activity.to_dict())
        return redirect(url_for('take_lecture_quiz', quiz_id=new_activity.id))
    elif tool == 'exam':
        activity_ref = db.collection('activities').document()
        activity_type = 'Mock Exam'
        new_activity = Activity(id=activity_ref.id, hub_id=hub_id, type=activity_type)
        activity_ref.set(new_activity.to_dict())
        return render_template("exam_customize.html", hub_id=hub_id, selected_files=selected_files, activity_id=new_activity.id)
    elif tool == 'analyse':
        analysis_html = markdown.markdown(analyse_papers_with_ai(hub_text))
        return render_template("analysis.html", analysis_html=analysis_html)
    elif tool == 'heatmap':
        heatmap_json_str = generate_interactive_heatmap_data(hub_text)
        heatmap_data = safe_load_json(heatmap_json_str)
        if not heatmap_data or "topics" not in heatmap_data or "documents" not in heatmap_data or not heatmap_data["documents"]:
            flash("Could not generate heatmap data. The AI may have had trouble analyzing the documents. Please try again with different files.", "warning")
            return redirect(url_for('hub_page', hub_id=hub_id))
        return render_template("heatmap.html", hub_id=hub_id, heatmap_data=heatmap_data)
    return "Invalid tool selected.", 400

@app.route("/note/<note_id>/download_pdf")
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
def create_exam(hub_id):
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
# 6. LECTURE, QUIZ, AND TUTOR ROUTES
# ==============================================================================
@app.route("/lecture/<lecture_id>")
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

@app.route("/quiz/<quiz_id>")
def take_lecture_quiz(quiz_id):
    quiz_doc = db.collection('activities').document(quiz_id).get()
    if not quiz_doc.exists: return "Quiz not found.", 404
    quiz_activity = Activity.from_dict(quiz_doc.to_dict())
    questions = quiz_activity.data.get('questions', [])
    return render_template("quiz.html", questions=questions, time_limit=600, activity_id=quiz_id)

# In app.py

@app.route("/quiz/<activity_id>/submit", methods=["POST"])
def submit_quiz(activity_id):
    activity_ref = db.collection('activities').document(activity_id)
    activity_doc = activity_ref.get()
    if not activity_doc.exists: return "Quiz not found.", 404
    activity = Activity.from_dict(activity_doc.to_dict())
    questions = activity.data.get('questions', [])
    graded_answers = []
    mcq_score, total_mcq, open_ended_score, total_open_ended_possible = 0, 0, 0, 0
    for i, q in enumerate(questions):
        user_answer = request.form.get(f'question-{i}')
        # ✅ ADDED: Get the topic for the current question. Default to 'General' if not found.
        question_topic = q.get('topic', 'General')

        if q.get('type') == 'multiple_choice':
            total_mcq += 1
            
            correct_answer_text = q.get('correct_answer', '').lower().strip().rstrip('.,;!?')
            user_answer_text = user_answer.lower().strip().rstrip('.,;!?') if user_answer else ''
            is_correct = (user_answer and user_answer_text == correct_answer_text)

            if is_correct: mcq_score += 1
            # ✅ CHANGED: Save the question's topic along with the answer.
            graded_answers.append({"user_answer": user_answer, "correct": is_correct, "topic": question_topic})
        
        elif q.get('type') in ['short_answer', 'explanation']:
            total_open_ended_possible += 10
            try:
                feedback_json_str = grade_answer_with_ai(q.get('question'), q.get('model_answer'), user_answer)
                feedback_data = json.loads(feedback_json_str)
                score = feedback_data.get('score', 0)
                open_ended_score += score
                # ✅ CHANGED: Save the topic for open-ended questions too.
                graded_answers.append({"user_answer": user_answer, "feedback": feedback_data.get('feedback'), "score": score, "topic": question_topic})
            except Exception as e:
                print(f"Error grading open-ended answer for quiz {activity_id}: {e}")
                graded_answers.append({"user_answer": user_answer, "feedback": "Could not be graded automatically.", "score": 0, "topic": question_topic})

    total_achieved_score = mcq_score + open_ended_score
    total_possible_score = total_mcq + total_open_ended_possible
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
    return redirect(url_for('quiz_results', activity_id=activity_id))


@app.route("/quiz/<activity_id>/results")
def quiz_results(activity_id):
    activity_doc = db.collection('activities').document(activity_id).get()
    if not activity_doc.exists: return "Quiz results not found.", 404
    activity = Activity.from_dict(activity_doc.to_dict())
    results = activity.graded_results
    return render_template("quiz_results.html", activity=activity, results=results)

# In app.py

@app.route("/flashcards/<activity_id>/save_game", methods=["POST"])
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

        # ✅ ADDED: Identify the actual flashcards the user got wrong.
        incorrect_cards = [all_cards[i] for i in incorrect_indices if i < len(all_cards)]

        activity_ref.update({
            "game_results": {
                "correct_count": len(correct_indices),
                "incorrect_count": len(incorrect_indices),
                "total_cards": len(all_cards),
                "last_played": datetime.now(timezone.utc),
                # ✅ ADDED: Store the list of cards that need more practice.
                "incorrect_cards": incorrect_cards 
            }
        })
        return jsonify({"success": True, "message": "Game results saved."})
    except Exception as e:
        print(f"Error saving flashcard game for {activity_id}: {e}")
        return jsonify({"success": False, "message": "An error occurred."}), 500

@app.route("/lecture/<lecture_id>/delete")
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

@app.route("/hub/<hub_id>/ask_tutor", methods=["POST"])
def ask_ai_tutor(hub_id):
    data = request.get_json()
    question = data.get('question')
    selected_files = data.get('selected_files')
    chat_history_json = data.get('chat_history', [])
    answer_style = data.get('answer_style', 'default')
    chat_history = []
    for msg in chat_history_json:
        if msg.get('role') == 'user': chat_history.append(HumanMessage(content=msg.get('content')))
        elif msg.get('role') == 'ai': chat_history.append(AIMessage(content=msg.get('content')))
    if not question: return jsonify({"error": "No question provided."}), 400
    if not selected_files: return jsonify({"answer": "Please select at least one document to use as context for my answer."})
    try:
        context_text = get_text_from_hub_files(selected_files)
        if not context_text: return jsonify({"answer": "I couldn't extract any text from the selected documents."})
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.create_documents([context_text])
        vector_store = FAISS.from_documents(documents, embeddings)
        retriever = vector_store.as_retriever()
        contextualize_q_prompt = ChatPromptTemplate.from_messages([("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."), MessagesPlaceholder("chat_history"), ("human", "{input}")])
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
        style_instructions = ""
        if answer_style == "summary": style_instructions = "Provide a concise, one-paragraph summary of the answer."
        elif answer_style == "bullets": style_instructions = "List the key points of the answer using bullet points."
        else: style_instructions = "Provide a clear and detailed explanation."
        
        # === FIX: Corrected placeholder from {{context}} to {context} ===
        qa_system_prompt = f"""You are an expert AI tutor. Your goal is to help the user understand concepts based on their documents. Use the following retrieved context as your primary source of knowledge to answer the user's question. Your answer must be grounded in the context provided.
        - **Synthesize, do not just search:** Create a comprehensive answer by combining information from the relevant parts of the context.
        - **If the context is related but doesn't answer directly:** Explain what you can find and how it relates.
        - **If the topic is completely absent:** Politely state that the provided documents do not cover the topic.
        - **Style instruction:** {style_instructions}
        Context:
        {{context}}"""

        # === FIX: Corrected placeholder from {{input}} to {input} ===
        qa_prompt = ChatPromptTemplate.from_messages([("system", qa_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{{input}}")])
        
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        response = rag_chain.invoke({"input": question, "chat_history": chat_history})
        answer = response.get("answer", "Sorry, I encountered an issue and couldn't generate a response.")
        return jsonify({"answer": answer})
    except Exception as e:
        print(f"Error in AI Tutor: {e}")
        return jsonify({"error": "An internal error occurred."}), 500

# ==============================================================================
# 6A. ASSET MANAGEMENT ROUTES (EDIT/DELETE)
# ==============================================================================
@app.route("/note/<note_id>/delete", methods=["POST"])
def delete_note(note_id):
    try:
        db.collection('notes').document(note_id).delete()
        return jsonify({"success": True, "message": "Note deleted successfully."})
    except Exception as e:
        print(f"Error deleting note {note_id}: {e}")
        return jsonify({"success": False, "message": "An error occurred."}), 500

@app.route("/note/<note_id>/edit", methods=["POST"])
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
def delete_flashcards(activity_id):
    try:
        db.collection('activities').document(activity_id).delete()
        return jsonify({"success": True, "message": "Flashcard set deleted successfully."})
    except Exception as e:
        print(f"Error deleting flashcard set {activity_id}: {e}")
        return jsonify({"success": False, "message": "An error occurred."}), 500

@app.route("/flashcards/<activity_id>/edit", methods=["POST"])
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
            collections_to_delete = ['notes', 'activities', 'lectures']
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
# 7. MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    app.run(debug=True)



