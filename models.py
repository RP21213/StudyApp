from datetime import datetime, timezone, timedelta # MODIFIED: Add timedelta
from flask_login import UserMixin 

# NEW: Model for Community-Shared Folders
class SharedFolder:
    def __init__(self, id, original_folder_id, original_hub_id, owner_id, 
                 title, description, tags=None, created_at=None, 
                 likes=0, imports=0, liked_by=None, imported_by=None, **kwargs): # MODIFIED: added imported_by
        self.id = id
        self.original_folder_id = original_folder_id
        self.original_hub_id = original_hub_id
        self.owner_id = owner_id
        self.title = title
        self.description = description
        self.tags = tags if tags is not None else []
        self.created_at = created_at or datetime.now(timezone.utc)
        self.likes = likes
        self.imports = imports
        self.liked_by = liked_by if liked_by is not None else []
        self.imported_by = imported_by if imported_by is not None else [] # NEW: Track users who have imported

    def to_dict(self):
        return {
            'id': self.id,
            'original_folder_id': self.original_folder_id,
            'original_hub_id': self.original_hub_id,
            'owner_id': self.owner_id,
            'title': self.title,
            'description': self.description,
            'tags': self.tags,
            'created_at': self.created_at,
            'likes': self.likes,
            'imports': self.imports,
            'liked_by': self.liked_by,
            'imported_by': self.imported_by, # NEW: Add to dictionary for Firestore
        }

    @staticmethod
    def from_dict(source):
        return SharedFolder(**source)
    
# --- UPDATED: User Model with Settings and Spotify Fields ---
class User(UserMixin):
    def __init__(self, id, email, password_hash, display_name=None, bio="", avatar_url=None, 
                 subscription_tier='free', subscription_active=False, stripe_customer_id=None, stripe_subscription_id=None,
                 # --- NEW: Fields for Settings ---
                 profile_visible=True, activity_visible=True, default_note_privacy='private',
                 font_size_preference='default', high_contrast_mode=False, language='en-US',
                 # --- NEW: Fields for Spotify ---
                 spotify_access_token=None, spotify_refresh_token=None, spotify_token_expires_at=None,
                 # --- NEW: Onboarding Fields ---
                 has_completed_onboarding=False,
                 referral_source=None, goals=None, email_opt_in=False, theme_preference='light',
                 **kwargs):
        self.id = id
        self.email = email
        self.password_hash = password_hash
        self.display_name = display_name if display_name else email.split('@')[0]
        self.bio = bio
        self.avatar_url = avatar_url if avatar_url else 'https://storage.googleapis.com/ai-study-hub-f3040.appspot.com/avatars/default_avatar.png'
        self.subscription_tier = subscription_tier
        self.subscription_active = subscription_active
        self.stripe_customer_id = stripe_customer_id
        self.stripe_subscription_id = stripe_subscription_id
        
        # --- NEW: Initialize Settings Properties ---
        self.profile_visible = profile_visible
        self.activity_visible = activity_visible
        self.default_note_privacy = default_note_privacy
        self.font_size_preference = font_size_preference
        self.high_contrast_mode = high_contrast_mode
        self.language = language

        # --- NEW: Initialize Spotify Properties ---
        self.spotify_access_token = spotify_access_token
        self.spotify_refresh_token = spotify_refresh_token
        self.spotify_token_expires_at = spotify_token_expires_at

        # --- NEW: Onboarding Properties ---
        self.has_completed_onboarding = has_completed_onboarding
        self.referral_source = referral_source
        self.goals = goals
        self.email_opt_in = email_opt_in
        self.theme_preference = theme_preference

    def to_dict(self):
        return {
            'id': self.id,
            'email': self.email,
            'password_hash': self.password_hash,
            'display_name': self.display_name,
            'bio': self.bio,
            'avatar_url': self.avatar_url,
            'subscription_tier': self.subscription_tier,
            'subscription_active': self.subscription_active,
            'stripe_customer_id': self.stripe_customer_id,
            'stripe_subscription_id': self.stripe_subscription_id,
            'profile_visible': self.profile_visible,
            'activity_visible': self.activity_visible,
            'default_note_privacy': self.default_note_privacy,
            'font_size_preference': self.font_size_preference,
            'high_contrast_mode': self.high_contrast_mode,
            'language': self.language,
            # --- NEW: Add Spotify to dict for Firestore ---
            'spotify_access_token': self.spotify_access_token,
            'spotify_refresh_token': self.spotify_refresh_token,
            'spotify_token_expires_at': self.spotify_token_expires_at,
            # --- NEW: Onboarding to dict ---
            'has_completed_onboarding': self.has_completed_onboarding,
            'referral_source': self.referral_source,
            'goals': self.goals,
            'email_opt_in': self.email_opt_in,
            'theme_preference': self.theme_preference,
        }

    @staticmethod
    def from_dict(source):
        default_avatar = 'https://storage.googleapis.com/ai-study-hub-f3040.appspot.com/avatars/default_avatar.png'
        return User(
            id=source.get('id'),
            email=source.get('email'),
            password_hash=source.get('password_hash'),
            display_name=source.get('display_name'),
            bio=source.get('bio'),
            avatar_url=source.get('avatar_url', default_avatar),
            subscription_tier=source.get('subscription_tier', 'free'),
            subscription_active=source.get('subscription_active', False),
            stripe_customer_id=source.get('stripe_customer_id'),
            stripe_subscription_id=source.get('stripe_subscription_id'),
            profile_visible=source.get('profile_visible', True),
            activity_visible=source.get('activity_visible', True),
            default_note_privacy=source.get('default_note_privacy', 'private'),
            font_size_preference=source.get('font_size_preference', 'default'),
            high_contrast_mode=source.get('high_contrast_mode', False),
            language=source.get('language', 'en-US'),
            # --- NEW: Retrieve Spotify from dict ---
            spotify_access_token=source.get('spotify_access_token'),
            spotify_refresh_token=source.get('spotify_refresh_token'),
            spotify_token_expires_at=source.get('spotify_token_expires_at'),
            # --- NEW: Onboarding from dict ---
            has_completed_onboarding=source.get('has_completed_onboarding', False),
            referral_source=source.get('referral_source'),
            goals=source.get('goals'),
            email_opt_in=source.get('email_opt_in', False),
            theme_preference=source.get('theme_preference', 'light')
        )
    
# --- NEW: Model for Note-Taking with Slides ---
class AnnotatedSlideDeck:
    def __init__(self, id, hub_id, user_id, title, source_file_path, slides_data=None, created_at=None, **kwargs):
        self.id = id
        self.hub_id = hub_id
        self.user_id = user_id
        self.title = title
        self.source_file_path = source_file_path
        # slides_data is a list of dicts: [{'slide_number': 0, 'notes_html': '<p>...</p>'}]
        self.slides_data = slides_data if slides_data is not None else []
        self.created_at = created_at or datetime.now(timezone.utc)

    def to_dict(self):
        return {
            'id': self.id,
            'hub_id': self.hub_id,
            'user_id': self.user_id,
            'title': self.title,
            'source_file_path': self.source_file_path,
            'slides_data': self.slides_data,
            'created_at': self.created_at,
        }

    @staticmethod
    def from_dict(source):
        return AnnotatedSlideDeck(**source)

# --- Your existing models below are unchanged ---

class Folder:
    def __init__(self, id, hub_id, name, items=None, created_at=None, **kwargs):
        self.id = id
        self.hub_id = hub_id
        self.name = name
        self.items = items if items is not None else []
        self.created_at = created_at if created_at else datetime.now(timezone.utc)
    def to_dict(self): return {'id': self.id, 'hub_id': self.hub_id, 'name': self.name, 'items': self.items, 'created_at': self.created_at}
    @staticmethod
    def from_dict(source): return Folder(**source)

class StudySession:
    def __init__(self, id, hub_id, title, source_files, status='not_started', session_plan=None, results=None, created_at=None, note_id=None, flashcard_activity_id=None, quiz_activity_ids=None, **kwargs):
        self.id, self.hub_id, self.title, self.source_files, self.status = id, hub_id, title, source_files if source_files is not None else [], status
        self.session_plan, self.results, self.created_at = session_plan if session_plan is not None else {}, results if results is not None else [], created_at if created_at else datetime.now(timezone.utc)
        self.note_id, self.flashcard_activity_id, self.quiz_activity_ids = note_id, flashcard_activity_id, quiz_activity_ids if quiz_activity_ids is not None else []
    def to_dict(self): return {'id': self.id, 'hub_id': self.hub_id, 'title': self.title, 'source_files': self.source_files, 'status': self.status, 'session_plan': self.session_plan, 'results': self.results, 'created_at': self.created_at, 'note_id': self.note_id, 'flashcard_activity_id': self.flashcard_activity_id, 'quiz_activity_ids': self.quiz_activity_ids}
    @staticmethod
    def from_dict(source): return StudySession(**source)
    
# --- Hub Model ---
# UPDATED: Added user_id to track ownership
class Hub:
    def __init__(self, id, name, user_id, files=None, color=None, header_image_url=None, header_color=None, header_pattern_url=None, **kwargs):
        self.id = id
        self.name = name
        self.user_id = user_id 
        self.files = files if files is not None else []
        self.color = color if color is not None else "#6366f1"
        self.header_color = header_color
        self.header_pattern_url = header_pattern_url
        
        # --- NEW: Fields for persistent progress ---
        self.total_xp = kwargs.get('total_xp', 0)
        self.streak_days = kwargs.get('streak_days', 0)
        self.last_study_date = kwargs.get('last_study_date') # Will be a datetime object

        # These will still be calculated on the fly from existing assets
        self.notes_count = 0
        self.flashcard_count = 0
        self.quizzes_taken = 0

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'user_id': self.user_id,
            'files': self.files,
            'color': self.color,
            'header_color': self.header_color,
            'header_pattern_url': self.header_pattern_url,
            # --- NEW: Save progress fields to Firestore ---
            'total_xp': self.total_xp,
            'streak_days': self.streak_days,
            'last_study_date': self.last_study_date
        }

    @staticmethod
    def from_dict(source):
        # Create the Hub instance
        hub = Hub(
            id=source.get('id'),
            name=source.get('name'),
            user_id=source.get('user_id'),
            files=source.get('files', []),
            color=source.get('color'),
            header_color=source.get('header_color'),
            header_pattern_url=source.get('header_pattern_url')
        )
        # --- NEW: Load progress fields from Firestore ---
        hub.total_xp = source.get('total_xp', 0)
        hub.streak_days = source.get('streak_days', 0)
        hub.last_study_date = source.get('last_study_date')
        return hub
class Activity:
    def __init__(self, id, hub_id, type, status='in_progress', created_at=None, data=None, **kwargs):
        self.id = id
        self.hub_id = hub_id
        self.type = type
        self.status = status
        self.created_at = created_at if created_at else datetime.now(timezone.utc)
        self.score = kwargs.get('score')
        self.exam_data = kwargs.get('exam_data')
        self.graded_results = kwargs.get('graded_results')
        self.mcq_score = kwargs.get('mcq_score')
        self.title = kwargs.get('title')
        self.game_results = kwargs.get('game_results')
        self.data = data if data is not None else {}
    def to_dict(self): return {'id': self.id, 'hub_id': self.hub_id, 'type': self.type, 'status': self.status, 'created_at': self.created_at, 'score': self.score, 'exam_data': self.exam_data, 'graded_results': self.graded_results, 'mcq_score': self.mcq_score, 'title': self.title, 'game_results': self.game_results, 'data': self.data}
    @staticmethod
    def from_dict(source): return Activity(**source)

class Note:
    def __init__(self, id, hub_id, title, content_html, created_at=None, **kwargs):
        self.id, self.hub_id, self.title, self.content_html = id, hub_id, title, content_html
        self.created_at = created_at if created_at else datetime.now(timezone.utc)
    def to_dict(self): return {'id': self.id, 'hub_id': self.hub_id, 'title': self.title, 'content_html': self.content_html, 'created_at': self.created_at}
    @staticmethod
    def from_dict(source): return Note(**source)


class Lecture:
    def __init__(self, id, hub_id, title, note_id, flashcard_id, quiz_ids, source_files, created_at=None, **kwargs):
        self.id = id
        self.hub_id = hub_id
        self.title = title
        self.note_id = note_id
        self.flashcard_id = flashcard_id
        self.quiz_ids = quiz_ids if quiz_ids is not None else []
        self.source_files = source_files if source_files is not None else []
        self.created_at = created_at if created_at else datetime.now(timezone.utc)

    def to_dict(self):
        return {
            'id': self.id,
            'hub_id': self.hub_id,
            'title': self.title,
            'note_id': self.note_id,
            'flashcard_id': self.flashcard_id,
            'quiz_ids': self.quiz_ids,
            'source_files': self.source_files,
            'created_at': self.created_at,
        }

    @staticmethod
    def from_dict(source):
        return Lecture(**source)
        
class Notification:
    def __init__(self, id, hub_id, message, link, read=False, created_at=None, **kwargs):
        self.id = id
        self.hub_id = hub_id
        self.message = message
        self.link = link
        self.read = read
        self.created_at = created_at or datetime.now(timezone.utc)

    def to_dict(self):
        return {
            'id': self.id,
            'hub_id': self.hub_id,
            'message': self.message,
            'link': self.link,
            'read': self.read,
            'created_at': self.created_at,
        }

    @staticmethod
    def from_dict(source):
        return Notification(**source)

class Assignment:
    def __init__(self, id, hub_id, title, module, word_count_target, due_date,
                 referencing_style, voice, originality_level, brief_text, rubric_text,
                 source_files, cite_only_uploaded, created_at=None, status='parsing',
                 parsed_requirements=None, outline=None, draft_content=None,
                 bibliography=None, **kwargs):
        self.id = id
        self.hub_id = hub_id
        self.title = title
        self.module = module
        self.word_count_target = word_count_target
        self.due_date = due_date
        self.referencing_style = referencing_style
        self.voice = voice
        self.originality_level = originality_level
        self.brief_text = brief_text
        self.rubric_text = rubric_text
        self.source_files = source_files if source_files is not None else []
        self.cite_only_uploaded = cite_only_uploaded
        self.created_at = created_at or datetime.now(timezone.utc)
        
        # --- Properties updated by the AI ---
        self.status = status  # e.g., 'parsing', 'outline_ready', 'drafting', 'complete'
        self.parsed_requirements = parsed_requirements if parsed_requirements is not None else {}
        self.outline = outline if outline is not None else {}
        self.draft_content = draft_content if draft_content is not None else {}
        self.bibliography = bibliography if bibliography is not None else []

    def to_dict(self):
        return {
            'id': self.id,
            'hub_id': self.hub_id,
            'title': self.title,
            'module': self.module,
            'word_count_target': self.word_count_target,
            'due_date': self.due_date,
            'referencing_style': self.referencing_style,
            'voice': self.voice,
            'originality_level': self.originality_level,
            'brief_text': self.brief_text,
            'rubric_text': self.rubric_text,
            'source_files': self.source_files,
            'cite_only_uploaded': self.cite_only_uploaded,
            'created_at': self.created_at,
            'status': self.status,
            'parsed_requirements': self.parsed_requirements,
            'outline': self.outline,
            'draft_content': self.draft_content,
            'bibliography': self.bibliography,
        }

    @staticmethod
    def from_dict(source):
        return Assignment(**source)
    
# --- UPDATED: Class for Calendar Events ---
class CalendarEvent:
    def __init__(self, id, hub_id, title, event_type, start_time, end_time, 
                 source_files=None, focus=None, status='scheduled', all_day=False, created_at=None, **kwargs):
        self.id = id
        self.hub_id = hub_id
        self.title = title
        self.event_type = event_type
        self.start_time = start_time
        self.end_time = end_time
        self.source_files = source_files if source_files is not None else []
        self.focus = focus
        self.status = status
        self.all_day = all_day
        self.created_at = created_at or datetime.now(timezone.utc)
        self.color = self.get_event_color()

    def get_event_color(self):
        colors = {
            "Study Session": "#3b82f6",
            "Mock Exam": "#ef4444",
            "Flashcard Review": "#f59e0b",
            "Draft Review": "#10b981",
            "Imported": "#6b7280",
            "Custom": "#8b5cf6"
        }
        return colors.get(self.event_type, "#6b7280")

    def to_dict(self):
        return {
            'id': self.id,
            'hub_id': self.hub_id,
            'title': self.title,
            'event_type': self.event_type,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'source_files': self.source_files,
            'focus': self.focus,
            'status': self.status,
            'all_day': self.all_day,
            'color': self.color,
            'created_at': self.created_at,
        }

    @staticmethod
    def from_dict(source):
        # The firestore library automatically converts timestamps to datetime objects.
        # This method is now simpler and more robust against library version changes.
        return CalendarEvent(**source)

# --- NEW: Study Group Models ---
class StudyGroup:
    def __init__(self, id, name, description, code, owner_id, created_at=None, member_count=1, **kwargs):
        self.id = id
        self.name = name
        self.description = description or ""
        self.code = code  # 5-digit code for joining
        self.owner_id = owner_id
        self.created_at = created_at or datetime.now(timezone.utc)
        self.member_count = member_count

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'code': self.code,
            'owner_id': self.owner_id,
            'created_at': self.created_at,
            'member_count': self.member_count,
        }

    @staticmethod
    def from_dict(source):
        return StudyGroup(**source)

class StudyGroupMember:
    def __init__(self, id, study_group_id, user_id, joined_at=None, **kwargs):
        self.id = id
        self.study_group_id = study_group_id
        self.user_id = user_id
        self.joined_at = joined_at or datetime.now(timezone.utc)

    def to_dict(self):
        return {
            'id': self.id,
            'study_group_id': self.study_group_id,
            'user_id': self.user_id,
            'joined_at': self.joined_at,
        }

    @staticmethod
    def from_dict(source):
        return StudyGroupMember(**source)

class SharedResource:
    def __init__(self, id, resource_type, resource_id, hub_id, owner_id, study_group_id=None, 
                 title, description, tags=None, created_at=None, likes=0, imports=0, 
                 liked_by=None, imported_by=None, **kwargs):
        self.id = id
        self.resource_type = resource_type  # 'folder', 'note', 'flashcard', 'quiz', 'cheatsheet'
        self.resource_id = resource_id  # ID of the original resource
        self.hub_id = hub_id  # Original hub ID
        self.owner_id = owner_id
        self.study_group_id = study_group_id  # None for global sharing
        self.title = title
        self.description = description
        self.tags = tags if tags is not None else []
        self.created_at = created_at or datetime.now(timezone.utc)
        self.likes = likes
        self.imports = imports
        self.liked_by = liked_by if liked_by is not None else []
        self.imported_by = imported_by if imported_by is not None else []

    def to_dict(self):
        return {
            'id': self.id,
            'resource_type': self.resource_type,
            'resource_id': self.resource_id,
            'hub_id': self.hub_id,
            'owner_id': self.owner_id,
            'study_group_id': self.study_group_id,
            'title': self.title,
            'description': self.description,
            'tags': self.tags,
            'created_at': self.created_at,
            'likes': self.likes,
            'imports': self.imports,
            'liked_by': self.liked_by,
            'imported_by': self.imported_by,
        }

    @staticmethod
    def from_dict(source):
        return SharedResource(**source)