# models.py
from datetime import datetime, timezone

class Hub:
    def __init__(self, id, name, files=None, color=None, header_image_url=None, header_color=None, header_pattern_url=None, **kwargs):
        self.id = id
        self.name = name
        self.files = files if files is not None else []
        self.color = color if color is not None else "#6366f1"
        self.header_color = header_color
        self.header_pattern_url = header_pattern_url
        
        self.notes_count = 0
        self.flashcard_count = 0
        self.quizzes_taken = 0

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'files': self.files,
            'color': self.color,
            'header_color': self.header_color,
            'header_pattern_url': self.header_pattern_url
        }

    @staticmethod
    def from_dict(source):
        return Hub(
            id=source.get('id'),
            name=source.get('name'),
            files=source.get('files', []),
            color=source.get('color'),
            header_color=source.get('header_color'),
            header_pattern_url=source.get('header_pattern_url')
        )

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

    def to_dict(self):
        return {
            'id': self.id,
            'hub_id': self.hub_id,
            'type': self.type,
            'status': self.status,
            'created_at': self.created_at,
            'score': self.score,
            'exam_data': self.exam_data,
            'graded_results': self.graded_results,
            'mcq_score': self.mcq_score,
            'title': self.title,
            'game_results': self.game_results,
            'data': self.data,
        }

    @staticmethod
    def from_dict(source):
        return Activity(**source)

class Note:
    def __init__(self, id, hub_id, title, content_html, created_at=None, **kwargs):
        self.id = id
        self.hub_id = hub_id
        self.title = title
        self.content_html = content_html
        self.created_at = created_at if created_at else datetime.now(timezone.utc)

    def to_dict(self):
        return {
            'id': self.id,
            'hub_id': self.hub_id,
            'title': self.title,
            'content_html': self.content_html,
            'created_at': self.created_at
        }
        
    @staticmethod
    def from_dict(source):
        return Note(**source)

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