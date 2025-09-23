# Spaced Repetition System - Complete User Workflow Documentation

## 🎯 Overview

The spaced repetition system in StudyBot implements the SM-2 algorithm (similar to Anki) to help users efficiently memorize and retain information through scientifically-proven spaced intervals. This document outlines the complete user workflow from flashcard creation to advanced review sessions.

## 📚 Complete User Workflow

### 1. Flashcard Creation
**Location**: Lecture notes, study sessions, or manual creation
**Process**:
- User uploads lecture materials or creates study content
- System automatically generates flashcards using AI
- Flashcards are stored in the `activities` collection with type 'Flashcards'
- Each flashcard contains `front` (question) and `back` (answer) content

**Example**:
```json
{
  "front": "What is photosynthesis?",
  "back": "The process by which plants convert sunlight into energy"
}
```

### 2. Automatic Migration to Spaced Repetition
**Trigger**: When flashcards are created or when user accesses spaced repetition features
**Process**:
- System automatically migrates flashcards to `spaced_repetition_cards` collection
- Each card gets spaced repetition properties:
  - `ease_factor`: 2.5 (default)
  - `interval_days`: 1 (starts with 1 day)
  - `repetitions`: 0 (new card)
  - `next_review`: null (due immediately)
  - `difficulty`: 'medium' (default)

**API Endpoint**: `/api/spaced_repetition/migrate_flashcards/<activity_id>`

### 3. Starting a Review Session
**Location**: Dashboard → Progress Tab → Spaced Repetition
**Process**:
- User clicks "Start Review Session" on dashboard
- System calls `/api/spaced_repetition/create_session`
- System finds all due cards for user's hubs
- Creates a `ReviewSession` with limited cards (default: 20)
- Returns session data to frontend

**API Endpoint**: `/api/spaced_repetition/create_session`
**Request**:
```json
{
  "hub_id": "user_hub_id",
  "max_cards": 20
}
```

### 4. Card Review Interface
**Location**: `/spaced_repetition/review`
**Features**:
- Clean, Anki-style card interface
- Click to flip cards
- Four rating buttons: Again (0), Hard (1), Good (2), Easy (3)
- Real-time progress tracking
- Session statistics display

**UI Elements**:
- Progress bar showing session completion
- Card counter (current/total)
- Accuracy tracking
- Keyboard shortcuts (Space to flip, 1-4 for ratings)

### 5. Rating Cards and Algorithm Updates
**Process**:
- User rates each card after seeing the answer
- System calls `/api/spaced_repetition/review_card`
- SM-2 algorithm calculates new interval and ease factor
- Card state is updated in database

**Algorithm Logic**:
- **Again (0)**: Reset repetitions to 0, interval to 1 day, decrease ease factor
- **Hard (1)**: Reset repetitions to 0, interval to 1 day, decrease ease factor
- **Good (2)**: Increment repetitions, calculate new interval, maintain ease factor
- **Easy (3)**: Increment repetitions, calculate new interval, increase ease factor

**API Endpoint**: `/api/spaced_repetition/review_card`
**Request**:
```json
{
  "card_id": "card_id",
  "quality_rating": 2
}
```

### 6. Session Completion
**Process**:
- After reviewing all cards, system calls `/api/spaced_repetition/complete_session`
- Session statistics are calculated and saved
- User earns XP based on accuracy
- Hub progress is updated

**API Endpoint**: `/api/spaced_repetition/complete_session`
**Request**:
```json
{
  "session_id": "session_id",
  "cards_reviewed": 20,
  "correct_count": 16,
  "incorrect_count": 4
}
```

### 7. Dashboard Integration
**Location**: Dashboard → Progress Tab
**Features**:
- Real-time spaced repetition statistics
- Due cards counter
- Recent session history
- Learning progress visualization
- Quick access to review sessions

**Statistics Displayed**:
- Total cards across all hubs
- Cards due for review
- New cards (never reviewed)
- Learning cards (1-2 repetitions)
- Mature cards (21+ day intervals)
- Retention rate
- Average ease factor

### 8. User Settings and Customization
**Location**: `/spaced_repetition/settings`
**Configurable Options**:
- New cards per day (default: 20)
- Maximum reviews per day (default: 200)
- Easy bonus multiplier (default: 1.3)
- Interval modifier (default: 1.0)
- Maximum interval (default: 36500 days)
- Learning steps (default: "1 10")
- Lapse steps (default: "10")
- Lapse interval (default: 0.1)

**API Endpoint**: `/api/spaced_repetition/user_settings`

### 9. Advanced Features

#### Learning Progress Tracking
**API Endpoint**: `/api/spaced_repetition/learning_progress/<hub_id>`
**Data Provided**:
- Card categorization (new, learning, review, mature)
- Difficulty distribution
- Average ease factor
- Retention rate
- Overdue cards count

#### Recent Sessions History
**API Endpoint**: `/api/spaced_repetition/recent_sessions/<hub_id>`
**Data Provided**:
- Session timestamps
- Cards reviewed
- Accuracy percentages
- Session durations

#### System Health Monitoring
**API Endpoint**: `/api/spaced_repetition/system_health`
**Monitoring**:
- System performance metrics
- Error tracking
- Log file information
- Spaced repetition statistics

## 🔧 Technical Implementation

### Database Collections

#### `spaced_repetition_cards`
```json
{
  "id": "card_id",
  "activity_id": "original_flashcard_activity_id",
  "card_index": 0,
  "front": "Question text",
  "back": "Answer text",
  "ease_factor": 2.5,
  "interval_days": 1,
  "repetitions": 0,
  "last_reviewed": "2025-09-23T19:30:00Z",
  "next_review": "2025-09-24T19:30:00Z",
  "difficulty": "medium",
  "created_at": "2025-09-23T19:30:00Z"
}
```

#### `review_sessions`
```json
{
  "id": "session_id",
  "user_id": "user_id",
  "hub_id": "hub_id",
  "session_type": "spaced_repetition",
  "cards_reviewed": 20,
  "correct_count": 16,
  "incorrect_count": 4,
  "session_duration_minutes": 15,
  "started_at": "2025-09-23T19:30:00Z",
  "completed_at": "2025-09-23T19:45:00Z",
  "cards_data": ["card_id_1", "card_id_2", ...]
}
```

#### `user_spaced_repetition_settings`
```json
{
  "id": "settings_id",
  "user_id": "user_id",
  "new_cards_per_day": 20,
  "max_reviews_per_day": 200,
  "easy_bonus": 1.3,
  "interval_modifier": 1.0,
  "max_interval": 36500,
  "graduated_interval": 1,
  "learning_steps": "1 10",
  "lapse_interval": 0.1,
  "lapse_steps": "10",
  "min_interval": 1,
  "created_at": "2025-09-23T19:30:00Z"
}
```

### SM-2 Algorithm Implementation

The system implements the SM-2 algorithm with the following logic:

```python
def calculate_next_review(self, quality_rating):
    """Calculate next review based on SM-2 algorithm"""
    previous_interval = self.interval_days
    
    if quality_rating <= 1:  # Again or Hard
        self.repetitions = 0
        self.interval_days = 1
    else:  # Good or Easy
        self.repetitions += 1
        if self.repetitions == 1:
            self.interval_days = 1
        elif self.repetitions == 2:
            self.interval_days = 6
        else:
            self.interval_days = int(previous_interval * self.ease_factor)
    
    # Update ease factor
    if quality_rating == 0:  # Again
        self.ease_factor = max(1.3, self.ease_factor - 0.2)
    elif quality_rating == 1:  # Hard
        self.ease_factor = max(1.3, self.ease_factor - 0.15)
    elif quality_rating == 2:  # Good
        self.ease_factor = self.ease_factor  # No change
    else:  # Easy
        self.ease_factor = min(2.5, self.ease_factor + 0.15)
    
    # Set next review date
    self.last_reviewed = datetime.now(timezone.utc)
    self.next_review = self.last_reviewed + timedelta(days=self.interval_days)
```

## 🎯 User Experience Flow

### First-Time User
1. **Create Content**: Upload lecture materials or create study notes
2. **Generate Flashcards**: System automatically creates flashcards
3. **Start Review**: Click "Start Review Session" on dashboard
4. **Review Cards**: Rate cards based on difficulty
5. **Track Progress**: View statistics and progress on dashboard

### Regular User
1. **Daily Review**: Check dashboard for due cards
2. **Quick Sessions**: Review 10-20 cards per session
3. **Monitor Progress**: Track learning statistics
4. **Adjust Settings**: Customize review parameters as needed

### Power User
1. **Advanced Settings**: Fine-tune algorithm parameters
2. **Bulk Operations**: Manage large flashcard sets
3. **Analytics**: Deep dive into learning patterns
4. **System Health**: Monitor system performance

## 🔍 Debug and Monitoring

### Debug Logging
The system includes comprehensive debug logging:
- API call tracking
- Algorithm calculations
- Session events
- Performance metrics
- Error handling

### Debug Endpoints
- `/api/spaced_repetition/system_health`: System status
- `/api/spaced_repetition/debug_report`: Comprehensive debug report

### Log Files
- `logs/spaced_repetition.log`: Main system log
- `logs/debug_report_*.json`: Debug reports
- `logs/error_*.json`: Error logs

## ✅ Testing Results

All tests have passed successfully:
- ✅ Core models working correctly
- ✅ Algorithm calculations accurate
- ✅ Session management functional
- ✅ API endpoints working
- ✅ Dashboard integration complete
- ✅ Edge cases handled properly
- ✅ Debug logging comprehensive
- ✅ User workflow complete

## 🚀 Production Readiness

The spaced repetition system is fully functional and ready for production use. All components have been tested and verified:

- **Models**: SpacedRepetitionCard, ReviewSession, UserSpacedRepetitionSettings
- **API Endpoints**: Complete CRUD operations
- **Frontend**: Responsive review interface
- **Dashboard**: Integrated progress tracking
- **Settings**: User customization options
- **Monitoring**: System health and debugging
- **Documentation**: Complete workflow documentation

The system provides a complete Anki-style spaced repetition experience integrated seamlessly into the StudyBot platform.
