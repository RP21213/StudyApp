# Google Drive Integration Setup

## Environment Variables Required

Add these to your `.env` file:

```
GOOGLE_CLIENT_ID=your_google_client_id_here
GOOGLE_CLIENT_SECRET=your_google_client_secret_here
GOOGLE_REDIRECT_URI=http://127.0.0.1:5000/google/callback
```

## Google Cloud Console Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the Google Drive API
4. Go to "Credentials" and create OAuth 2.0 Client ID
5. Set the authorized redirect URI to: `http://127.0.0.1:5000/google/callback`
6. Copy the Client ID and Client Secret to your `.env` file

## Features Added

### Backend Integration
- Google Drive OAuth authentication
- Google Docs and Slides import functionality
- Automatic file conversion (Google Docs → DOCX, Google Slides → PPTX)
- Integration with existing file processing pipeline

### Frontend Integration
- Google Drive button in hub interface
- Modal for Google Drive file selection
- Real-time file import with progress indicators
- Seamless integration with existing study tools

### Supported File Types
- Google Docs (exported as DOCX)
- Google Slides (exported as PPTX)

### Integration with Existing Features
- ✅ File upload and processing
- ✅ AI-powered note generation
- ✅ Flashcard creation
- ✅ Quiz generation
- ✅ Study session planning
- ✅ Assignment helper
- ✅ All existing study tools

## Usage

1. Click the "Google Drive" button in your hub
2. Connect your Google Drive account (one-time setup)
3. Browse and select Google Docs or Slides to import
4. Files are automatically converted and added to your hub
5. Use imported files with all existing study tools and features
