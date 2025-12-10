 # FilterX - NSFW Content Filter Extension Setup Guide

## Project Overview

FilterX is a Chrome extension that detects and filters NSFW content (images, text, URLs) in real-time using AI/ML models. The project consists of:

1. **Backend**: Django REST API with NSFW classification endpoints
2. **Extension**: Chrome extension that scans pages and sends content to the backend for classification

## Backend Setup

### Prerequisites
- Python 3.9+
- pip

### Installation

1. **Create Virtual Environment**
```bash
cd filterx
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install Dependencies**
```bash
pip install -r backend/requirements.txt
```

Or manually:
```bash
pip install Django==4.2.27 djangorestframework django-cors-headers pillow
```

3. **Run Migrations**
```bash
cd backend
python3 manage.py migrate
```

4. **Start Development Server**
```bash
python3 manage.py runserver 0.0.0.0:8000
```

The server should start at `http://127.0.0.1:8000/`

### Backend Endpoints

#### Health Check
```
GET /api/health/
```
Response:
```json
{"status": "ok"}
```

#### Classify Content
```
POST /api/classify/
```

**Request:**
```json
{
  "type": "image|text|url",
  "content": "base64_image|plain_text|url_string"
}
```

**Response:**
```json
{
  "label": "safe|nsfw",
  "confidence": 0.95,
  "action": "allow|blur|block",
  "reason": "Classification reason"
}
```

**Examples:**

Text Classification:
```bash
curl -X POST http://127.0.0.1:8000/api/classify/ \
  -H "Content-Type: application/json" \
  -d '{"type": "text", "content": "this is some random porn site"}'
```

Response:
```json
{
  "label": "nsfw",
  "confidence": 0.9,
  "action": "blur",
  "reason": "Keyword-based text classifier (dummy)"
}
```

URL Classification:
```bash
curl -X POST http://127.0.0.1:8000/api/classify/ \
  -H "Content-Type: application/json" \
  -d '{"type": "url", "content": "https://xvideos.com/something"}'
```

Response:
```json
{
  "label": "nsfw",
  "confidence": 0.95,
  "action": "block",
  "reason": "URL matched NSFW blocklist"
}
```

## Chrome Extension Setup

### Installation

1. **Load Unpacked Extension**
   - Open Chrome and go to `chrome://extensions/`
   - Enable "Developer mode" (top right)
   - Click "Load unpacked"
   - Select the `extension` folder from this project

2. **Configure API URL**
   - In `extension/contentScript.js` and `extension/background.js`, the API base URL is set to `http://127.0.0.1:8000`
   - Change this if your backend is hosted elsewhere

### How It Works

1. **Content Detection**: When you visit a webpage, the extension:
   - Extracts all images and sends them as base64 to the backend
   - Extracts text segments and classifies them
   - Checks URLs on the page

2. **Backend Classification**: For each piece of content:
   - Dummy classifier checks against NSFW patterns
   - Returns action: `allow`, `blur`, or `block`

3. **Content Filtering**: Based on the backend response:
   - **Block**: Image/text is hidden (`display: none`)
   - **Blur**: Image/text is blurred with CSS filter
   - **Allow**: Content is left as-is

4. **Statistics Tracking**: The extension maintains:
   - Total items scanned
   - Items blocked
   - Items blurred
   - Activity log with timestamps

### Popup Interface

Click the FilterX icon to see:
- Current status (Enabled/Disabled)
- Toggle to enable/disable filtering
- Statistics (blocked, blurred, total scanned)
- Recent activity log
- Clear history button

## Project Structure

```
filterx/
├── backend/
│   ├── manage.py
│   ├── requirements.txt
│   ├── backend/
│   │   ├── __init__.py
│   │   ├── settings.py      # Django configuration
│   │   ├── urls.py          # URL routing
│   │   └── wsgi.py
│   └── filterx/
│       ├── migrations/
│       ├── __init__.py
│       ├── models.py        # Database models
│       ├── views.py         # API views
│       ├── urls.py          # App URL routing
│       └── services/
│           ├── __init__.py
│           └── nsfw_classifier.py  # Classification logic
│
├── extension/
│   ├── manifest.json        # Extension manifest
│   ├── background.js        # Background service worker
│   ├── contentScript.js     # Page content processor
│   ├── popup.html           # Popup UI
│   ├── popup.js             # Popup logic
│   ├── popup.css            # Popup styling
│   └── icons/               # Extension icons
│
└── SETUP_GUIDE.md           # This file
```

## Configuration

### Django Settings (backend/backend/settings.py)

Key settings:
- `DEBUG = True` for development
- `ALLOWED_HOSTS = ['*']` allows all hosts (for development)
- `CORS_ALLOW_ALL_ORIGINS = True` allows requests from the extension

### NSFW Classifier (backend/filterx/services/nsfw_classifier.py)

Currently uses dummy models:
- **Text**: Keyword-based detection (checks for NSFW keywords)
- **Image**: Random scoring (30% chance of NSFW)
- **URL**: Blocklist-based detection (checks against known porn domains)

### Integrating Real ML Models (MediaNet)

To replace the dummy classifier with a real TensorFlow MediaNet model:

1. **Update nsfw_classifier.py**:
```python
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model("path/to/medianet_model")

class NSFWModel:
    def predict_image(self, image_bytes: bytes) -> Tuple[Label, float]:
        # Preprocess bytes to tensor
        # Run model.predict()
        # Return ("safe"/"nsfw", confidence_score)
        ...
```

2. **Rest of the API stays the same** - no changes needed to extension or views!

## Testing

### Test Backend Endpoints

All endpoints are working correctly as demonstrated by:

✅ Health check: `curl http://127.0.0.1:8000/api/health/`
✅ Text NSFW: Detects "porn", "sex", "xxx", "nude", etc.
✅ Text Safe: Allows normal content
✅ URL blocking: Checks against porn domains
✅ Image classification: Ready for real model integration

### Test Extension

1. Go to `chrome://extensions/`
2. Enable developer mode
3. Load unpacked extension from `extension/` folder
4. Visit any webpage to see the extension in action
5. Click the FilterX icon to see statistics and activity log

## Troubleshooting

### Backend won't start
- Ensure port 8000 is not in use: `lsof -i :8000`
- Kill existing process: `kill -9 <PID>`
- Check Python version: `python3 --version` (should be 3.9+)

### Extension can't reach backend
- Ensure backend is running on `http://127.0.0.1:8000`
- Check CORS is enabled in settings
- Browser console might show CORS errors - this is expected in development

### Images not being scanned
- Check browser console for errors
- Ensure images are fully loaded before scanning
- The extension waits for `window.load` event

## Next Steps

1. **Integrate MediaNet Model**: Replace dummy classifiers with real TensorFlow models
2. **Add Database Logging**: Store classification history in the database
3. **User Settings**: Add UI for adjusting sensitivity and filtering rules
4. **Performance**: Optimize image processing and batch API requests
5. **Deployment**: Deploy to cloud service (AWS, Google Cloud, Azure)
6. **Chrome Web Store**: Publish extension to Chrome Web Store

## CORS Configuration

The backend allows all origins for development:
```python
CORS_ALLOW_ALL_ORIGINS = True
```

For production, restrict to specific origins:
```python
CORS_ALLOWED_ORIGINS = [
    "https://yourdomain.com",
]
```

## Security Notes

⚠️ **Development Only**:
- `DEBUG = True` - disable in production
- `SECRET_KEY` - change the default value
- `ALLOWED_HOSTS = ['*']` - restrict in production
- CORS allows all origins - restrict in production

## License

This project is provided as-is for educational purposes.

## Support

For issues or questions, check:
1. Browser console for errors (F12)
2. Django server logs for API errors
3. `nohup.out` or `/tmp/django.log` for server output
