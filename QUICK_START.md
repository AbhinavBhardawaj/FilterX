# FilterX - Quick Start Guide

## 🎯 What's Done

✅ **Backend API** - Fully functional Django REST API  
✅ **Extension** - Complete Chrome extension with content scanning  
✅ **Classification Service** - Dummy classifiers for text, images, and URLs  
✅ **Real-time Filtering** - Block/blur NSFW content automatically  
✅ **Statistics Tracking** - Track blocked, blurred, and scanned items  

## ⚡ Quick Start (30 seconds)

### 1. Start Backend
```bash
cd "/Users/atulkumar/Desktop/filterx demo/backend"
python3 manage.py runserver 0.0.0.0:8000
```

**Backend running at:** `http://127.0.0.1:8000`  
**API available at:** `http://127.0.0.1:8000/api/`

### 2. Load Extension
1. Open Chrome: `chrome://extensions/`
2. Toggle **Developer mode** (top right)
3. Click **Load unpacked**
4. Select: `/Users/atulkumar/Desktop/filterx demo/extension`
5. Done! The extension is now active

### 3. Test It
- Visit any website
- Click the FilterX icon in your toolbar
- See the statistics and activity log
- Images and text are automatically scanned and filtered

## 📡 API Endpoints

### Health Check
```bash
curl http://127.0.0.1:8000/api/health/
```

### Classify Content
```bash
# NSFW Text (will be blurred)
curl -X POST http://127.0.0.1:8000/api/classify/ \
  -H "Content-Type: application/json" \
  -d '{"type": "text", "content": "porn"}'

# NSFW URL (will be blocked)
curl -X POST http://127.0.0.1:8000/api/classify/ \
  -H "Content-Type: application/json" \
  -d '{"type": "url", "content": "https://xvideos.com"}'

# Safe Content (will be allowed)
curl -X POST http://127.0.0.1:8000/api/classify/ \
  -H "Content-Type: application/json" \
  -d '{"type": "text", "content": "hello world"}'
```

## 🧠 How It Works

### Text Scanning
- Extracts all text from webpage
- Checks for NSFW keywords: "xxx", "sex", "porn", "nude", "boobs", "nsfw"
- Blurs detected NSFW content

### URL Scanning  
- Checks URLs against blocklist
- Blocklist: "porn", "xvideos", "xnxx", "redtube", "xhampster"
- Blocks detected porn domains

### Image Scanning
- Converts images to base64
- Sends to backend for classification
- Blocks or blurs based on response

### Content Filtering Actions
- **Block** (`display: none`) - Content is hidden
- **Blur** (`filter: blur()`) - Content is blurred  
- **Allow** - Content is left as-is

## 📊 Extension Features

### Popup Interface
Click the FilterX icon to see:
- **Status** - Enabled/Disabled toggle
- **Statistics**
  - Blocked items count
  - Blurred items count
  - Total scanned count
- **Activity Log** - Recent 10 actions with timestamps
- **Clear History** - Reset statistics

### Automatic Scanning
- Scans on page load
- Monitors dynamic content (MutationObserver)
- Real-time filtering as you browse

## 🔧 Configuration

### API Base URL
Edit these files to change backend URL:
- `extension/background.js` (line 1)
- `extension/contentScript.js` (line 1)

```javascript
const API_BASE_URL = "http://127.0.0.1:8000";
```

### Classification Keywords
Edit `backend/filterx/services/nsfw_classifier.py`:
```python
NSFW_KEYWORDS = ["xxx", "sex", "porn", "nude", "boobs", "nsfw"]
BLOCKLIST = ["porn", "xvideos", "xnxx", "redtube", "xhampster"]
```

## 🚀 Next Steps

### To Upgrade with Real ML Model
Your friend will:
1. Install TensorFlow: `pip install tensorflow`
2. Replace `DummyNSFWModel` in `backend/filterx/services/nsfw_classifier.py`
3. Load MediaNet model: `tf.keras.models.load_model(...)`

**The API and extension don't need ANY changes!**

### To Deploy to Production
1. Set `DEBUG = False` in `backend/settings.py`
2. Update `SECRET_KEY` 
3. Restrict `ALLOWED_HOSTS` and `CORS_ALLOWED_ORIGINS`
4. Deploy to cloud (AWS, Google Cloud, Azure, Heroku)
5. Update API URL in extension

### To Publish to Chrome Web Store
1. Create Google account
2. Pay $5 developer fee
3. Upload extension
4. Submit for review (usually 1-3 hours)

## 🧪 Testing Checklist

- ✅ Backend starts without errors
- ✅ Health endpoint works
- ✅ Text classification works (NSFW and safe)
- ✅ URL classification works (blocked and safe)
- ✅ Extension loads in Chrome
- ✅ Extension can reach backend API
- ✅ Content is scanned and filtered
- ✅ Statistics update correctly
- ✅ Activity log shows recent actions

## 📁 File Locations

| File | Location | Purpose |
|------|----------|---------|
| Django Server | `backend/manage.py` | Start backend: `python3 manage.py runserver` |
| API Views | `backend/filterx/views.py` | ClassifyView, HealthCheckView |
| Classifier | `backend/filterx/services/nsfw_classifier.py` | Classification logic |
| Extension | `extension/` | Chrome extension files |
| Background Script | `extension/background.js` | Service worker, message handler |
| Content Script | `extension/contentScript.js` | Page scanner, content modifier |
| Popup UI | `extension/popup.html` | User interface |
| Manifest | `extension/manifest.json` | Extension configuration |

## 🆘 Troubleshooting

### Backend won't start
```bash
# Check if port 8000 is in use
lsof -i :8000

# Kill existing process
kill -9 <PID>
```

### Extension can't connect to backend
- Ensure backend is running: `http://127.0.0.1:8000`
- Check CORS is enabled in settings
- Check browser console (F12) for errors

### Images not being scanned
- Check browser console for JavaScript errors
- Ensure images are fully loaded
- Check contentScript runs at "document_end"

### "Couldn't import Django" error
```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 📞 Support

For full documentation, see:
- `SETUP_GUIDE.md` - Complete setup instructions
- `IMPLEMENTATION_SUMMARY.md` - Technical details
- Django Docs: https://docs.djangoproject.com
- Chrome Extension Docs: https://developer.chrome.com/docs/extensions

## 🎉 You're Ready!

1. Backend running ✅
2. Extension loaded ✅
3. Tests passing ✅
4. Ready to browse safely ✅

Happy filtering! 🛡️
