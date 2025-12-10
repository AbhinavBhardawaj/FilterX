# FilterX Local Model Integration Guide

## Overview
The NSFW model (`nsfw_model.tflite`) has been successfully integrated into the FilterX browser extension to enable local inference. The extension now classifies images locally first, with automatic fallback to the backend API if local inference fails.

## What Was Added/Modified

### New Files
1. **`extension/sw.js`** - Service Worker initialization script
   - Loads TensorFlow.js library from CDN
   - Loads TensorFlow.js TFLite support
   - Imports model_loader.js and background.js

2. **`extension/model_loader.js`** - Local model inference engine
   - `loadModel()` - Loads the TFLite model from extension storage
   - `classifyImageLocal(base64Image)` - Performs local image classification
   - `base64ToImageTensor(base64Image)` - Converts base64 images to tensors

### Modified Files
1. **`extension/manifest.json`**
   - Updated `background` service_worker to use `sw.js`
   - Added `model/nsfw_model.tflite` to `web_accessible_resources`

2. **`extension/background.js`**
   - Added `_useLocalModel` and `_modelReady` flags
   - Added `_initializeLocalModel()` function for async model loading
   - Updated `classifyImage()` to try local inference first
   - Automatic fallback to backend API if local inference fails

## How It Works

### 1. Model Loading (on extension installation)
```
Service Worker starts → Loads TensorFlow.js libraries → Initializes model loader
→ Asynchronously loads nsfw_model.tflite → Sets _modelReady = true
```

### 2. Image Classification Flow
```
Content Script detects image → Sends to Background Worker
↓
Background Worker checks cache
↓
If local model ready: Try local inference with TensorFlow.js
  ├─ Success: Use local prediction, return result
  └─ Failure: Fall through to backend
↓
If local failed or not ready: Use backend API
↓
Cache result and return to Content Script
```

### 3. Model Input/Output
- **Input**: Base64 encoded image or data URL
- **Preprocessing**: Resize to 224×224, normalize to [0, 1]
- **Output**: NSFW probability score (0-1)
- **Decision**: Score ≥ 0.5 → "nsfw", otherwise "safe"

## Advantages

✅ **Offline Capability**: Images can be classified without backend connection
✅ **Faster**: Local inference is faster than network round trip
✅ **Privacy**: Image data stays on user's device for local classification
✅ **Fallback**: Automatically uses backend if local inference fails
✅ **Caching**: Results cached to avoid redundant processing

## Configuration

### Enable/Disable Local Model
Edit `extension/background.js` line 4:
```javascript
let _useLocalModel = true;  // Set to false to disable local inference
```

### Adjust Classification Threshold
Edit `extension/model_loader.js` around line 148:
```javascript
label: nsfwProb >= 0.5 ? 'nsfw' : 'safe'  // Change 0.5 to adjust sensitivity
```

## Dependencies

- **TensorFlow.js v4.11.0** (loaded from CDN)
- **TensorFlow.js TFLite v0.0.1-alpha.8** (loaded from CDN)
- **Chrome Extension API** (storage, runtime, messaging)

## Testing

### 1. Check Extension Status
Open Chrome DevTools → Background page logs
Should see: `[FilterX] NSFW TFLite model loaded successfully`

### 2. Check Model Availability
Look for logs in service worker console:
```
[FilterX] Service worker initialized with TensorFlow.js and model loader
[FilterX] Loading NSFW TFLite model...
[FilterX] Model file loaded: [size] bytes
[FilterX] NSFW TFLite model loaded successfully
```

### 3. Monitor Classification
Check logs to see if images are classified locally or via backend:
```
[FilterX] Using local model for classification  // Local inference
[FilterX] Image nsfw/safe blocked/allowed (local)
```

## Troubleshooting

### Model Fails to Load
- Ensure `nsfw_model.tflite` exists in `extension/model/`
- Check that manifest.json includes the model in `web_accessible_resources`
- Verify TensorFlow.js CDN is accessible

### Local Classification Returns null
- Falls back to backend API automatically
- Check browser console for error details
- Verify image is valid JPEG/PNG base64

### Performance Issues
- Local inference may be slower on older devices
- Disable local model: set `_useLocalModel = false`
- Use backend API instead

## File Sizes
- `nsfw_model.tflite`: ~50-100 MB (depends on model)
- `sw.js`: <1 KB
- `model_loader.js`: ~8 KB
- **Total overhead**: TensorFlow.js libraries loaded from CDN

## Next Steps

1. **Test the extension** in Chrome by loading the unpacked extension
2. **Monitor logs** in the service worker console
3. **Verify predictions** match expected results
4. **Adjust threshold** in model_loader.js if needed
5. **Deploy** the updated extension

For more information about TensorFlow.js TFLite:
https://github.com/tensorflow/tfjs/tree/master/tfjs-tflite
