/**
 * Local NSFW Model Loader
 * Uses TensorFlow.js to load and run the TFLite model locally
 */

let _modelInstance = null;
let _modelLoading = false;
let _modelLoadPromise = null;

/**
 * Load the TFLite model using TensorFlow.js
 * @returns {Promise<Object>} The loaded model interpreter
 */
async function loadModel() {
  // Return cached model if already loaded
  if (_modelInstance) {
    return _modelInstance;
  }

  // Return existing loading promise if model is currently loading
  if (_modelLoading) {
    return _modelLoadPromise;
  }

  _modelLoading = true;

  _modelLoadPromise = (async () => {
    try {
      // Ensure TensorFlow.js is loaded
      if (typeof tf === 'undefined') {
        throw new Error('TensorFlow.js is not loaded');
      }

      console.log('[FilterX] Loading NSFW TFLite model...');

      // Get the model file path using chrome.runtime.getURL
      const modelPath = chrome.runtime.getURL('model/nsfw_model.tflite');
      console.log('[FilterX] Model path:', modelPath);

      // Fetch the model file
      const response = await fetch(modelPath);
      if (!response.ok) {
        throw new Error(`Failed to fetch model: ${response.status} ${response.statusText}`);
      }

      const modelBuffer = await response.arrayBuffer();
      console.log(`[FilterX] Model file loaded: ${modelBuffer.byteLength} bytes`);

      // Load the TFLite model
      // Using tf.lite.loadTFLiteModel with a Uint8Array
      _modelInstance = await tf.lite.loadTFLiteModel(
        tf.io.fromMemory({
          modelTopology: undefined,
          weightData: new Uint8Array(modelBuffer),
        })
      );

      console.log('[FilterX] NSFW TFLite model loaded successfully');
      return _modelInstance;
    } catch (error) {
      console.error('[FilterX] Failed to load NSFW model:', error);
      _modelInstance = null;
      _modelLoading = false;
      _modelLoadPromise = null;
      throw error;
    } finally {
      _modelLoading = false;
    }
  })();

  return _modelLoadPromise;
}

/**
 * Convert base64 image to tensor
 * @param {string} base64Image - Base64 encoded image (with or without data URL prefix)
 * @returns {Promise<tf.Tensor3D>} Image tensor
 */
async function base64ToImageTensor(base64Image) {
  return new Promise((resolve, reject) => {
    try {
      // Remove data URL prefix if present
      const b64 = base64Image.includes(',') ? base64Image.split(',')[1] : base64Image;
      
      // Create an image element
      const img = new Image();
      img.onload = () => {
        try {
          // Get tensor from image
          let tensor = tf.browser.fromPixels(img, 3);
          resolve(tensor);
        } catch (err) {
          reject(err);
        }
      };
      img.onerror = () => reject(new Error('Failed to load image'));
      
      // Set the base64 data as image source
      img.src = `data:image/jpeg;base64,${b64}`;
    } catch (error) {
      reject(error);
    }
  });
}

/**
 * Classify image using local TFLite model
 * @param {string} base64Image - Base64 encoded image
 * @returns {Promise<Object|null>} Classification result {label, confidence} or null if failed
 */
async function classifyImageLocal(base64Image) {
  let tensor = null;
  let resized = null;
  let normalized = null;
  let batched = null;
  let predictions = null;

  try {
    const model = await loadModel();
    
    console.log('[FilterX] Starting local image classification...');

    // Convert base64 to tensor
    tensor = await base64ToImageTensor(base64Image);
    console.log('[FilterX] Image tensor shape:', tensor.shape);

    // Resize to 224x224 (standard size for NSFW models)
    resized = tf.image.resizeBilinear(tensor, [224, 224]);

    // Normalize to [0, 1]
    normalized = resized.div(255.0);

    // Add batch dimension
    batched = tf.expandDims(normalized, 0);

    console.log('[FilterX] Input tensor shape:', batched.shape);

    // Run inference
    predictions = model.predict(batched);

    // Get output data
    const outputData = await predictions.data();
    const outputArray = Array.from(outputData);
    
    console.log('[FilterX] Model output:', outputArray);

    // Interpret the output
    // Most NSFW models output either:
    // 1. Single value: probability of NSFW
    // 2. Two values: [safe_prob, nsfw_prob]
    let nsfwProb = 0;

    if (outputArray.length === 1) {
      // Single output - assume it's NSFW probability
      nsfwProb = outputArray[0];
    } else if (outputArray.length === 2) {
      // Two outputs - take the second one (NSFW probability)
      nsfwProb = outputArray[1];
    } else {
      // Multiple outputs - take the last one or the max
      nsfwProb = Math.max(...outputArray);
    }

    // Clamp to [0, 1]
    nsfwProb = Math.max(0, Math.min(1, nsfwProb));

    console.log('[FilterX] NSFW probability:', nsfwProb);

    // Determine action based on label (matching backend behavior)
    let action = 'allow';
    if (nsfwProb >= 0.5) {
      action = 'block'; // Block NSFW content
    }

    return {
      label: nsfwProb >= 0.5 ? 'nsfw' : 'safe',
      confidence: nsfwProb,
      action: action,
      reason: 'TFLite local classifier (image)',
      local: true,
    };
  } catch (error) {
    console.error('[FilterX] Local classification error:', error);
    return null;
  } finally {
    // Clean up tensors to prevent memory leaks
    if (tensor) tensor.dispose();
    if (resized) resized.dispose();
    if (normalized) normalized.dispose();
    if (batched) batched.dispose();
    if (predictions) predictions.dispose();
  }
}

// Export for use in background.js
if (typeof self !== 'undefined' && !self.window) {
  // In service worker context
  self.loadModel = loadModel;
  self.classifyImageLocal = classifyImageLocal;
}
