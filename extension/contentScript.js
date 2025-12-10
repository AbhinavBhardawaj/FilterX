const API_BASE_URL = "http://127.0.0.1:8000"; // later change to deployed URL

// Performance tuning constants
const MIN_WIDTH = 80;
const MIN_HEIGHT = 80;
const MAX_TEXT_LEN = 5000; // limit text size sent to backend

// Simple async queue so we don't block the page
const _imageQueue = [];
let _processing = false;

window.addEventListener("load", () => {
  scanImages();
  scanTextOncePerPage();
});

// throttle mutation-driven scans
let _scanPending = false;
const observer = new MutationObserver(() => {
  if (!_scanPending) {
    _scanPending = true;
    requestIdleCallback(() => {
      scanImages();
      _scanPending = false;
    }, { timeout: 500 });
  }
});
observer.observe(document.body, { childList: true, subtree: true });

function shouldCheckImage(img) {
  const w = img.naturalWidth || img.width || 0;
  const h = img.naturalHeight || img.height || 0;
  if (w < MIN_WIDTH || h < MIN_HEIGHT) return false;
  // skip hidden images
  const style = window.getComputedStyle(img);
  if (style && (style.display === 'none' || style.visibility === 'hidden' || parseFloat(style.opacity) === 0)) return false;
  return true;
}

function enqueueImage(img) {
  _imageQueue.push(img);
  if (!_processing) processQueue();
}

async function processQueue() {
  _processing = true;
  while (_imageQueue.length > 0) {
    const img = _imageQueue.shift();
    try {
      // yield to idle if available
      if (window.requestIdleCallback) {
        await new Promise((res) => requestIdleCallback(res, { timeout: 200 }));
      }
      await classifyAndApply(img);
    } catch (err) {
      console.error('Error in classifyAndApply', err);
    }
    // small delay so we yield to the UI thread
    await new Promise((r) => setTimeout(r, 10));
  }
  _processing = false;
}

// Cache classification results by image src (or data URL prefix)
const _imageCache = new Map();

// IntersectionObserver: only process images when they are near viewport
const io = new IntersectionObserver((entries) => {
  for (const entry of entries) {
    if (entry.isIntersecting) {
      const img = entry.target;
      // stop observing once queued
      io.unobserve(img);
      if (!img.dataset.filterxChecked && shouldCheckImage(img)) {
        img.dataset.filterxChecked = 'true';
        enqueueImage(img);
      }
    }
  }
}, { root: null, rootMargin: '200px', threshold: 0.05 });

function scanImages() {
  const images = document.querySelectorAll("img");
  for (const img of images) {
    if (img.dataset.filterxChecked) continue;
    if (!shouldCheckImage(img)) continue;
    try {
      io.observe(img);
    } catch (e) {
      // fallback
      img.dataset.filterxChecked = 'true';
      enqueueImage(img);
    }
  }
}

async function classifyAndApply(img) {
  try {
    const key = img.currentSrc || img.src || img.getAttribute('data-src') || null;
    if (key && _imageCache.has(key)) {
      applyResult(img, _imageCache.get(key));
      return;
    }

    const MAX_DIM = 224;
    const iw = img.naturalWidth || img.width || MAX_DIM;
    const ih = img.naturalHeight || img.height || MAX_DIM;
    const scale = Math.min(1, MAX_DIM / Math.max(1, iw, ih));
    const cw = Math.max(1, Math.round(iw * scale));
    const ch = Math.max(1, Math.round(ih * scale));

    const canvas = document.createElement("canvas");
    canvas.width = cw;
    canvas.height = ch;
    const ctx = canvas.getContext("2d");
    try {
      ctx.drawImage(img, 0, 0, cw, ch);
    } catch (err) {
      // Cross-origin image – fall back to URL classification
      if (key) {
        chrome.runtime.sendMessage({ action: "classifyUrl", url: key }, (response) => {
          if (!response) return;
          if (key) _imageCache.set(key, response);
          applyResult(img, response);
        });
      }
      return;
    }

    const base64 = canvas.toDataURL("image/jpeg", 0.7);
    chrome.runtime.sendMessage({ action: "classifyImage", imageData: base64 }, (response) => {
      if (!response) return;
      if (key) _imageCache.set(key, response);
      applyResult(img, response);
    });
  } catch (error) {
    console.error("Error processing image:", error);
  }
}

function applyResult(img, response) {
  if (!response) return;
  if (response.error) {
    console.warn('[FilterX] Error in response:', response.error);
    return;
  }
  if (response.label === "nsfw") {
    const action = response.action || "block"; // Default to block if action not specified
    if (action === "block") {
      img.style.display = "none";
      console.log('[FilterX] Blocked NSFW image');
    } else if (action === "blur") {
      img.style.filter = "blur(10px)";
      console.log('[FilterX] Blurred NSFW image');
    }
  }
}

// Only classify page text once per page load to avoid spamming the backend
let _pageTextClassified = false;
function scanTextOncePerPage() {
  if (_pageTextClassified) return;
  _pageTextClassified = true;

  try {
    const pageText = (document.body && document.body.innerText) ? document.body.innerText.slice(0, MAX_TEXT_LEN) : "";
    if (!pageText || pageText.length < 50) return;

    chrome.runtime.sendMessage({ action: "classifyText", text: pageText }, (response) => {
      if (!response) return;
      if (response.label === "nsfw") {
        if (response.action === "block") {
          // replace body content with a notice
          document.body.innerHTML = '<div style="padding:20px;font-family:Arial,sans-serif;">Content blocked by FilterX</div>';
        } else if (response.action === "blur") {
          document.body.style.filter = "blur(5px)";
        }
      }
    });
  } catch (err) {
    console.error('Error scanning page text', err);
  }
}

// Helper kept for backwards compatibility if needed elsewhere
async function classifyImage(base64Image) {
  const res = await fetch(`${API_BASE_URL}/api/classify/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ type: "image", content: base64Image }),
  });
  return await res.json();
}
