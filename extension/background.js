const API_BASE_URL = "http://127.0.0.1:8000";

// Simple in-memory cache for recent classifications
const _cache = new Map();

// Concurrency and queueing to avoid too many simultaneous network requests
// Will be configured by the popup 'aggressiveness' setting.
let MAX_CONCURRENT = 2; // default to 'normal' (2 concurrent requests)
let _inFlight = 0;
const _requestQueue = [];

function _enqueueRequest(fn) {
  return new Promise((resolve, reject) => {
    _requestQueue.push({ fn, resolve, reject });
    _processQueue();
  });
}

function _processQueue() {
  while (_inFlight < MAX_CONCURRENT && _requestQueue.length > 0) {
    const job = _requestQueue.shift();
    _inFlight += 1;
    job.fn()
      .then((res) => job.resolve(res))
      .catch((err) => job.reject(err))
      .finally(() => {
        _inFlight -= 1;
        // process next jobs
        setTimeout(_processQueue, 0);
      });
  }
}

// Initialize storage
chrome.runtime.onInstalled.addListener(() => {
  chrome.storage.local.get(["filterxEnabled", "stats", "activityLog", "aggressiveness"], (items) => {
    const defaults = {};
    if (typeof items.filterxEnabled === 'undefined') defaults.filterxEnabled = true;
    if (typeof items.stats === 'undefined') defaults.stats = { blocked: 0, blurred: 0, total: 0 };
    if (typeof items.activityLog === 'undefined') defaults.activityLog = [];
    // default aggressiveness to 'normal' if missing
    if (typeof items.aggressiveness === 'undefined') defaults.aggressiveness = 'normal';
    if (Object.keys(defaults).length > 0) chrome.storage.local.set(defaults);
    // apply aggressiveness setting immediately
    const ag = items.aggressiveness || defaults.aggressiveness || 'normal';
    _applyAggressiveness(ag);
  });
});

// Listen for aggressiveness changes and update concurrency dynamically
chrome.storage.onChanged.addListener((changes, area) => {
  if (area !== 'local') return;
  if (changes.aggressiveness) {
    const newVal = changes.aggressiveness.newValue;
    _applyAggressiveness(newVal);
  }
});

function _applyAggressiveness(value) {
  // map setting to concurrency
  if (value === 'aggressive') {
    MAX_CONCURRENT = 3;
  } else if (value === 'light') {
    MAX_CONCURRENT = 1;
  } else {
    // normal
    MAX_CONCURRENT = 2;
  }
  console.info('FilterX: set aggressiveness=', value, 'MAX_CONCURRENT=', MAX_CONCURRENT);
  // try to process queue in case we increased concurrency
  setTimeout(_processQueue, 0);
}

// Listen for messages from content script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "classifyImage") {
    // Use queue to limit concurrency
    _enqueueRequest(() => classifyImage(request.imageData)).then(result => sendResponse(result)).catch(err => sendResponse({ error: String(err) }));
    return true; // Keep the message channel open for async response
  } else if (request.action === "classifyText") {
    _enqueueRequest(() => classifyText(request.text)).then(result => sendResponse(result)).catch(err => sendResponse({ error: String(err) }));
    return true;
  } else if (request.action === "classifyUrl") {
    _enqueueRequest(() => classifyUrl(request.url)).then(result => sendResponse(result)).catch(err => sendResponse({ error: String(err) }));
    return true;
  }
});

async function classifyImage(base64Image) {
  try {
    // use short key to avoid storing huge strings
    const key = base64Image.slice(0, 200);
    if (_cache.has(key)) return _cache.get(key);

    const res = await fetch(`${API_BASE_URL}/api/classify/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ type: "image", content: base64Image })
    });
    const result = await res.json();
    _cache.set(key, result);
    await _scheduleLog(`Image ${result.label === "nsfw" ? "blocked" : "allowed"}`);
    return result;
  } catch (error) {
    console.error("Error classifying image:", error);
    return { error: String(error) };
  }
}

async function classifyText(text) {
  try {
    const key = `text:${text.slice(0,200)}`;
    if (_cache.has(key)) return _cache.get(key);
    const res = await fetch(`${API_BASE_URL}/api/classify/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ type: "text", content: text })
    });
    const result = await res.json();
    _cache.set(key, result);
    await _scheduleLog(`Text ${result.label === "nsfw" ? "blocked" : "allowed"}`);
    return result;
  } catch (error) {
    console.error("Error classifying text:", error);
    return { error: String(error) };
  }
}

async function classifyUrl(url) {
  try {
    if (_cache.has(url)) return _cache.get(url);
    const res = await fetch(`${API_BASE_URL}/api/classify/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ type: "url", content: url })
    });
    const result = await res.json();
    _cache.set(url, result);
    await _scheduleLog(`URL ${result.label === "nsfw" ? "blocked" : "allowed"}`);
    return result;
  } catch (error) {
    console.error("Error classifying URL:", error);
    return { error: String(error) };
  }
}

// Batch logs so we don't write to storage for every single event
const _pendingLogs = [];
let _logFlushTimer = null;

function _scheduleLog(action) {
  _pendingLogs.push({ action, timestamp: Date.now() });
  if (!_logFlushTimer) {
    _logFlushTimer = setTimeout(_flushLogs, 1500);
  }
}

async function _flushLogs() {
  _logFlushTimer = null;
  if (_pendingLogs.length === 0) return;

  try {
    const toFlush = _pendingLogs.splice(0, _pendingLogs.length);
    const { activityLog = [], stats = {} } = await chrome.storage.local.get(["activityLog", "stats"]);
    activityLog.push(...toFlush);

    // update stats based on flushed actions
    for (const entry of toFlush) {
      stats.total = (stats.total || 0) + 1;
      if (entry.action && entry.action.includes("blocked")) {
        stats.blocked = (stats.blocked || 0) + 1;
      } else if (entry.action && entry.action.includes("blurred")) {
        stats.blurred = (stats.blurred || 0) + 1;
      }
    }

    await chrome.storage.local.set({ activityLog, stats });
  } catch (e) {
    console.error('Failed to flush logs', e);
  }
}
