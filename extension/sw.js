/**
 * Service Worker Initialization
 * Loads model loader and background scripts
 */

console.log('[FilterX] Service worker starting...');

// Load model loader first (needed by background.js)
importScripts('model_loader.js');
console.log('[FilterX] Model loader script loaded');

// Load main background script
importScripts('background.js');
console.log('[FilterX] Background script loaded and initialized');
