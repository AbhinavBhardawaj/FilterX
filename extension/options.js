// options.js - external script for options.html

// helper to toggle class
function setActiveMode(mode){
  ['strict','balanced','lenient'].forEach(m => {
    const el = document.getElementById('mode-'+m);
    if(el) el.classList.toggle('active', m===mode);
  });
}
function setSmallSwitch(id,on){
  const el = document.getElementById(id);
  if(!el) return;
  el.classList.toggle('on', !!on);
}

// load stored settings into UI
chrome.storage.sync.get({
  mode: 'balanced',
  blockExplicit: true,
  blurSuggestive: true,
  filterText: true,
  backendUrl: 'http://localhost:8000'
}, data => {
  setActiveMode(data.mode);
  setSmallSwitch('blockExplicit', data.blockExplicit);
  setSmallSwitch('blurSuggestive', data.blurSuggestive);
  setSmallSwitch('filterText', data.filterText);
  const backendEl = document.getElementById('backend');
  if(backendEl) backendEl.value = data.backendUrl || '';
});

// mode clicks
['strict','balanced','lenient'].forEach(m => {
  const el = document.getElementById('mode-'+m);
  if(!el) return;
  el.addEventListener('click', () => {
    chrome.storage.sync.set({ mode: m }, () => setActiveMode(m));
  });
});

// small toggles
['blockExplicit','blurSuggestive','filterText'].forEach(id => {
  const el = document.getElementById(id);
  if(!el) return;
  el.addEventListener('click', () => {
    chrome.storage.sync.get(id, d => {
      const next = !d[id];
      chrome.storage.sync.set({ [id]: next }, () => setSmallSwitch(id, next));
    });
  });
});

// save button (backend url)
const saveBtn = document.getElementById('save');
if (saveBtn) {
  saveBtn.addEventListener('click', () => {
    const backendEl = document.getElementById('backend');
    const backendUrl = backendEl ? backendEl.value.trim() : '';
    chrome.storage.sync.get({ mode: 'balanced' }, cur => {
      // persist backend + (mode already saved on click) but save all to be safe
      chrome.storage.sync.set({
        backendUrl,
        mode: cur.mode
      }, () => {
        // notify background of new backend if needed
        chrome.runtime.sendMessage({ type: 'backend-updated', backendUrl });
        alert('Saved');
      });
    });
  });
}
