// dashboard.js - external script for dashboard.html

document.addEventListener('DOMContentLoaded', () => {
  // tabs switching
  document.querySelectorAll('.tab').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
      btn.classList.add('active');
      const tab = btn.dataset.tab;
      document.querySelectorAll('.panel').forEach(s => s.classList.remove('active'));
      document.getElementById(tab).classList.add('active');
    });
  });

  // initialize values from storage
  chrome.storage.sync.get({
    enabled: true, blockedCount: 0, blurredCount: 0,
    mode: 'balanced', blockExplicit: true, blurSuggestive: true, filterText: true, backendUrl: 'http://localhost:8000'
  }, data => {
    setToggleVisual(data.enabled);
    document.getElementById('statBlocked').innerText = data.blockedCount || 0;
    document.getElementById('statBlurred').innerText = data.blurredCount || 0;

    document.getElementById('uiBackend').value = data.backendUrl || '';
    document.querySelectorAll('.mode').forEach(m => m.classList.toggle('active', m.dataset.mode === data.mode));
    document.getElementById('uiBlockExplicit').checked = !!data.blockExplicit;
    document.getElementById('uiBlurSuggestive').checked = !!data.blurSuggestive;
    document.getElementById('uiFilterText').checked = !!data.filterText;
  });

  // demo toggle click
  const demoToggle = document.getElementById('demoToggle');
  if (demoToggle) {
    demoToggle.addEventListener('click', () => {
      chrome.storage.sync.get({ enabled: true }, d => {
        const next = !d.enabled;
        chrome.storage.sync.set({ enabled: next }, () => setToggleVisual(next));
        chrome.tabs.query({}, tabs => {
          for (const t of tabs) chrome.tabs.sendMessage(t.id, { type: 'settings-updated', enabled: next }).catch(()=>{});
        });
      });
    });
  }

  // settings mode clicks
  document.querySelectorAll('.mode').forEach(btn => {
    btn.addEventListener('click', () => {
      const mode = btn.dataset.mode;
      chrome.storage.sync.set({ mode });
      document.querySelectorAll('.mode').forEach(m=>m.classList.toggle('active', m.dataset.mode === mode));
    });
  });

  // save settings
  const saveSettingsBtn = document.getElementById('saveSettings');
  if (saveSettingsBtn) {
    saveSettingsBtn.addEventListener('click', () => {
      const backendUrl = document.getElementById('uiBackend').value.trim();
      const blockExplicit = document.getElementById('uiBlockExplicit').checked;
      const blurSuggestive = document.getElementById('uiBlurSuggestive').checked;
      const filterText = document.getElementById('uiFilterText').checked;
      chrome.storage.sync.set({ backendUrl, blockExplicit, blurSuggestive, filterText }, () => {
        chrome.runtime.sendMessage({ type: 'backend-updated', backendUrl });
        alert('Saved');
      });
    });
  }

  // open options from popup demo
  const openSettingsBtn = document.getElementById('openSettingsBtn');
  if (openSettingsBtn) openSettingsBtn.addEventListener('click', () => chrome.runtime.openOptionsPage());

  // blocked overlay demo
  const showBlockedDemo = document.getElementById('showBlockedDemo');
  if (showBlockedDemo) showBlockedDemo.addEventListener('click', () => {
    // use same overlay logic as content script
    showBlockedOverlay();
  });

  // helper to update toggle UI
  function setToggleVisual(on){
    const el = document.getElementById('demoToggle');
    if(!el) return;
    el.classList.toggle('on', !!on);
  }

  // overlay function (same style used in content script)
  function showBlockedOverlay() {
    // remove existing first
    const existing = document.querySelector('.filterx-block-overlay');
    if (existing) existing.remove();

    const overlay = document.createElement('div');
    overlay.className = 'filterx-block-overlay';
    overlay.style = `
      position:fixed;inset:0;display:flex;align-items:center;justify-content:center;
      background:linear-gradient(180deg, rgba(10,20,40,0.6), rgba(8,12,24,0.5)); z-index:9999999;
    `;
    overlay.innerHTML = `
      <div style="background:linear-gradient(180deg,#fff,#fbfdff);padding:26px;border-radius:18px;text-align:center;max-width:560px;box-shadow:0 30px 80px rgba(6,14,28,0.5);border:1px solid rgba(30,60,120,0.06)">
        <h2 style="margin:0;color:#1f4a7a">FilterX blocked this content</h2>
        <p style="color:#6d8298;margin-top:10px">This content may be explicit or unsafe based on your current AI protection settings.</p>
        <div style="margin-top:16px"><button id="demoViewAnyway" style="padding:12px 18px;border-radius:26px;background:linear-gradient(90deg,#7c5cff,#47d0ff);color:#fff;border:none;cursor:pointer">View Anyway (Not Recommended)</button></div>
        <div style="margin-top:10px;color:#7b8ca3">Adjust your filters in extension settings if needed</div>
      </div>
    `;
    document.body.appendChild(overlay);
    document.getElementById('demoViewAnyway').addEventListener('click', () => overlay.remove());
  }
});
