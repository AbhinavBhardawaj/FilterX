const API_BASE_URL = "http://127.0.0.1:8000";

document.addEventListener("DOMContentLoaded", async () => {
  updateUI();
  
  const toggleBtn = document.getElementById("toggle-btn");
  const dashboardBtn = document.getElementById("dashboard-btn");
  const settingsBtn = document.getElementById("settings-btn");
  
  if (toggleBtn) {
    toggleBtn.addEventListener("change", async (e) => {
      const enabled = e.target.checked;
      await chrome.storage.local.set({ filterxEnabled: enabled });
      updateUI();
    });
  }
  
  if (dashboardBtn) {
    dashboardBtn.addEventListener("click", () => {
      // Open the dashboard in a new tab
      chrome.tabs.create({ url: chrome.runtime.getURL('dashboard.html') });
    });
  }

  if (settingsBtn) {
    settingsBtn.addEventListener("click", () => {
      // Open settings in a new tab
      chrome.runtime.openOptionsPage();
    });
  }

  const select = document.getElementById('aggressiveness-select');
  if (select) {
    select.addEventListener('change', async (e) => {
      const val = e.target.value;
      await chrome.storage.local.set({ aggressiveness: val });
      select.disabled = true;
      setTimeout(() => select.disabled = false, 300);
    });
  }
});

async function updateUI() {
  const items = await chrome.storage.local.get(["filterxEnabled", "stats", "aggressiveness"]);
  const filterxEnabled = (typeof items.filterxEnabled === 'undefined') ? true : items.filterxEnabled;
  const stats = items.stats || {};
  const aggressiveness = items.aggressiveness || 'normal';
  
  // Update toggle switch
  const toggleBtn = document.getElementById("toggle-btn");
  if (toggleBtn) {
    toggleBtn.checked = filterxEnabled;
  }
  
  // Update stats
  document.getElementById("blocked-count").textContent = stats.blocked || 0;
  document.getElementById("blurred-count").textContent = stats.blurred || 0;
  
  // Reflect aggressiveness in dropdown
  const select = document.getElementById('aggressiveness-select');
  if (select) {
    select.value = aggressiveness;
  }
}

