# Sankatmochan 🚫🔞  
**An AI-Powered Adult Content Blocker for a Safer Digital Experience**  

## 📌 Overview  
Sankatmochan is a smart content filtering solution designed to block adult and inappropriate content across devices and browsers.  
Unlike traditional blockers that rely only on keyword filtering, **Sankatmochan integrates AI models** (text + image classification) to provide **real-time and context-aware content moderation**.  

The goal is to create a **secure, family-friendly browsing environment** without compromising performance or usability.  

---

## ✨ Key Features  
- 🔍 **AI-Powered Text Filtering** – Detects inappropriate language, context, and disguised adult terms.  
- 🖼️ **AI-Powered Image Filtering** – Uses image classification models (e.g., NSFW detection) to block adult images.  
- 🌐 **Cross-Platform Support** – Works as a **browser extension** and can also integrate at the **system level**.  
- ⚡ **Real-Time Blocking** – Instant detection & blocking without noticeable lag.  
- 🛡️ **Bypass Protection** – Prevents disabling the blocker without authentication.  
- 📊 **Usage Dashboard** – View logs and insights of blocked attempts.  
- 👨‍👩‍👧‍👦 **Parental Control Mode** – Admins/parents can customize filtering levels.  
- 🆕 **Unique Feature: Context-Aware Filtering**  
  - Unlike most blockers that block just by keywords, Sankatmochan uses AI to understand **context**.  
  - Example: "breast cancer awareness" will **not** be blocked, but "adult sites" will.  

---

## 🛠️ Tech Stack  
- **Frontend**: JavaScript (Browser Extension), HTML, CSS  
- **Backend (Optional for Dashboard/Logs)**: Django / Flask  
- **AI Models**:  
  - Text classification (HuggingFace transformers / spaCy)  
  - Image classification (MobileNet, OpenAI CLIP, or custom CNN)  
- **Database**: SQLite / PostgreSQL (for logs & parental controls)  

---

## 🚀 Installation  

### 1. Clone the Repository  
```bash
git clone https://github.com/AbhinavBhardawaj/Sankat-Mochan.git
cd sankatmochan
