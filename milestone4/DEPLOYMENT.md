# 🚀 Mood Mentor — Deployment & Hosting Guide

This guide outlines how to deploy Mood Mentor to live public platforms (Streamlit Community Cloud, Render, Hugging Face Spaces, or Docker) with zero hardcoded dependencies.

---

## 🌐 Option 1: Free 1-Click Live Hosting on Streamlit Community Cloud (Recommended)
This gives you a free public URL (e.g., `https://moodmentor-wellness.streamlit.app`) to share with your professor, mentors, or recruiters:

1. **Push Milestone 4 to GitHub:**
   Ensure your GitHub repo (`https://github.com/Riteshtiwarii/mood-mentor`) has the `milestone4` folder.
2. **Log in to [share.streamlit.io](https://share.streamlit.io/):**
   Sign in with your GitHub account.
3. **Click "New app":**
   * **Repository:** `Riteshtiwarii/mood-mentor`
   * **Branch:** `main`
   * **Main file path:** `milestone4/dashboard/app.py`
4. **Click "Deploy!":**
   Streamlit will automatically read `requirements.txt` and `.streamlit/config.toml`. Within 60 seconds, your enterprise dashboard will be live on the internet!

---

## 🌐 Option 2: Cloud Deployment on Render.com (Web Service)
1. Link your GitHub repository on [Render.com](https://render.com/).
2. Select **Web Service** (Python 3).
3. Set **Build Command:**
   ```bash
   pip install -r requirements.txt && pip install -e .
   ```
4. Set **Start Command:**
   ```bash
   streamlit run dashboard/app.py --server.port $PORT --server.address 0.0.0.0
   ```

---

## 🐳 Option 3: Local or Server Docker Deployment
Run containerized using Docker without needing Python installed locally:

```bash
# 1. Build the image
docker build -t moodmentor:v4 .

# 2. Run container
docker run -d -p 8501:8501 --name moodmentor_app moodmentor:v4

# 3. Access in browser
open http://localhost:8501
```

Or using Docker Compose:
```bash
docker-compose up -d
```

---

## 💻 Option 4: Local Development & CLI Execution
```bash
# Activate your virtual environment
source .venv/bin/activate

# Install package in development mode
pip install -e .

# Run CLI analysis
moodmentor analyze "I feel overwhelmed with tight project deadlines."

# Launch Streamlit web dashboard
streamlit run dashboard/app.py
```
