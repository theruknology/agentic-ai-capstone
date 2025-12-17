import sys
import streamlit as st
import os
import time
from datetime import datetime

# --- PATH SETUP ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.infra.db import get_redis_client

# --- CONFIGURATION ---
UPLOAD_DIR = "data/inbox"
JOB_FILE = "data/jobs/current_job.txt"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Connect to Redis
try:
    r = get_redis_client()
    redis_online = True
except:
    redis_online = False

# --- PAGE SETUP ---
st.set_page_config(page_title="Apply @ J*bLess", page_icon="🚀", layout="centered")

# Custom CSS
st.markdown("""
    <style>
    .main-header {text-align: center; font-size: 3rem; font-weight: 700; margin-bottom: 10px;}
    .sub-header {text-align: center; color: #666; margin-bottom: 30px;}
    .job-card {padding: 20px; border-radius: 10px; border-left: 5px solid #ff4b4b; margin-bottom: 20px;}
    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("<h1 class='main-header'>J*bLess</h1>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>The Autonomous AI Hiring Platform</div>", unsafe_allow_html=True)

# --- DYNAMIC JOB DISPLAY ---
current_role = "General Application"
job_description_preview = "We are always looking for great talent."

if os.path.exists(JOB_FILE):
    with open(JOB_FILE, "r") as f:
        full_text = f.read()
        # Simple logic: First line is title, rest is details
        lines = full_text.split('\n')
        if lines:
            current_role = lines[0].replace("#", "").strip() or "Open Role"
            job_description_preview = "\n".join(lines[1:])[:500] + "..." # Preview first 500 chars

st.markdown(f"""
    <div class='job-card'>
        <h3>📌 Applying for: {current_role}</h3>
        <p style='font-size: 0.9em;'>{job_description_preview}</p>
    </div>
""", unsafe_allow_html=True)

with st.expander("📄 View Full Job Description"):
    if os.path.exists(JOB_FILE):
        st.markdown(full_text)
    else:
        st.warning("No specific job description loaded. Application will be general.")

# --- APPLICATION FORM ---
with st.container(border=True):
    st.subheader("Submit Your Application")
    
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Full Name", placeholder="Jane Doe")
    with col2:
        email = st.text_input("Email Address", placeholder="jane@example.com")
        
    uploaded_file = st.file_uploader("Upload CV / Resume (PDF)", type="pdf")

    if st.button("🚀 Send Application", type="primary", use_container_width=True):
        if not redis_online:
            st.error("❌ System Offline (Redis Disconnected). Please try again later.")
        elif name and email and uploaded_file:
            try:
                # 1. Save Metadata to Redis (The "Database")
                candidate_key = f"candidate:{uploaded_file.name}"
                r.hset(candidate_key, mapping={
                    "name": name,
                    "email": email,
                    "status": "queued",
                    "submitted_at": datetime.now().isoformat(),
                })

                # 2. Save PDF to Disk (The "Storage")
                save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # 3. Trigger the Worker (The "Event")
                r.lpush("resume_queue", uploaded_file.name)

                st.success(f"🎉 Success! We received your application for **{current_role}**.")
                st.balloons()
                time.sleep(3)
                st.rerun()
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("⚠️ Please fill in all fields and upload a PDF.")