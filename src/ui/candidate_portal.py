import sys
import streamlit as st
import os
import time
from datetime import datetime

# Fix path to import infra
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.infra.db import get_redis_client

# Connect to Redis
r = get_redis_client()

# Configuration
UPLOAD_DIR = "data/inbox"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(page_title="Careers @ TechCorp", page_icon="🚀", layout="centered")

st.markdown("""
    <style>
    .stApp {}
    .main-header {text-align: center; padding: 20px;}
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>🚀 Join Our Team</h1>", unsafe_allow_html=True)
st.write("We are looking for top talent in AI, Bioinformatics, and Engineering.")

with st.container(border=True):
    st.subheader("📌 Open Role: Senior Bioinformatics Scientist")
    st.write("**Location:** Boston, MA (Hybrid)")
    st.write("**Role:** Lead NGS data analysis using Python and R.")
    
    st.divider()
    
    # Application Form
    name = st.text_input("Full Name")
    email = st.text_input("Email Address")
    uploaded_file = st.file_uploader("Upload your CV (PDF)", type="pdf")

    if st.button("Submit Application", type="primary", use_container_width=True):
        if name and email and uploaded_file:
            # --- STEP 1: Save Metadata to Redis ---
            # We do this FIRST so the "database" record exists.
            candidate_key = f"candidate:{uploaded_file.name}"
            r.hset(candidate_key, mapping={
                "name": name,
                "email": email,
                "status": "queued",
                "submitted_at": datetime.now().isoformat(),
            })

            # --- STEP 2: Save PDF to Disk ---
            save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # --- STEP 3: Trigger the Worker ---
            # Only push to the queue after the file is physically on disk
            r.lpush("resume_queue", uploaded_file.name)

            st.success(f"🎉 Thanks, {name}! Application Received.")
            time.sleep(2)
        else:
            st.error("Please fill in all fields.")