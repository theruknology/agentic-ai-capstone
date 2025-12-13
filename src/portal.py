import streamlit as st
import os
import time
import json
from datetime import datetime

# Configuration
UPLOAD_DIR = "data/inbox"
METADATA_FILE = "data/candidates.json"  # The "Database"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(page_title="Careers @ TechCorp", page_icon="🚀", layout="centered")

# Custom CSS for "Public" feel
st.markdown("""
    <style>
    .main-header {text-align: left; padding: 20px;}
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
            # 1. Save PDF to Inbox
            save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # 2. Update Metadata "Database"
            metadata = {}
            if os.path.exists(METADATA_FILE):
                try:
                    with open(METADATA_FILE, "r") as f:
                        metadata = json.load(f)
                except json.JSONDecodeError:
                    metadata = {}
            
            # Map Filename -> Candidate Details
            metadata[uploaded_file.name] = {
                "name": name,
                "email": email,
                "submitted_at": datetime.now().isoformat()
            }
            
            with open(METADATA_FILE, "w") as f:
                json.dump(metadata, f, indent=2)

            st.success(f"🎉 Thanks, {name}! Application Received.")
            time.sleep(2)
        else:
            st.error("Please fill in all fields.")