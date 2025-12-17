import sys
import streamlit as st
import os
import shutil
import pandas as pd
import json
import time
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- PATH SETUP ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.ai_engine.graph import HiringOrchestrator
from src.infra.db import get_redis_client

# --- CONFIGURATION ---
DATA_DIR = "data"
RESUME_DIR = os.path.join(DATA_DIR, "inbox")
JOB_DIR = os.path.join(DATA_DIR, "jobs")
REPORTS_DIR = "reports"

os.makedirs(RESUME_DIR, exist_ok=True)
os.makedirs(JOB_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Connect to Redis
try:
    redis_client = get_redis_client()
    redis_connected = True
except:
    redis_connected = False

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="J*bLess Admin", 
    page_icon="🔥", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .metric-card {background-color: #f0f2f6; border-radius: 10px; padding: 20px; text-align: center;}
    .stProgress > div > div > div > div {background-color: #FF4B4B;}
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR (Batch Upload & Status) ---
with st.sidebar:
    st.title("🔥 J*bLess")
    st.caption("Recruiter Control Center")
    
    # 1. Quick Batch Upload
    with st.expander("⚡ Quick Batch Upload", expanded=True):
        st.write("Drop multiple PDFs here to auto-queue them.")
        batch_files = st.file_uploader("Upload Resumes", type="pdf", accept_multiple_files=True, label_visibility="collapsed")
        
        if batch_files and st.button("Process Batch"):
            if not redis_connected:
                st.error("Redis Offline!")
            else:
                progress_text = st.empty()
                bar = st.progress(0)
                
                for i, file in enumerate(batch_files):
                    # Save File to Inbox (Worker will ingest it)
                    save_path = os.path.join(RESUME_DIR, file.name)
                    with open(save_path, "wb") as f:
                        f.write(file.getbuffer())
                    
                    # Create Placeholder Metadata
                    candidate_key = f"candidate:{file.name}"
                    redis_client.hset(candidate_key, mapping={
                        "name": os.path.splitext(file.name)[0].replace("_", " ").title(),
                        "email": "N/A (Batch Upload)",
                        "status": "queued",
                        "submitted_at": datetime.now().isoformat()
                    })
                    
                    # Push to Queue
                    redis_client.lpush("resume_queue", file.name)
                    bar.progress((i + 1) / len(batch_files))
                
                st.success(f"Queued {len(batch_files)} candidates!")
                time.sleep(2)
                st.rerun()

    st.divider()
    
    # 2. System Status
    st.subheader("System Status")
    groq_status = "✅ Online" if os.getenv("GROQ_API_KEY") else "❌ Offline"
    redis_status = "✅ Connected" if redis_connected else "❌ Disconnected"
    
    c1, c2 = st.columns(2)
    c1.metric("AI Engine", groq_status)
    c2.metric("Event Bus", redis_status)

    if redis_connected:
        queue_len = redis_client.llen("resume_queue")
        st.metric("Live Queue Depth", f"{queue_len} Jobs")

# --- MAIN HELPERS ---
def load_data():
    reports = []
    files = [f for f in os.listdir(REPORTS_DIR) if f.endswith("_report.json")]
    for f in files:
        try:
            with open(os.path.join(REPORTS_DIR, f)) as file:
                data = json.load(file)
                meta = data.get("evaluation_metadata", {})
                details = data.get("full_details", {})
                screening = details.get("screening", {})
                
                reports.append({
                    "Candidate": meta.get("candidate_id", "Unknown"),
                    "Email": meta.get("candidate_email", "N/A"),
                    "Score": data.get("match_score", 0),
                    "Status": "Passed" if data.get("match_score", 0) >= 70 else "Rejected",
                    "Missing Skills": len(screening.get("missing_skills", [])),
                    "File": f,
                    "Timestamp": meta.get("timestamp", "")
                })
        except Exception as e:
            continue
    return pd.DataFrame(reports)

# --- MAIN CONTENT ---
st.title("📊 J*bLess Dashboard")

# Top Level Metrics
df = load_data()
if not df.empty:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Candidates", len(df))
    m2.metric("Avg Match Score", f"{df['Score'].mean():.1f}%")
    m3.metric("Top Talent (>80%)", len(df[df['Score'] >= 80]))
    m4.metric("Avg Skills Missing", f"{df['Missing Skills'].mean():.1f}")
else:
    st.info("Waiting for data... Use the Sidebar to upload resumes.")

# Tabs
tab_analysis, tab_config = st.tabs(["📈 Rankings & Analysis", "📝 Job Config"])

# --- TAB 1: ANALYTICS ---
with tab_analysis:
    if df.empty:
        st.warning("No data available yet.")
    else:
        # Charts
        c1, c2 = st.columns([2, 1])
        with c1:
            fig = px.histogram(df, x="Score", nbins=10, 
                               color="Status", 
                               color_discrete_map={"Passed": "#FF4B4B", "Rejected": "#333333"},
                               title="Match Score Distribution")
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            fig2 = px.pie(df, names="Status", hole=0.4, color="Status",
                          color_discrete_map={"Passed": "#FF4B4B", "Rejected": "#D3D3D3"})
            st.plotly_chart(fig2, use_container_width=True)

        st.divider()
        st.subheader("📋 Candidate Leaderboard")
        
        st.dataframe(
            df.sort_values(by="Score", ascending=False),
            column_config={
                "Score": st.column_config.ProgressColumn(
                    "Match Score", format="%d%%", min_value=0, max_value=100,
                ),
            },
            hide_index=True,
            use_container_width=True
        )

        st.divider()
        st.subheader("🔍 Deep Dive")
        
        selected_name = st.selectbox("Select Candidate:", df["Candidate"].unique())
        
        if selected_name:
            row = df[df["Candidate"] == selected_name].iloc[0]
            with open(os.path.join(REPORTS_DIR, row["File"])) as f:
                full_report = json.load(f)
                
            details = full_report.get("full_details", {})
            screening = details.get("screening", {})
            
            col_l, col_r = st.columns([1, 2])
            
            with col_l:
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number", value = row["Score"],
                    title = {'text': "Fit Score"},
                    gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#FF4B4B"}}
                ))
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                with st.expander("🚫 Missing Skills", expanded=True):
                    for skill in screening.get("missing_skills", []):
                        st.write(f"- ❌ {skill}")

            with col_r:
                st.markdown("### 🧠 AI Analysis")
                st.info(screening.get("reasoning", "No analysis."))
                
                st.markdown("### 🛠️ Assessment Plan")
                plan = details.get("plan", {}).get("steps", [])
                for step in plan:
                    st.write(f"1. {step}")

# --- TAB 2: CONFIG ---
with tab_config:
    st.subheader("Job Description Editor")
    current_job = ""
    if os.path.exists(os.path.join(JOB_DIR, "current_job.txt")):
        with open(os.path.join(JOB_DIR, "current_job.txt")) as f:
            current_job = f.read()
            
    new_job = st.text_area("Update Job Description (Candidates see this!)", value=current_job, height=400)
    
    if st.button("💾 Save Changes"):
        with open(os.path.join(JOB_DIR, "current_job.txt"), "w") as f:
            f.write(new_job)
        st.success("Job Updated! Candidates will see the new description immediately.")