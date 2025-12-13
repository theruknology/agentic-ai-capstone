import time
import schedule
import os
import shutil
import glob
import json
from langchain_community.document_loaders import PyPDFLoader
from ingest import chunk_documents, save_to_chroma 
from main import HiringOrchestrator
from notifier import send_alert

# Configuration
INBOX_DIR = "data/inbox"
PROCESSED_DIR = "data/processed"
JOB_FILE = "data/jobs/current_job.txt"

os.makedirs(INBOX_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

def process_new_candidates():
    files = [f for f in os.listdir(INBOX_DIR) if f.endswith(".pdf")]
    
    if not files:
        print(f"💤 No new candidates in {INBOX_DIR}...")
        return

    print(f"⚡ Found {len(files)} new applications! Waking up AI Agents...")

    # 1. Ingest
    all_chunks = []
    for f in files:
        src = os.path.join(INBOX_DIR, f)
        print(f"   - Ingesting {f}...")
        loader = PyPDFLoader(src)
        docs = loader.load()
        chunks = chunk_documents(docs)
        all_chunks.extend(chunks)

    if all_chunks:
        save_to_chroma(all_chunks)

    # 2. Run Pipeline
    if os.path.exists(JOB_FILE):
        orchestrator = HiringOrchestrator()
        orchestrator.run_workflow(JOB_FILE)

        # 3. Check & Alert
        check_and_alert_latest_reports()

    # 4. Cleanup
    for f in files:
        shutil.move(os.path.join(INBOX_DIR, f), os.path.join(PROCESSED_DIR, f))
    print("✅ Batch complete. Going back to sleep.")

def check_and_alert_latest_reports():
    # Check reports created/modified in the last 2 minutes
    reports = glob.glob("reports/*_report.json")
    for r in reports:
        if os.path.getmtime(r) > time.time() - 120:
            with open(r) as f:
                data = json.load(f)
                score = data["evaluation"]["screening"].get("fit_score", 0)
                
                if score >= 0.8:
                    send_alert(
                        data["meta"]["id"],          # Name
                        data["meta"].get("email", "N/A"), # Email
                        score * 100, 
                        data["evaluation"]["screening"].get("reasoning", "")
                    )

# --- Run Loop ---
print("🕒 Scheduler Started. Checking for candidates every 1 minute...")
process_new_candidates() # Run once immediately on startup
schedule.every(1).minutes.do(process_new_candidates)

while True:
    schedule.run_pending()
    time.sleep(10)