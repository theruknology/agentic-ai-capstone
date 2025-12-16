import sys
import os
import redis
import shutil
import json
import time
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from langchain_community.document_loaders import PyPDFLoader
from src.infra.db import get_redis_client
from src.infra.ingest import chunk_documents, save_to_chroma 
from src.infra.notifier import send_alert
from src.ai_engine.graph import HiringOrchestrator

load_dotenv()

r = get_redis_client()

INBOX_DIR = "data/inbox"
PROCESSED_DIR = "data/processed"
JOB_FILE = "data/jobs/current_job.txt"

os.makedirs(INBOX_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

def process_candidate(filename):
    print(f"⚡ EVENT: Picked up {filename} from Queue")
    
    # 1. Update Status
    candidate_key = f"candidate:{filename}"
    r.hset(candidate_key, "status", "processing")
    
    filepath = os.path.join(INBOX_DIR, filename)
    if not os.path.exists(filepath):
        print(f"⚠️ File missing: {filepath}")
        return

    # 2. Ingest
    print(f"   - Ingesting...")
    loader = PyPDFLoader(filepath)
    docs = loader.load()
    chunks = chunk_documents(docs)
    save_to_chroma(chunks)
    
    # 3. Run Pipeline
    if os.path.exists(JOB_FILE):
        orchestrator = HiringOrchestrator()
        orchestrator.run_workflow(JOB_FILE) # This saves the JSON report
        
        # 4. Notify
        # We fetch Name/Email from Redis to look up the JSON report
        metadata = r.hgetall(candidate_key)
        check_and_alert(metadata, filename)

    # 5. Cleanup
    shutil.move(filepath, os.path.join(PROCESSED_DIR, filename))
    r.hset(candidate_key, "status", "completed")
    print(f"✅ Finished processing {filename}")

def check_and_alert(metadata, filename):
    # Retrieve details
    name = metadata.get("name", "Unknown")
    email = metadata.get("email", "No Email")
    
    # Locate the report by sanitized name (same logic as main.py)
    safe_id = "".join([c for c in name if c.isalnum() or c in (' ', '_')]).replace(" ", "_")
    report_path = f"reports/{safe_id}_report.json"
    
    if os.path.exists(report_path):
        with open(report_path) as f:
            data = json.load(f)
            score = data.get("match_score", 0)
            reasoning = data.get("full_details", {}).get("screening", {}).get("reasoning", "No reasoning provided.")
            
            if score >= 70:
                send_alert(name, email, score, reasoning)

# --- Event Loop ---
print("👷 Redis Worker Started. Waiting for jobs...")

while True:
    # BLPOP is a "Blocking Pop". It freezes here until a job arrives.
    # It consumes 0% CPU while waiting. Efficient!
    # Returns tuple: ('resume_queue', 'filename.pdf')
    try:
        queue_name, filename = r.blpop("resume_queue")
        process_candidate(filename)
    except Exception as e:
        print(f"❌ Worker Error: {e}")
        time.sleep(1)