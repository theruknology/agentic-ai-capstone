import sys
import os
import time
import shutil
import json
import redis
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

# --- PATH SETUP ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.infra.ingest import chunk_documents, save_to_chroma 
from src.ai_engine.graph import HiringOrchestrator
from src.infra.notifier import send_alert
from src.infra.db import get_redis_client
from src.infra.logger import logger

load_dotenv()

# Setup Redis
r = get_redis_client()

INBOX_DIR = "data/inbox"
PROCESSED_DIR = "data/processed"
JOB_FILE = "data/jobs/current_job.txt"

os.makedirs(INBOX_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

def process_candidate(filename):
    logger.info(f"⚡ EVENT: Picked up {filename} from Queue")
    
    # 1. Update Status
    candidate_key = f"candidate:{filename}"
    r.hset(candidate_key, "status", "processing")
    
    filepath = os.path.join(INBOX_DIR, filename)
    if not os.path.exists(filepath):
        logger.warning(f"⚠️ File missing: {filepath}")
        return

    # 2. Ingest (With Retry Logic for DB Locking)
    logger.info(f"   - Ingesting...")
    try:
        loader = PyPDFLoader(filepath)
        docs = loader.load()
        chunks = chunk_documents(docs)
        
        # RETRY LOOP for Database Locks
        max_retries = 3
        for attempt in range(max_retries):
            try:
                save_to_chroma(chunks)
                break # Success
            except Exception as e:
                if "locked" in str(e) or "readonly" in str(e):
                    logger.warning(f"⚠️ DB Locked. Retrying in 1s... ({attempt+1}/{max_retries})")
                    time.sleep(1)
                else:
                    raise e # Real error, crash it
    except Exception as e:
        logger.error(f"❌ Ingestion Failed: {e}")
        # Move to processed anyway to unblock queue
        shutil.move(filepath, os.path.join(PROCESSED_DIR, filename))
        return

    # 3. Run Pipeline
    if os.path.exists(JOB_FILE):
        orchestrator = HiringOrchestrator()
        orchestrator.run_workflow(JOB_FILE) 
        
        # 4. Notify
        metadata = r.hgetall(candidate_key)
        check_and_alert(metadata)

    # 5. Cleanup
    if os.path.exists(filepath):
        shutil.move(filepath, os.path.join(PROCESSED_DIR, filename))
    
    r.hset(candidate_key, "status", "completed")
    logger.info(f"✅ Finished processing {filename}")

def check_and_alert(metadata):
    name = metadata.get("name", "Unknown")
    email = metadata.get("email", "No Email")
    
    # Locate report
    safe_id = "".join([c for c in name if c.isalnum() or c in (' ', '_')]).replace(" ", "_")
    report_path = f"reports/{safe_id}_report.json"
    
    if os.path.exists(report_path):
        try:
            with open(report_path) as f:
                data = json.load(f)
                # NEW SCHEMA ACCESS
                score = data.get("match_score", 0)
                reasoning = data.get("full_details", {}).get("screening", {}).get("reasoning", "")
                
                if score >= 80:
                    send_alert(name, email, score, reasoning)
        except Exception as e:
            logger.error(f"Alert Error: {e}")

# --- Worker Loop ---
logger.info("👷 Redis Worker Started. Waiting for jobs...")

while True:
    try:
        # Blocking Pop - Efficient Wait
        item = r.blpop("resume_queue", timeout=5)
        if item:
            queue_name, filename = item
            process_candidate(filename)
    except Exception as e:
        logger.error(f"❌ Worker Error: {e}")
        time.sleep(1)