import json
import os
import glob
from datetime import datetime

REPORTS_DIR = "reports"
FINAL_OUTPUT_FILE = "final_evaluation_report.json"

def generate_master_report():
    print("📊 Generating Master Report...")
    files = glob.glob(os.path.join(REPORTS_DIR, "*_report.json"))
    reports = []
    
    high_match = 0
    total_score = 0
    
    for f_path in files:
        with open(f_path, "r") as f:
            data = json.load(f)
            reports.append(data)
            score = data["evaluation"]["screening"].get("fit_score", 0) * 100
            total_score += score
            if score >= 80: high_match += 1

    summary = {
        "total_candidates": len(reports),
        "high_match_count": high_match,
        "average_score": round(total_score / len(reports), 2) if reports else 0
    }

    final_report = {
        "meta": {"generated_at": datetime.now().isoformat(), "model": "Llama-3.3-Groq"},
        "summary": summary,
        "details": reports
    }

    with open(FINAL_OUTPUT_FILE, "w") as f:
        json.dump(final_report, f, indent=2)
        
    print(f"✅ Final Report Saved: {FINAL_OUTPUT_FILE}")
    print(f"   Stats: {summary}")

if __name__ == "__main__":
    generate_master_report()