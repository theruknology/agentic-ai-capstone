import json
import os
import glob
from datetime import datetime

# Configuration
REPORTS_DIR = "reports"
FINAL_OUTPUT_FILE = "final_evaluation_report.json"

def generate_master_report():
    print("📊 Generating Master Report from Agent Logs...")
    
    # Get all JSON files
    files = glob.glob(os.path.join(REPORTS_DIR, "*_report.json"))
    if not files:
        print("⚠️ No report files found in 'reports/'")
        return

    reports = []
    high_match = 0
    total_score = 0
    valid_count = 0
    
    for f_path in files:
        try:
            with open(f_path, "r") as f:
                data = json.load(f)
                
                # --- NEW SCHEMA ADAPTATION ---
                # 1. Grab ID safely
                candidate_id = data.get("evaluation_metadata", {}).get("candidate_id", "Unknown")
                
                # 2. Grab Score (Handle both 0-100 and 0-1 float edge cases)
                # The new agent saves "match_score" (0-100) at root
                raw_score = data.get("match_score", 0)
                
                # 3. Grab Decision
                # Located deep in full_details -> screening -> reasoning
                details = data.get("full_details", {})
                reasoning = details.get("screening", {}).get("reasoning", "No reasoning provided.")
                
                # Add to list
                reports.append({
                    "id": candidate_id,
                    "score": raw_score,
                    "summary": reasoning[:150] + "...", # Truncate for readability
                    "filename": os.path.basename(f_path)
                })

                # Stats
                total_score += raw_score
                if raw_score >= 80: 
                    high_match += 1
                valid_count += 1
                
        except Exception as e:
            print(f"❌ Error reading {f_path}: {e}")

    # Calculate Averages
    avg_score = round(total_score / valid_count, 2) if valid_count > 0 else 0

    # Construct Master JSON
    master_report = {
        "meta": {
            "generated_at": datetime.now().isoformat(),
            "model_version": "J*bLess v2.0 (Llama-3-70B)",
            "total_processed": valid_count
        },
        "statistics": {
            "high_match_count": high_match,
            "average_match_score": avg_score,
            "success_rate": f"{(high_match/valid_count*100):.1f}%" if valid_count else "0%"
        },
        "rankings": sorted(reports, key=lambda x: x['score'], reverse=True) # Sort Best -> Worst
    }

    # Save to Disk
    with open(FINAL_OUTPUT_FILE, "w") as f:
        json.dump(master_report, f, indent=2)
        
    print(f"\n✅ Master Report Saved: {FINAL_OUTPUT_FILE}")
    print("------------------------------------------------")
    print(f"   Candidates Processed: {valid_count}")
    print(f"   High Matches (>80%):  {high_match}")
    print(f"   Average Score:        {avg_score}")
    print("------------------------------------------------")

if __name__ == "__main__":
    generate_master_report()