import os
import requests
import datetime
from dotenv import load_dotenv

load_dotenv()
DISCORD_URL = os.getenv("DISCORD_WEBHOOK_URL")

def send_alert(candidate_name, email, score, reason):
    """Sends a rich Embed to Discord."""
    if not DISCORD_URL: 
        print("⚠️ DISCORD_WEBHOOK_URL missing.")
        return

    print(f"📨 Sending Alert for {candidate_name}...")
    
    # Color logic: Green for High Match, Yellow for Medium
    color = 0x00ff00 if score >= 80 else 0xffff00

    payload = {
        "username": "J*bLess AI 🤖",
        "embeds": [{
            "title": f"🚀 Top Candidate: {candidate_name}",
            "description": f"**Match Score:** {score:.1f}%\n\n_{reason}_",
            "color": color,
            "fields": [
                {"name": "📧 Email", "value": email, "inline": True},
                {"name": "⚡ Status", "value": "Recommended for Interview", "inline": True}
            ],
            "footer": {"text": f"Processed at {datetime.datetime.now().strftime('%H:%M')}"}
        }]
    }

    try:
        requests.post(DISCORD_URL, json=payload)
        print("✅ Alert Sent!")
    except Exception as e:
        print(f"❌ Alert Failed: {e}")

if __name__ == "__main__":
    send_alert("Test User", "test@example.com", 95.0, "Perfect skills match.")