"""
whatsapp_bot_service.py
=======================
WhatsApp Chatbot Integration Service for PediaMorph AI Frontline Malnutrition Screening.

Provides:
1. Meta WhatsApp Cloud API Client (send text, send image media with badges)
2. Twilio WhatsApp API Client (alternative gateway)
3. Local Interactive Simulator (test WhatsApp conversations in CLI without external credentials)
4. Integration bridge directly invoking the FastAPI backend /api/v1/diagnose endpoint
"""

import os
import sys
import json
import base64
import requests
from pathlib import Path
from typing import Optional, Dict

CURRENT_DIR = Path(__file__).resolve().parent

# Meta Cloud API Configuration (set in environment or .env)
WHATSAPP_TOKEN = os.environ.get("WHATSAPP_TOKEN", "")
WHATSAPP_PHONE_NUMBER_ID = os.environ.get("WHATSAPP_PHONE_NUMBER_ID", "")
WHATSAPP_API_URL = f"https://graph.facebook.com/v19.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"

# Twilio Configuration (alternative)
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER", "whatsapp:+14155238886")


class WhatsAppBotService:
    def __init__(self, api_base_url: str = "http://127.0.0.1:8000"):
        self.api_base_url = api_base_url.rstrip("/")

    def diagnose_image_file(self, image_path: str) -> Dict:
        """Calls PediaMorph local API with an image path or bytes."""
        url = f"{self.api_base_url}/api/v1/diagnose-path"
        resp = requests.post(url, json={"image_path": str(image_path), "include_visuals": True})
        if resp.status_code != 200:
            raise RuntimeError(f"API Error ({resp.status_code}): {resp.text}")
        return resp.json()

    def diagnose_image_bytes(self, image_bytes: bytes, filename: str = "photo.jpg") -> Dict:
        """Calls PediaMorph local API with multipart image upload."""
        url = f"{self.api_base_url}/api/v1/diagnose"
        files = {"file": (filename, image_bytes, "image/jpeg")}
        resp = requests.post(url, files=files)
        if resp.status_code != 200:
            raise RuntimeError(f"API Error ({resp.status_code}): {resp.text}")
        return resp.json()

    def resolve_multimodal(self, residuals_36: list, gender: str, tape_muac_cm: float) -> Dict:
        """Calls secondary escalation endpoint for borderline Yellow cases."""
        url = f"{self.api_base_url}/api/v1/multimodal-resolve"
        payload = {
            "residuals_36": residuals_36,
            "gender": gender,
            "tape_muac_cm": float(tape_muac_cm)
        }
        resp = requests.post(url, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"Multimodal Error ({resp.status_code}): {resp.text}")
        return resp.json()

    def format_reply_text(self, res: dict) -> str:
        """Formats the full frontline medical triage message for WhatsApp."""
        triage = res.get("triage_summary", {})
        bio = res.get("biometrics", {})
        tier = triage.get("tier", "GREEN")
        badge = triage.get("badge", "")
        action = triage.get("clinical_action", "")
        p_ens = triage.get("malnutrition_probability", 0.0) * 100
        p_lda = triage.get("probability_lda", 0.0) * 100
        p_xgb = triage.get("probability_xgboost", 0.0) * 100
        egat_p_raw = triage.get("backup_egat_probability")
        egat_str = f"{egat_p_raw * 100:.1f}%" if egat_p_raw is not None else "Disabled (Offline Edge Mode)"
        wasting_site = triage.get("primary_wasting_locus", "Mid-Upper Arm")
        muac_cm = bio.get("vision_estimated_muac_cm", "N/A")
        arm_cm = bio.get("total_arm_length_cm", "N/A")
        audit_id = res.get("audit_log", {}).get("audit_id", "QA-PENDING")

        msg = (
            f"📋 *PEDIAMORPH AI - CHILD NUTRITION TRIAGE*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🆔 *Session:* `{audit_id}`\n"
            f"🎯 *Triage Status:*\n{badge}\n\n"
            f"📊 *Primary Ensemble Risk:* *{p_ens:.1f}%*\n"
            f"   • Shrinkage LDA (Sensitivity): {p_lda:.1f}%\n"
            f"   • XGBoost Shape (Specificity): {p_xgb:.1f}%\n\n"
            f"🔬 *Backup Cloud EGAT Cross-Check:* {egat_str}\n"
            f"📍 *Primary Wasting Indicator:* {wasting_site}\n\n"
            f"📏 *Estimated Dimensions:*\n"
            f"• Vision Estimated MUAC: ~{muac_cm} cm\n"
            f"• Total Arm Length: {arm_cm} cm\n\n"
            f"🩺 *Clinical Action Guide for ASHA / Health Worker:*\n"
            f"{action}\n"
        )

        if tier == "YELLOW":
            msg += (
                f"\n💡 *Borderline Action:*\n"
                f"If you have a MUAC tape, reply with:\n"
                f"`MUAC <value_in_cm>` (e.g., `MUAC 12.5`)\n"
                f"for immediate >89.7% multimodal resolution."
            )

        msg += f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n_PediaMorph AI Frontline Pediatric Health Assistant_"
        return msg

    def send_meta_whatsapp_message(self, recipient_phone: str, message_text: str, media_url: Optional[str] = None):
        """Sends WhatsApp message via Meta WhatsApp Business Cloud API."""
        if not WHATSAPP_TOKEN or not WHATSAPP_PHONE_NUMBER_ID:
            print("[INFO] Meta WhatsApp credentials not set. Message rendered locally:")
            print(message_text)
            return {"status": "mock_sent", "text": message_text}

        headers = {
            "Authorization": f"Bearer {WHATSAPP_TOKEN}",
            "Content-Type": "application/json"
        }

        # 1. Send Text Message
        payload = {
            "messaging_product": "whatsapp",
            "to": recipient_phone,
            "type": "text",
            "text": {"body": message_text}
        }
        resp = requests.post(WHATSAPP_API_URL, headers=headers, json=payload)

        # 2. Optionally send annotated image
        if media_url:
            media_payload = {
                "messaging_product": "whatsapp",
                "to": recipient_phone,
                "type": "image",
                "image": {"link": media_url}
            }
            requests.post(WHATSAPP_API_URL, headers=headers, json=media_payload)

        return resp.json()


# ---------------------------------------------------------------------------
# CLI Interactive WhatsApp Bot Simulator
# ---------------------------------------------------------------------------
def run_cli_simulator():
    """Simulates WhatsApp conversation between Frontline Worker and PediaMorph AI Bot."""
    if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    print("=" * 75)
    print("  PEDIAMORPH AI - WHATSAPP CHATBOT FRONTEND SIMULATOR")
    print("=" * 75)
    print("  Simulating WhatsApp interface for ASHA / Anganwadi Frontline Worker.")
    print("  Zero calibration board required. Snaps photo of child's left arm.")
    print("=" * 75)

    bot = WhatsAppBotService()
    sample_dir = CURRENT_DIR / "sample_images"
    sample_img = sample_dir / "03f74f_image1_without_ruler.jpg"

    if not sample_img.exists():
        print(f"[ERROR] Sample image not found at {sample_img}")
        return

    print(f"\n[ASHA WORKER]: (Sends child left-arm photo: {sample_img.name})")
    print("[BOT ENGINE]: Processing natural photo through MediaPipe + LDA-XGBoost Ensemble...")

    try:
        diagnosis = bot.diagnose_image_file(str(sample_img))
        reply = bot.format_reply_text(diagnosis)
        print("\n[BOT TO WORKER VIA WHATSAPP]:\n")
        print(reply)

        if diagnosis.get("triage_summary", {}).get("is_yellow_borderline"):
            print("\n[ASHA WORKER]: MUAC 12.5")
            print("[BOT ENGINE]: Running Multimodal Secondary Escalation...")
            res36 = diagnosis.get("residuals_36", [])
            multi_res = bot.resolve_multimodal(res36, gender="M", tape_muac_cm=12.5)
            print(f"\n[BOT TO WORKER VIA WHATSAPP]:\n"
                  f"✅ *Multimodal Confirmation Complete!*\n"
                  f"Tape MUAC: 12.5 cm | Probability: {multi_res.get('multimodal_probability')*100:.1f}%\n"
                  f"Definitive Triage: *{multi_res.get('multimodal_triage')}*\n"
                  f"Clinical Confidence: {multi_res.get('clinical_confidence')}")

    except Exception as e:
        print(f"[ERROR]: {e}")
        print("[HINT]: Ensure the FastAPI backend is running via 'python run_demo.py' on http://127.0.0.1:8000")


if __name__ == "__main__":
    run_cli_simulator()
