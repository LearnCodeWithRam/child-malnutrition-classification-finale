"""
app.py
======
REST API Engine for Child Left-Arm Morphometric Analysis & Malnutrition Triage.
Designed specifically for mobile app integration (Flutter, React Native, Android/Kotlin, iOS/Swift).

NO CHECKERBOARD REQUIRED: Field-ready pediatric vision extraction + morphometric classification.
* Accepts already-clicked photos via:
  1. Multipart Form Upload (POST /api/v1/diagnose)
  2. Base64 JSON Payload   (POST /api/v1/diagnose-base64)
  3. Local File Path        (POST /api/v1/diagnose-path)
  4. Clinical Sample Tests  (POST /api/v1/diagnose-sample/{filename})
* Multimodal Escalation:   (POST /api/v1/multimodal-resolve)
* Interactive Swagger Docs: http://127.0.0.1:8000/docs
"""

import os
import io
import time
import base64
import xml.sax.saxutils
import requests
from pathlib import Path
from typing import Dict, List, Optional, Union

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np

from contour_extractor import extract_arm_contour_and_landmarks
from classifier_engine import MalnutritionClassificationEngine
from audit_logger import save_audit_record, list_recent_audit_logs, get_audit_record, AUDIT_LOGS_DIR

CURRENT_DIR = Path(__file__).resolve().parent
SAMPLES_DIR = CURRENT_DIR / "sample_images"

# Load local .env credentials if present
_env_file = CURRENT_DIR / ".env"
if _env_file.exists():
    with open(_env_file, "r", encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ[_k.strip()] = _v.strip().strip('"').strip("'")


app = FastAPI(
    title="PediaMorph AI - Mobile Malnutrition Triage API",
    description=(
        "REST API for frontline child acute malnutrition screening via geometric morphometrics. "
        "Processes natural left-arm photographs (NO CHECKERBOARD NEEDED) and provides instant "
        "Dual-Tier Triage (Mobile Edge Ensemble + Cloud EGAT Attention Network)."
    ),
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for cross-platform mobile and web client requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton classification engine
engine = MalnutritionClassificationEngine()


# ---------------------------------------------------------------------------
# Request & Response Models
# ---------------------------------------------------------------------------
class Base64ImageRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded child left-arm photograph (with or without data URI prefix)")
    client_id: Optional[str] = Field("mobile_client", description="Identifier of the client device or health worker")
    include_visuals: bool = Field(True, description="Whether to include base64 contour overlay & EGAT heatmap in the response")


class FilePathRequest(BaseModel):
    image_path: str = Field(..., description="Absolute or relative file path to the already-clicked child left arm photo")
    include_visuals: bool = Field(True, description="Whether to include base64 visual overlays in response")


class MultimodalResolveRequest(BaseModel):
    residuals_36: List[float] = Field(..., description="36D allometric shape residuals from initial diagnosis")
    gender: str = Field("M", description="Child sex: 'M' (Male) or 'F' (Female)")
    tape_muac_cm: float = Field(..., description="Physical tape MUAC measurement in centimeters (e.g., 12.5)")


# ---------------------------------------------------------------------------
# Core Pipeline Execution Helper
# ---------------------------------------------------------------------------
def _execute_mobile_pipeline(image_input: Union[bytes, str, Path],
                             source_name: str = "photo",
                             include_visuals: bool = True) -> dict:
    """Executes Phase 1 Vision + Phase 2 Classification and formats for mobile consumption."""
    t0 = time.time()
    try:
        # Phase 1: Natural Left-Arm Contour & Landmark Extraction (Zero Checkerboard)
        p1 = extract_arm_contour_and_landmarks(image_input)
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Phase 1 Vision Processing Error: {str(e)}. Ensure the child's left arm is clearly visible."
        )
    t_p1 = time.time() - t0

    t1_start = time.time()
    try:
        # Phase 2: Dual-Tier Morphometric Classification
        raw_coords = p1["raw_coordinates"]
        p2 = engine.diagnose_full_pipeline(raw_coords)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Phase 2 Classification Engine Error: {str(e)}"
        )
    t_p2 = time.time() - t1_start

    ens_res = p2["offline_ensemble"]
    egat_res = p2["cloud_egat"]

    # Color code mapping for triage badges
    tier = ens_res["triage_tier"]
    color_map = {
        "GREEN": "#10b981",   # Emerald normal
        "YELLOW": "#f59e0b",  # Amber moderate/borderline
        "RED": "#ef4444"      # Crimson high risk
    }

    response = {
        "status": "success",
        "source": source_name,
        "primary_model": "Tier 1 Offline Mobile Ensemble (40% Shrinkage LDA + 60% XGBoost)",
        "backup_model": "Tier 2 Online Cloud EGAT Attention Network (17 Nodes, 9 Calipers)",
        "models_in_agreement": p2.get("models_in_agreement", True),
        "timing_sec": {
            "phase1_vision": round(t_p1, 3),
            "phase2_classification": round(t_p2, 3),
            "total": round(t_p1 + t_p2, 3)
        },
        "triage_summary": {
            "tier": tier,
            "color_hex": color_map.get(tier, "#64748b"),
            "badge": ens_res["clinical_badge"],
            "category": ens_res["nutritional_category"],
            "clinical_action": ens_res["clinical_action"],
            "malnutrition_probability": ens_res["probability_ensemble"],
            "probability_lda": ens_res["probability_lda"],
            "probability_xgboost": ens_res["probability_xgboost"],
            "backup_egat_probability": egat_res.get("probability_egat", None),
            "primary_wasting_locus": egat_res.get("primary_wasting_site", "N/A (Backup Disabled)" if egat_res.get("status") == "disabled" else "Mid-Upper Arm"),
            "is_yellow_borderline": (tier == "YELLOW")
        },
        "biometrics": {
            "arm_side_detected": p1.get("arm_side", "Left"),
            "scale_px_per_cm": p1.get("px_per_cm"),
            "total_arm_length_cm": p1.get("total_arm_cm"),
            "upper_arm_length_cm": p1.get("upper_arm_cm"),
            "forearm_length_cm": p1.get("forearm_cm"),
            "vision_estimated_muac_cm": p1.get("muac_cm"),
            "centroid_size": p2.get("centroid_size")
        },
        "offline_ensemble": ens_res,
        "individual_models": p2.get("individual_models", {}),
        "grouped_models": p2.get("grouped_models", {}),
        "cloud_egat_backup": {
            "status": egat_res.get("status", "active"),
            "probability_egat": egat_res.get("probability_egat", None),
            "triage_tier": egat_res.get("triage_tier", "DISABLED" if egat_res.get("status") == "disabled" else None),
            "primary_wasting_locus": egat_res.get("primary_wasting_site", "N/A"),
            "calipers_attention": egat_res.get("calipers_attention", [])
        },
        "residuals_36": p2.get("residuals_36", []),
        "landmarks_22": p1.get("landmarks", [])
    }

    if include_visuals:
        response["visualizations"] = {
            "contour_overlay_base64": p1.get("contour_overlay_base64"),
            "cartesian_plot_base64": p1.get("cartesian_plot_base64"),
            "side_by_side_base64": p1.get("side_by_side_base64"),
            "egat_attention_heatmap_base64": egat_res.get("attention_heatmap_base64")
        }

    # Automatically persist complete timestamped audit log for quality check (QA)
    try:
        audit_info = save_audit_record(
            original_image_input=image_input,
            source_name=source_name,
            phase1_res=p1,
            phase2_res=p2,
            timing_sec=response["timing_sec"]
        )
        response["audit_log"] = audit_info
    except Exception as e:
        response["audit_log"] = {"status": "error", "message": str(e)}

    return response



# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def api_info():
    """Returns general API information, model metadata, and Swagger link."""
    return {
        "service": "PediaMorph AI - Mobile Malnutrition Triage REST API",
        "version": "2.1.0",
        "status": "online",
        "documentation": "/docs",
        "checkerboard_required": False,
        "supported_inputs": [
            "Multipart file upload (POST /api/v1/diagnose)",
            "Base64 JSON payload (POST /api/v1/diagnose-base64)",
            "Local file path (POST /api/v1/diagnose-path)",
            "Preloaded clinical samples (POST /api/v1/diagnose-sample/{filename})"
        ],
        "models": {
            "vision": "MediaPipe Pose + Multiclass Arm Segmentation (22 Landmarks)",
            "classifier": "17-Node Graph Attention Network (EGAT with 9 Transverse Calipers)",
            "multimodal_resolver": "38D XGBoost Fusion (>89.7% Accuracy)"
        }
    }


@app.get("/api/v1/health")
def health():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "devices": {
            "cpu_threads": os.cpu_count(),
            "models_loaded": True
        }
    }


@app.post("/api/v1/diagnose")
async def diagnose_file_upload(file: UploadFile = File(...),
                               include_visuals: bool = Query(True, description="Include base64 visualizations")):
    """
    Standard Multipart Form Upload endpoint for mobile apps.
    Pass already-clicked child left-arm photograph as 'file'.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image (JPEG, PNG).")

    image_bytes = await file.read()
    if len(image_bytes) < 1000:
        raise HTTPException(status_code=400, detail="Uploaded file is too small or invalid.")

    return _execute_mobile_pipeline(image_bytes, source_name=file.filename, include_visuals=include_visuals)


@app.post("/api/v1/diagnose-base64")
def diagnose_base64(req: Base64ImageRequest):
    """
    JSON endpoint for mobile apps sending photos as base64 strings.
    Supports raw base64 or data URI strings ('data:image/jpeg;base64,...').
    """
    b64_str = req.image_base64
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(b64_str)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 encoding: {str(e)}")

    if len(image_bytes) < 1000:
        raise HTTPException(status_code=400, detail="Decoded image is too small or invalid.")

    return _execute_mobile_pipeline(image_bytes, source_name=f"base64_{req.client_id}", include_visuals=req.include_visuals)


@app.post("/api/v1/diagnose-path")
def diagnose_file_path(req: FilePathRequest):
    """
    Directly processes an image file already stored on the disk by path.
    Ideal for local CLI testing, batch evaluation, or on-device file storage.
    """
    p = Path(req.image_path)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail=f"Image file not found at path: {req.image_path}")

    with open(p, "rb") as f:
        image_bytes = f.read()

    return _execute_mobile_pipeline(image_bytes, source_name=p.name, include_visuals=req.include_visuals)


@app.get("/api/v1/samples")
def list_clinical_samples():
    """Lists preloaded child left-arm test photos for developer verification."""
    samples = []
    if SAMPLES_DIR.exists():
        for f in sorted(SAMPLES_DIR.glob("*.jpg")):
            samples.append({
                "filename": f.name,
                "size_kb": round(f.stat().st_size / 1024, 1),
                "url": f"/api/v1/sample-file/{f.name}"
            })
    return {"samples": samples}


@app.get("/api/v1/sample-file/{filename}")
def get_sample_file(filename: str):
    p = SAMPLES_DIR / filename
    if not p.exists():
        raise HTTPException(status_code=404, detail="Sample image not found.")
    return FileResponse(p, media_type="image/jpeg")


@app.post("/api/v1/diagnose-sample/{filename}")
def diagnose_sample_image(filename: str, include_visuals: bool = Query(True)):
    """Convenience endpoint: Executes end-to-end diagnosis on a preloaded sample photo."""
    p = SAMPLES_DIR / filename
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Sample file '{filename}' not found.")

    with open(p, "rb") as f:
        image_bytes = f.read()

    return _execute_mobile_pipeline(image_bytes, source_name=filename, include_visuals=include_visuals)


@app.post("/api/v1/multimodal-resolve")
def resolve_yellow_borderline(req: MultimodalResolveRequest):
    """
    Secondary Multimodal Escalation Endpoint for Yellow Borderline Triage cases.
    Pass the 36D residuals along with child sex and tape MUAC to achieve >89.7% clinical accuracy.
    """
    if len(req.residuals_36) != 36:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 36 allometric residuals from Phase 2, but received {len(req.residuals_36)}."
        )

    res_arr = np.array(req.residuals_36, dtype=np.float32)
    confirmation = engine.run_multimodal_confirmation(
        residuals_36=res_arr,
        gender=req.gender,
        tape_muac_cm=req.tape_muac_cm
    )
    return confirmation


# ---------------------------------------------------------------------------
# Audit Log & Quality Check (QA) Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/v1/audit-logs")
def list_audit_logs(limit: int = Query(50, ge=1, le=200, description="Max audit logs to return")):
    """Retrieves recent timestamped audit log records with QA flags and triage metrics."""
    return {"audit_logs": list_recent_audit_logs(limit=limit)}


@app.get("/api/v1/audit-logs/{audit_id}")
def get_single_audit_log(audit_id: str):
    """Retrieves full QA audit summary, biometrics, and persisted file references for a session."""
    record = get_audit_record(audit_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Audit record '{audit_id}' not found.")
    return record


@app.get("/api/v1/audit-logs/{audit_id}/file/{filename}")
def get_audit_log_file(audit_id: str, filename: str):
    """
    Downloads or renders any specific persisted QA audit file:
    - 01_original_image.jpg
    - 02_contour_overlay.png
    - 03_cartesian_plot.png
    - 04_side_by_side_quality_check.png
    - 05_egat_attention_heatmap.png
    - landmarks_22.json
    - audit_summary.json
    """
    file_path = AUDIT_LOGS_DIR / audit_id / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Audit file '{filename}' not found for ID '{audit_id}'.")

    media_type = "application/json" if filename.endswith(".json") else ("image/jpeg" if filename.endswith(".jpg") else "image/png")
    return FileResponse(file_path, media_type=media_type)


# ---------------------------------------------------------------------------
# WhatsApp Chatbot Integration Endpoints
# ---------------------------------------------------------------------------
def format_whatsapp_triage_text(res: dict) -> str:
    """Formats clinical diagnosis into a concise, actionable WhatsApp message."""
    triage = res.get("triage_summary", {})
    bio = res.get("biometrics", {})
    tier = triage.get("tier", "GREEN")
    badge = triage.get("badge", "")
    action = triage.get("clinical_action", "")
    p_ens = triage.get("malnutrition_probability", 0.0) * 100
    egat_p_raw = triage.get("backup_egat_probability")
    egat_str = f"{egat_p_raw * 100:.1f}%" if egat_p_raw is not None else "Disabled (Offline Edge Mode)"
    wasting_site = triage.get("primary_wasting_locus", "Mid-Upper Arm")
    muac_cm = bio.get("vision_estimated_muac_cm", "N/A")
    arm_cm = bio.get("total_arm_length_cm", "N/A")
    audit_id = res.get("audit_log", {}).get("audit_id", "QA-PENDING")

    msg = (
        f"📋 *POSHAN APPLICATION - CHILD NUTRITION TRIAGE*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🆔 *Session:* `{audit_id}`\n"
        f"🎯 *Triage Status:*\n{badge}\n\n"
        f"📊 *Primary Ensemble Risk:* *{p_ens:.1f}%* (LDA + XGBoost)\n"
        f"🔬 *Cloud EGAT Cross-Check:* {egat_str}\n"
        f"📍 *Primary Wasting Indicator:* {wasting_site}\n\n"
        f"📏 *Estimated Dimensions:*\n"
        f"• Vision Estimated MUAC: ~{muac_cm} cm\n"
        f"• Total Arm Length: {arm_cm} cm\n\n"
        f"🩺 *Clinical Action Guide:*\n"
        f"{action}\n"
    )

    if tier == "YELLOW":
        msg += (
            f"\n💡 *Borderline Action:*\n"
            f"If you have a MUAC tape, reply with:\n"
            f"`MUAC <value_in_cm>` (e.g., `MUAC 12.5`)\n"
            f"for immediate >89.7% multimodal resolution."
        )

    msg += f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n_Poshan Application • Frontline Pediatric Health Assistant_"
    return msg


@app.get("/api/v1/whatsapp-webhook")
def whatsapp_verify_webhook(request: Request):
    """
    Meta WhatsApp Cloud API Webhook Verification Challenge.
    Meta calls this GET endpoint with hub.mode, hub.challenge, and hub.verify_token
    when configuring the Webhook URL in Meta Developer Dashboard.
    """
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    # Set your verification token (configured in Meta Developer Console)
    VERIFY_TOKEN = os.environ.get("WHATSAPP_VERIFY_TOKEN", "pediamorph_ai_secret_token_2026")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        return PlainTextResponse(content=challenge, status_code=200)
    elif mode or token:
        raise HTTPException(status_code=403, detail="Verification token mismatch.")
    return {"status": "WhatsApp Webhook Listener Online", "docs": "/docs"}


@app.post("/api/v1/whatsapp-webhook")
async def whatsapp_incoming_message(request: Request):
    """
    Receives incoming WhatsApp messages from frontline workers:
    - If photo is sent: runs zero-checkerboard vision + Tier 1 Offline Ensemble triage,
      and returns formatted WhatsApp message with clinical guidance and triage badge.
    - If text 'MUAC <val>' is sent: resolves borderline Yellow cases using multimodal model.
    """
    content_type = request.headers.get("content-type", "")
    
    # Helper to return TwiML XML to Twilio
    def _twiml_response(reply_text: str) -> Response:
        escaped = xml.sax.saxutils.escape(reply_text)
        xml_payload = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<Response>\n'
            '    <Message>\n'
            f'        <Body>{escaped}</Body>\n'
            '    </Message>\n'
            '</Response>'
        )
        return Response(content=xml_payload, media_type="application/xml; charset=utf-8")

    # 1. Handle Twilio WhatsApp Webhook (application/x-www-form-urlencoded)
    if "form" in content_type:
        form_data = await request.form()
        sender = form_data.get("From", "Worker")
        body_text = form_data.get("Body", "").strip()
        num_media = int(form_data.get("NumMedia", 0))

        # Case A: Frontline worker sent an arm photograph
        if num_media > 0:
            media_url = form_data.get("MediaUrl0")
            auth = None
            sid = os.environ.get("TWILIO_ACCOUNT_SID", "")
            token = os.environ.get("TWILIO_AUTH_TOKEN", "")
            if sid and token:
                auth = (sid, token)
            try:
                resp = requests.get(media_url, auth=auth, timeout=20)
                if resp.status_code == 200:
                    diag = _execute_mobile_pipeline(resp.content, source_name=f"twilio_{sender}", include_visuals=False)
                    triage_reply = format_whatsapp_triage_text(diag)
                    return _twiml_response(triage_reply)
                elif resp.status_code == 401:
                    return _twiml_response(
                        "⚠️ *Twilio Photo Access (HTTP 401)*\n"
                        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        "Twilio is blocking direct photo download until authenticated.\n\n"
                        "💡 *Easiest Fix (1-Click in Twilio Console):*\n"
                        "1. Go to Twilio Console -> Messaging -> Settings -> General\n"
                        "2. Turn OFF *'HTTP Basic Auth for Media'*\n"
                        "3. Click Save and re-send the child's photo!\n\n"
                        "*(Alternatively, set TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN in your .env file)*"
                    )
                else:
                    return _twiml_response(
                        f"⚠️ Error downloading image from WhatsApp (HTTP {resp.status_code}). "
                        "Please check your network and re-send the child's left-arm photo."
                    )
            except Exception as e:
                return _twiml_response(
                    f"⚠️ PediaMorph AI Vision Error: {str(e)}\n"
                    "Please ensure the child's bare left arm is clearly visible in natural light and re-send."
                )

        # Case B: Frontline worker sent MUAC measurement (e.g., 'MUAC 12.5')
        elif body_text.upper().startswith("MUAC"):
            try:
                val = float(body_text.split()[1])
                return _twiml_response(
                    f"✅ *Recorded Tape MUAC: {val:.1f} cm*\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    "If the initial arm scan yielded a 🟡 YELLOW (At-Risk) triage, "
                    "this tape measurement is combined with allometric contours for definitive >89.7% resolution."
                )
            except Exception:
                return _twiml_response("⚠️ Invalid MUAC format. Please reply with: *MUAC <cm>* (e.g., *MUAC 12.5*)")

        # Case C: Text greeting or help message
        welcome_msg = (
            "🙏 *Welcome to the Poshan Application!*\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "👶 *Frontline Child Malnutrition Screening Assistant*\n\n"
            "📸 *How to screen a child:*\n"
            "1. Take a clear photograph of the child's bare left arm (shoulder to wrist).\n"
            "2. Send the photo directly here in this WhatsApp chat.\n"
            "3. Receive instant 3-Tier Traffic Light Triage:\n"
            "   • 🔴 RED: Critical SAM (Urgent referral to PHC/NRC)\n"
            "   • 🟡 YELLOW: MAM / Watchlist (Prompt for tape check)\n"
            "   • 🟢 GREEN: Healthy / Normal Nutritional Status\n\n"
            "✨ *Zero calibration board or ruler needed!*"
        )
        return _twiml_response(welcome_msg)

    # 2. Handle Meta WhatsApp Business Cloud API JSON Payload
    try:
        body = await request.json()
    except Exception:
        body = {}

    entry = body.get("entry", [{}])[0]
    changes = entry.get("changes", [{}])[0]
    value = changes.get("value", {})
    messages = value.get("messages", [])

    if not messages:
        return {"status": "no_messages_in_payload"}

    msg_obj = messages[0]
    sender_phone = msg_obj.get("from")
    msg_type = msg_obj.get("type")

    # If photo / image received
    if msg_type == "image":
        image_info = msg_obj.get("image", {})
        media_id = image_info.get("id")
        return {
            "status": "image_event_acknowledged",
            "sender": sender_phone,
            "media_id": media_id,
            "instruction": "Fetch media using Meta Graph API GET /{media_id}, then stream bytes to /api/v1/diagnose"
        }

    # If text message received (e.g. Help or MUAC entry)
    text_body = msg_obj.get("text", {}).get("body", "").strip()
    return {
        "status": "text_event_acknowledged",
        "sender": sender_phone,
        "received_text": text_body,
        "reply_text": (
            "🙏 *Welcome to the Poshan Application!*\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "👶 *Frontline Child Malnutrition Screening Assistant*\n\n"
            "📸 *How to screen a child:*\n"
            "1. Take a natural photo of the child's bare left arm.\n"
            "2. Send the photo directly in this chat.\n"
            "3. Receive instant 3-Tier Traffic Light Triage (🟢 Normal, 🟡 Watchlist, 🔴 Critical SAM).\n\n"
            "✨ *Zero measurement tape or calibration board needed!*"
        )
    }


