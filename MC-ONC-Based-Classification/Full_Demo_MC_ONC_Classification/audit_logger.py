"""
audit_logger.py
===============
Timestamped Audit Logger & Quality Assurance (QA) Engine for Frontline Image Processing.
Saves the original input photograph, 22-landmark JSON, contour overlay, Cartesian plot,
composite side-by-side quality-check image, and Cloud EGAT attention heatmap for every request.
"""

import os
import json
import base64
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
AUDIT_LOGS_DIR = CURRENT_DIR / "audit_logs"
AUDIT_LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _decode_b64_to_bytes(b64_str: str) -> bytes:
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    return base64.b64decode(b64_str)


def save_audit_record(
    original_image_input: Union[bytes, str, Path],
    source_name: str,
    phase1_res: Dict,
    phase2_res: Dict,
    timing_sec: Dict
) -> Dict:
    """
    Creates a timestamped audit folder and persists all QA validation files for the request.
    
    Persisted files:
      1. 01_original_image.jpg
      2. 02_contour_overlay.png
      3. 03_cartesian_plot.png
      4. 04_side_by_side_quality_check.png
      5. 05_egat_attention_heatmap.png
      6. landmarks_22.json
      7. audit_summary.json
      
    Returns:
        Dict with audit_id, timestamp, folder path, and file links.
    """
    now = datetime.now()
    timestamp_iso = now.isoformat()
    folder_prefix = now.strftime("%Y%m%d_%H%M%S")
    unique_suffix = uuid.uuid4().hex[:8]
    audit_id = f"{folder_prefix}_{unique_suffix}"

    audit_dir = AUDIT_LOGS_DIR / audit_id
    audit_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []

    # 1. Save Original Image
    orig_path = audit_dir / "01_original_image.jpg"
    try:
        if isinstance(original_image_input, (bytes, bytearray)):
            with open(orig_path, "wb") as f:
                f.write(original_image_input)
        elif isinstance(original_image_input, (str, Path)):
            p = Path(original_image_input)
            if p.exists():
                with open(p, "rb") as f_in, open(orig_path, "wb") as f_out:
                    f_out.write(f_in.read())
        saved_files.append("01_original_image.jpg")
    except Exception as e:
        print(f"[AUDIT LOG WARNING] Could not save original image: {e}")

    # 2. Save Contour Overlay
    try:
        overlay_b64 = phase1_res.get("contour_overlay_base64", "")
        if overlay_b64:
            overlay_bytes = _decode_b64_to_bytes(overlay_b64)
            overlay_path = audit_dir / "02_contour_overlay.png"
            with open(overlay_path, "wb") as f:
                f.write(overlay_bytes)
            saved_files.append("02_contour_overlay.png")
    except Exception as e:
        print(f"[AUDIT LOG WARNING] Could not save contour overlay: {e}")

    # 3. Save Cartesian Plot
    try:
        cart_b64 = phase1_res.get("cartesian_plot_base64", "")
        if cart_b64:
            cart_bytes = _decode_b64_to_bytes(cart_b64)
            cart_path = audit_dir / "03_cartesian_plot.png"
            with open(cart_path, "wb") as f:
                f.write(cart_bytes)
            saved_files.append("03_cartesian_plot.png")
    except Exception as e:
        print(f"[AUDIT LOG WARNING] Could not save cartesian plot: {e}")

    # 4. Save Side-By-Side Quality Check Composite
    try:
        sbs_b64 = phase1_res.get("side_by_side_base64", "")
        if sbs_b64:
            sbs_bytes = _decode_b64_to_bytes(sbs_b64)
            sbs_path = audit_dir / "04_side_by_side_quality_check.png"
            with open(sbs_path, "wb") as f:
                f.write(sbs_bytes)
            saved_files.append("04_side_by_side_quality_check.png")
    except Exception as e:
        print(f"[AUDIT LOG WARNING] Could not save side-by-side: {e}")

    # 5. Save EGAT Attention Heatmap
    try:
        egat_data = phase2_res.get("cloud_egat", {})
        egat_b64 = egat_data.get("attention_heatmap_base64", "")
        if egat_b64:
            egat_bytes = _decode_b64_to_bytes(egat_b64)
            egat_path = audit_dir / "05_egat_attention_heatmap.png"
            with open(egat_path, "wb") as f:
                f.write(egat_bytes)
            saved_files.append("05_egat_attention_heatmap.png")
    except Exception as e:
        print(f"[AUDIT LOG WARNING] Could not save EGAT heatmap: {e}")

    # 6. Save Landmarks 22 JSON (Exact structure with full precision)
    landmarks = phase1_res.get("landmarks", [])
    lm_path = audit_dir / "landmarks_22.json"
    try:
        with open(lm_path, "w", encoding="utf-8") as f:
            json.dump(landmarks, f, indent=2)
        saved_files.append("landmarks_22.json")
    except Exception as e:
        print(f"[AUDIT LOG WARNING] Could not save landmarks JSON: {e}")

    # 7. Quality Assurance Checks & Audit Summary
    muac_val = phase1_res.get("muac_cm")
    arm_len_val = phase1_res.get("total_arm_cm")
    egat_res = phase2_res.get("cloud_egat", {})

    qa_passed = True
    qa_flags = []

    if len(landmarks) != 22:
        qa_passed = False
        qa_flags.append(f"Incomplete landmarks count: {len(landmarks)}/22 detected.")

    if muac_val is None or muac_val < 8.0 or muac_val > 22.0:
        qa_flags.append(f"MUAC measurement ({muac_val} cm) outside expected pediatric range (8-22 cm).")

    if arm_len_val is None or arm_len_val < 15.0 or arm_len_val > 45.0:
        qa_flags.append(f"Arm length ({arm_len_val} cm) outside expected pediatric range.")

    summary = {
        "audit_id": audit_id,
        "timestamp_iso": timestamp_iso,
        "source_filename": source_name,
        "qa_status": "PASS" if qa_passed and len(qa_flags) == 0 else "FLAGGED_FOR_REVIEW",
        "qa_flags": qa_flags,
        "timing_sec": timing_sec,
        "biometrics": {
            "arm_side_detected": phase1_res.get("arm_side", "Left"),
            "scale_px_per_cm": phase1_res.get("px_per_cm"),
            "total_arm_length_cm": arm_len_val,
            "upper_arm_length_cm": phase1_res.get("upper_arm_cm"),
            "forearm_length_cm": phase1_res.get("forearm_cm"),
            "vision_estimated_muac_cm": muac_val,
            "centroid_size": phase2_res.get("centroid_size")
        },
        "clinical_triage": {
            "tier": egat_res.get("triage_tier"),
            "badge": egat_res.get("clinical_badge"),
            "category": egat_res.get("nutritional_category"),
            "malnutrition_probability": egat_res.get("probability_egat"),
            "primary_wasting_locus": egat_res.get("primary_wasting_site"),
            "clinical_action": egat_res.get("clinical_action")
        },
        "saved_files": saved_files,
        "storage_path": str(audit_dir.resolve())
    }

    summary_path = audit_dir / "audit_summary.json"
    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        saved_files.append("audit_summary.json")
    except Exception as e:
        print(f"[AUDIT LOG WARNING] Could not save audit summary: {e}")

    return {
        "audit_id": audit_id,
        "timestamp": timestamp_iso,
        "qa_status": summary["qa_status"],
        "storage_dir": str(audit_dir.name),
        "saved_files_count": len(saved_files),
        "saved_files": saved_files
    }


def list_recent_audit_logs(limit: int = 50) -> List[Dict]:
    """Retrieves metadata of the most recent audit logs for quality inspection."""
    records = []
    if not AUDIT_LOGS_DIR.exists():
        return records

    dirs = sorted([d for d in AUDIT_LOGS_DIR.iterdir() if d.is_dir()], reverse=True)
    for d in dirs[:limit]:
        sum_path = d / "audit_summary.json"
        if sum_path.exists():
            try:
                with open(sum_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    records.append({
                        "audit_id": data.get("audit_id", d.name),
                        "timestamp": data.get("timestamp_iso"),
                        "source": data.get("source_filename"),
                        "qa_status": data.get("qa_status"),
                        "triage_tier": data.get("clinical_triage", {}).get("tier"),
                        "malnutrition_probability": data.get("clinical_triage", {}).get("malnutrition_probability"),
                        "estimated_muac_cm": data.get("biometrics", {}).get("vision_estimated_muac_cm"),
                        "files_available": data.get("saved_files", [])
                    })
            except Exception:
                continue
    return records


def get_audit_record(audit_id: str) -> Optional[Dict]:
    """Retrieves full details of a specific audit log record."""
    audit_dir = AUDIT_LOGS_DIR / audit_id
    if not audit_dir.exists() or not audit_dir.is_dir():
        return None

    sum_path = audit_dir / "audit_summary.json"
    if sum_path.exists():
        with open(sum_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"audit_id": audit_id, "status": "no summary found"}
