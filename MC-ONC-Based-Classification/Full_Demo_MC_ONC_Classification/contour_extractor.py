"""
contour_extractor.py
====================
Frontline Child Left-Arm Contour & 22-Landmark Extraction Engine.
NO CHECKERBOARD REQUIRED: Designed for direct natural child arm photography in community health settings.
Uses MediaPipe Pose Landmarker + Multiclass Segmentation + Perpendicular Morphometric Scanning.
"""

import os
import cv2
import numpy as np
import base64
import io
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

CURRENT_DIR = Path(__file__).resolve().parent
MODEL_DIR = CURRENT_DIR / "poselandmark_model"

# Singleton model instances
_landmarker_instance = None
_segmenter_instance = None


def get_pose_landmarker():
    global _landmarker_instance
    if _landmarker_instance is None:
        pose_model_path = MODEL_DIR / "pose_landmarker_heavy.task"
        if not pose_model_path.exists():
            # Fallback path if run from other cwd
            pose_model_path = Path(r"E:\child-data-annotation-codebase\geometric_morphometric_image_annotation-main\geometric_morphometric_image_annotation-main\poselandmark_model\pose_landmarker_heavy.task")
        
        base_options_pose = python.BaseOptions(model_asset_path=str(pose_model_path))
        options_pose = vision.PoseLandmarkerOptions(
            base_options=base_options_pose,
            output_segmentation_masks=False,
            num_poses=1
        )
        _landmarker_instance = vision.PoseLandmarker.create_from_options(options_pose)
    return _landmarker_instance


def get_image_segmenter():
    global _segmenter_instance
    if _segmenter_instance is None:
        seg_model_path = MODEL_DIR / "selfie_multiclass_256x256.tflite"
        if not seg_model_path.exists():
            seg_model_path = Path(r"E:\child-data-annotation-codebase\geometric_morphometric_image_annotation-main\geometric_morphometric_image_annotation-main\poselandmark_model\selfie_multiclass_256x256.tflite")
            
        base_options_seg = python.BaseOptions(model_asset_path=str(seg_model_path))
        options_seg = vision.ImageSegmenterOptions(
            base_options=base_options_seg,
            output_category_mask=True
        )
        _segmenter_instance = vision.ImageSegmenter.create_from_options(options_seg)
    return _segmenter_instance


def _find_contours(img, mode, method):
    res = cv2.findContours(img, mode, method)
    return res[0] if len(res) == 2 else res[1]


def _threshold(img, thresh, maxval, type_):
    res = cv2.threshold(img, thresh, maxval, type_)
    return res[1] if isinstance(res, (tuple, list)) and len(res) > 1 else res


def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=140, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{b64}"


def cv2_img_to_base64(img) -> str:
    success, encoded = cv2.imencode('.png', img)
    if not success:
        return ""
    b64 = base64.b64encode(encoded.tobytes()).decode('utf-8')
    return f"data:image/png;base64,{b64}"


def extract_arm_contour_and_landmarks(image_input, canonical_arm_cm: float = 28.5) -> dict:
    """
    Extracts 22 anatomical contour landmarks from a child left arm photograph.
    NO CHECKERBOARD NEEDED: Normalizes scale using the anatomical arm length.
    
    Args:
        image_input: File path (str/Path) OR image bytes OR numpy BGR array.
        canonical_arm_cm: Standard pediatric left arm length for metric calibration (default: 28.5 cm).
        
    Returns:
        dict containing:
            - px_per_cm: calibrated pixel-to-cm ratio
            - total_arm_cm, upper_arm_cm, forearm_cm, muac_cm
            - landmarks: List of 22 landmark items with ID, Name, coordinates
            - raw_coordinates: Dict of {X1: ..., Y1: ..., ... X22: ..., Y22: ...}
            - contour_overlay_base64: Base64 data URL of arm contour visual
            - cartesian_plot_base64: Base64 data URL of wireframe Cartesian plot
            - side_by_side_base64: Base64 data URL of composite comparison
    """
    # 1. Ingest image
    if isinstance(image_input, (str, Path)):
        image_full = cv2.imread(str(image_input))
        if image_full is None:
            raise ValueError(f"Failed to load image from: {image_input}")
    elif isinstance(image_input, (bytes, bytearray)):
        nparr = np.frombuffer(image_input, np.uint8)
        image_full = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image_full is None:
            raise ValueError("Failed to decode image from bytes.")
    elif isinstance(image_input, np.ndarray):
        image_full = image_input.copy()
    else:
        raise TypeError("Unsupported image_input type. Expected path, bytes, or numpy array.")

    h_full, w_full = image_full.shape[:2]
    WORK_MAX_PX = 1500
    work_scale = min(WORK_MAX_PX / max(w_full, h_full), 1.0)
    if work_scale < 1.0:
        w = int(w_full * work_scale)
        h = int(h_full * work_scale)
        image = cv2.resize(image_full, (w, h), interpolation=cv2.INTER_AREA)
    else:
        image = image_full
        w, h = w_full, h_full

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if rgb_image.shape[2] == 4:
        rgb_image = rgb_image[:, :, :3]
    rgb_image = np.ascontiguousarray(rgb_image)

    # 2. MediaPipe Pose Landmark Detection
    landmarker = get_pose_landmarker()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    pose_result = landmarker.detect(mp_image)

    if not pose_result.pose_landmarks or len(pose_result.pose_landmarks) == 0:
        raise ValueError("No child pose detected in photograph. Ensure the child's arm is visible and unobstructed.")

    landmarks_pose = pose_result.pose_landmarks[0]
    
    # Check left arm vs right arm visibility
    left_s, left_e, left_w = landmarks_pose[11], landmarks_pose[13], landmarks_pose[15]
    right_s, right_e, right_w = landmarks_pose[12], landmarks_pose[14], landmarks_pose[16]

    left_vis = (getattr(left_s, 'visibility', 0) + getattr(left_e, 'visibility', 0) + getattr(left_w, 'visibility', 0)) / 3.0
    right_vis = (getattr(right_s, 'visibility', 0) + getattr(right_e, 'visibility', 0) + getattr(right_w, 'visibility', 0)) / 3.0

    if right_vis > left_vis and right_vis > 0.35:
        shoulder, elbow, wrist = right_s, right_e, right_w
        arm_side = "Right (Mirrored to Left)"
    else:
        shoulder, elbow, wrist = left_s, left_e, left_w
        arm_side = "Left"

    def to_px(lm):
        return (int(round(lm.x * w)), int(round(lm.y * h)))

    shoulder_coords = to_px(shoulder)
    elbow_coords    = to_px(elbow)
    wrist_coords    = to_px(wrist)

    # 3. Segmentation Mask Generation & Arm Region Refinement
    segmenter = get_image_segmenter()
    segmenter_result = segmenter.segment(mp_image)
    category_mask = segmenter_result.category_mask.numpy_view()

    # Category 2 = skin, Category 4 = clothes -> body mask
    person_mask = np.where((category_mask == 2) | (category_mask == 4), 255, 0).astype(np.uint8)

    upper_arm_len_px = float(np.linalg.norm(np.array(elbow_coords) - np.array(shoulder_coords)))
    fore_arm_len_px  = float(np.linalg.norm(np.array(wrist_coords) - np.array(elbow_coords)))
    total_arm_len_px = float(np.linalg.norm(np.array(wrist_coords) - np.array(shoulder_coords)))

    upper_thick = max(18, int(upper_arm_len_px * 0.32))
    fore_thick  = max(14, int(fore_arm_len_px  * 0.28))

    mask_roi = np.zeros((h, w), dtype=np.uint8)
    cv2.line(mask_roi, shoulder_coords, elbow_coords, 255, upper_thick)
    cv2.line(mask_roi, elbow_coords,    wrist_coords, 255, fore_thick)
    cv2.circle(mask_roi, shoulder_coords, upper_thick // 2, 255, -1)
    cv2.circle(mask_roi, elbow_coords, max(upper_thick, fore_thick) // 2, 255, -1)
    cv2.circle(mask_roi, wrist_coords, fore_thick // 2, 255, -1)
    mask_roi = cv2.dilate(mask_roi, np.ones((11, 11), np.uint8), iterations=1)

    arm_mask = cv2.bitwise_and(person_mask, mask_roi)
    if cv2.countNonZero(arm_mask) < 100:
        arm_mask = mask_roi.copy()

    # Morphological contour smoothing
    arm_mask = cv2.morphologyEx(arm_mask, cv2.MORPH_OPEN,  np.ones((3, 3),  np.uint8), iterations=1)
    arm_mask = cv2.morphologyEx(arm_mask, cv2.MORPH_CLOSE, np.ones((7, 7),  np.uint8), iterations=2)
    arm_mask = cv2.morphologyEx(arm_mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8), iterations=1)

    contours = _find_contours(arm_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        lc = max(contours, key=cv2.contourArea)
        eps = 0.002 * cv2.arcLength(lc, True)
        smooth = cv2.approxPolyDP(lc, eps, True)
        arm_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(arm_mask, [smooth], -1, 255, -1)
        arm_mask = cv2.GaussianBlur(arm_mask, (9, 9), 0)
        arm_mask = _threshold(arm_mask, 127, 255, cv2.THRESH_BINARY)

    arm_mask = cv2.GaussianBlur(arm_mask, (5, 5), 0)
    arm_mask = _threshold(arm_mask, 127, 255, cv2.THRESH_BINARY)

    for pt in [shoulder_coords, elbow_coords, wrist_coords]:
        cv2.circle(arm_mask, pt, 8, 255, -1)

    # 4. Canonical Anatomical Scale Calibration (NO CHECKERBOARD)
    # The left arm length from shoulder to wrist is calibrated to canonical_arm_cm (default 28.5 cm)
    # This aligns the extracted Cartesian coordinates directly with the reference dataset.
    px_per_cm = max(total_arm_len_px / canonical_arm_cm, 1.0)

    def px_to_cm(pixels):
        return pixels / px_per_cm

    origin_px = shoulder_coords

    def to_cartesian(pt_px):
        if pt_px is None:
            return (0.0, 0.0)
        x_cm = (pt_px[0] - origin_px[0]) / px_per_cm
        y_cm = -(pt_px[1] - origin_px[1]) / px_per_cm  # Up is positive
        return (round(x_cm, 3), round(y_cm, 3))

    # 5. Perpendicular Intersections & Semi-Landmarks
    def find_perpendicular_intersections(point, direction_vector, mask_in, max_length=250, is_endpoint=False):
        dx, dy = direction_vector
        length = np.sqrt(dx**2 + dy**2)
        if length == 0:
            return None, None

        perp_x = -dy / length
        perp_y =  dx / length

        if perp_y > 0:
            perp_x = -perp_x
            perp_y = -perp_y

        arm_x  =  dx / length
        arm_y  =  dy / length

        contours_all = _find_contours(mask_in.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours_all:
            return None, None
        contour_pts = max(contours_all, key=cv2.contourArea).reshape(-1, 2)

        def scan(px, py):
            last_in = None
            for d_idx in range(1, max_length):
                tx = int(point[0] + px * d_idx)
                ty = int(point[1] + py * d_idx)
                if not (0 <= tx < w and 0 <= ty < h):
                    break
                if mask_in[ty, tx] == 255:
                    last_in = (tx, ty)
                elif last_in:
                    return last_in
            if last_in:
                return last_in
            cands = []
            for cp in contour_pts:
                vec = (cp[0] - point[0], cp[1] - point[1])
                pp = vec[0] * px + vec[1] * py
                if pp > 5:
                    ap = abs(vec[0] * arm_x + vec[1] * arm_y)
                    if ap < 60:
                        cands.append((pp, tuple(cp)))
            if cands:
                return max(cands, key=lambda c: c[0])[1]
            return None

        top    = scan( perp_x,  perp_y)
        bottom = scan(-perp_x, -perp_y)

        if is_endpoint and top and bottom is None:
            top_dist = np.linalg.norm(np.array(top) - np.array(point))
            cands = []
            for cp in contour_pts:
                vec = (cp[0] - point[0], cp[1] - point[1])
                pp = vec[0] * (-perp_x) + vec[1] * (-perp_y)
                if pp > 5:
                    d_val = np.linalg.norm(vec)
                    ap = abs(vec[0] * arm_x + vec[1] * arm_y)
                    dd = abs(d_val - top_dist)
                    if dd < top_dist * 0.25 and ap < 40:
                        cands.append((dd, ap, tuple(cp)))
            if cands:
                bottom = sorted(cands, key=lambda c: (c[0], c[1]))[0][2]

        return top, bottom

    def add_semi_landmarks(start, end, direction, mask_in, n=3):
        result = []
        for i_idx in range(1, n + 1):
            r_val = i_idx / (n + 1)
            cx_val = int(start[0] + (end[0] - start[0]) * r_val)
            cy_val = int(start[1] + (end[1] - start[1]) * r_val)
            top_pt, bot_pt = find_perpendicular_intersections((cx_val, cy_val), direction, mask_in)
            result.append({'center': (cx_val, cy_val), 'top': top_pt, 'bottom': bot_pt})
        return result

    shoulder_dir = (elbow_coords[0] - shoulder_coords[0], elbow_coords[1] - shoulder_coords[1])
    elbow_dir    = (wrist_coords[0]  - shoulder_coords[0], wrist_coords[1]  - shoulder_coords[1])
    wrist_dir    = (wrist_coords[0]  - elbow_coords[0],   wrist_coords[1]  - elbow_coords[1])

    shoulder_top, shoulder_bottom = find_perpendicular_intersections(shoulder_coords, shoulder_dir, arm_mask, is_endpoint=True)
    elbow_top,    elbow_bottom    = find_perpendicular_intersections(elbow_coords,    elbow_dir,    arm_mask, is_endpoint=False)
    wrist_top,    wrist_bottom    = find_perpendicular_intersections(wrist_coords,    wrist_dir,    arm_mask, is_endpoint=True)

    upper_semis   = add_semi_landmarks(shoulder_coords, elbow_coords, shoulder_dir, arm_mask, n=3)
    forearm_semis = add_semi_landmarks(elbow_coords,   wrist_coords, wrist_dir,    arm_mask, n=3)

    # 6. Physical Dimensions & Estimated MUAC
    def measure(pt1, pt2):
        if pt1 is None or pt2 is None:
            return None
        return round(px_to_cm(np.linalg.norm(np.array(pt1) - np.array(pt2))), 2)

    upper_arm_cm = measure(shoulder_coords, elbow_coords)
    forearm_cm   = measure(elbow_coords,    wrist_coords)
    total_arm_cm = measure(shoulder_coords, wrist_coords)

    upper_semi_widths = [measure(s['top'], s['bottom']) for s in upper_semis]
    muac_width_cm = upper_semi_widths[1] if len(upper_semi_widths) > 1 and upper_semi_widths[1] is not None else None
    muac_cm = round(muac_width_cm * np.pi, 2) if muac_width_cm else None

    # 7. Build Structured Landmark Dictionary (22 Landmarks)
    landmark_data = []

    def add_numbered_lm(lm_id, ltype, cat, name, pt_px, color):
        if pt_px is None:
            pt_px = (0, 0)
        x_cm = round(pt_px[0] / px_per_cm, 3)
        y_cm = round(pt_px[1] / px_per_cm, 3)
        cx_cart, cy_cart = to_cartesian(pt_px)
        landmark_data.append({
            "Landmark_ID": lm_id,
            "Type": ltype,
            "Category": cat,
            "Name": name,
            "X_Pixel": float(pt_px[0]),
            "Y_Pixel": float(pt_px[1]),
            "X_cm": float(x_cm),
            "Y_cm": float(y_cm),
            "X_cartesian_cm": float(cx_cart),
            "Y_cartesian_cm": float(cy_cart),
            "Color": color
        })

    add_numbered_lm(21, 'Anatomical', 'Main', 'Shoulder_Center', shoulder_coords, 'red')
    add_numbered_lm(2,  'Anatomical', 'Main', 'Elbow_Center',    elbow_coords,    'red')
    add_numbered_lm(3,  'Anatomical', 'Main', 'Wrist_Center',    wrist_coords,    'red')

    armpit_coords = upper_semis[0]['bottom'] if len(upper_semis) > 0 and upper_semis[0]['bottom'] else (shoulder_coords[0], shoulder_coords[1] + 10)
    add_numbered_lm(4,  'Anatomical', 'Main', 'Armpit_Same_As_11', armpit_coords, 'red')

    if len(forearm_semis) >= 3:
        for i_idx in range(2, -1, -1):
            s = forearm_semis[i_idx]
            lm_id = 5 + (2 - i_idx) * 2
            if s['bottom']:
                add_numbered_lm(lm_id, 'Semilandmark', 'Forearm', f'Forearm_Semi{i_idx+1}_Bottom', s['bottom'], 'yellow')
            if s['top']:
                add_numbered_lm(lm_id + 1, 'Semilandmark', 'Forearm', f'Forearm_Semi{i_idx+1}_Top', s['top'], 'yellow')

    if len(upper_semis) >= 3:
        for i_idx in range(3):
            s = upper_semis[i_idx]
            lm_id = 11 + i_idx * 2
            if s['bottom']:
                color = 'red' if lm_id == 11 else 'yellow'
                add_numbered_lm(lm_id, 'Semilandmark', 'UpperArm', f'UpperArm_Semi{i_idx+1}_Bottom', s['bottom'], color)
            if s['top']:
                add_numbered_lm(lm_id + 1, 'Semilandmark', 'UpperArm', f'UpperArm_Semi{i_idx+1}_Top', s['top'], 'yellow')

    if elbow_bottom:
        add_numbered_lm(17, 'Anatomical', 'Main', 'Elbow_Bottom', elbow_bottom, 'yellow')
    if elbow_top:
        add_numbered_lm(18, 'Anatomical', 'Main', 'Elbow_Top',    elbow_top,    'yellow')

    if wrist_bottom:
        add_numbered_lm(19, 'Anatomical', 'Main', 'Wrist_Bottom', wrist_bottom, 'yellow')
    if wrist_top:
        add_numbered_lm(20, 'Anatomical', 'Main', 'Wrist_Top',    wrist_top,    'yellow')

    if shoulder_top:
        add_numbered_lm(1, 'Semilandmark', 'Main', 'Shoulder_Top', shoulder_top, 'red')
    else:
        add_numbered_lm(1, 'Semilandmark', 'Main', 'Shoulder_Top_Fallback', (shoulder_coords[0], shoulder_coords[1] - 20), 'red')

    if shoulder_bottom:
        add_numbered_lm(22, 'Anatomical', 'Main', 'Shoulder_Bottom', shoulder_bottom, 'yellow')

    landmark_data.sort(key=lambda x: x["Landmark_ID"])

    # Build raw coordinate dictionary {X1: ..., Y1: ...} for Phase 2 classification
    raw_coords_dict = {}
    for lm in landmark_data:
        raw_coords_dict[f"X{lm['Landmark_ID']}"] = lm["X_cartesian_cm"]
        raw_coords_dict[f"Y{lm['Landmark_ID']}"] = lm["Y_cartesian_cm"]

    # 8. Render Visual Overlays
    img_vis = image.copy()

    # Translucent mask overlay
    mask_color = np.zeros_like(image)
    mask_color[:, :, 1] = arm_mask
    mask_color[:, :, 2] = (arm_mask * 0.3).astype(np.uint8)
    img_vis = cv2.addWeighted(img_vis, 1.0, mask_color, 0.12, 0)

    # Contour outline
    ctrs = _find_contours(arm_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if ctrs:
        cv2.drawContours(img_vis, [max(ctrs, key=cv2.contourArea)], -1, (0, 230, 230), 2)

    # Perpendicular lines & caliper connections
    CYAN = (255, 255, 0)
    def draw_perp(pt1, pt2):
        if pt1 and pt2:
            cv2.line(img_vis, pt1, pt2, CYAN, 1)

    draw_perp(shoulder_top, shoulder_bottom)
    draw_perp(elbow_top, elbow_bottom)
    draw_perp(wrist_top, wrist_bottom)
    for s in upper_semis + forearm_semis:
        draw_perp(s['top'], s['bottom'])

    cv2.line(img_vis, shoulder_coords, elbow_coords, (0, 255, 255), 2)
    cv2.line(img_vis, elbow_coords, wrist_coords, (0, 255, 255), 2)

    # Draw numbered landmarks
    for lm in landmark_data:
        pt = (int(lm['X_Pixel']), int(lm['Y_Pixel']))
        color = (0, 0, 255) if lm['Color'] == 'red' else (0, 255, 255)
        cv2.circle(img_vis, pt, 4, color, -1)
        cv2.circle(img_vis, pt, 4, (0, 0, 0), 1)
        cv2.putText(img_vis, str(lm['Landmark_ID']), (pt[0] + 5, pt[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img_vis, str(lm['Landmark_ID']), (pt[0] + 5, pt[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    contour_overlay_b64 = cv2_img_to_base64(img_vis)

    # 9. Render Cartesian Plot (Exact Clinical Morphometric Layout)
    fig_cart, ax_cart = plt.subplots(figsize=(10, 5), facecolor='#ffffff')
    ax_cart.set_facecolor('#ffffff')
    ax_cart.axhline(0, color='#cbd5e1', linewidth=0.8, linestyle=':')
    ax_cart.axvline(0, color='#cbd5e1', linewidth=0.8, linestyle=':')

    lm_dict = {lm['Landmark_ID']: (lm['X_cartesian_cm'], lm['Y_cartesian_cm']) for lm in landmark_data}

    # Upper contour points: 1, 12, 14, 16, 18, 10, 8, 6, 20
    upper_seq = [1, 12, 14, 16, 18, 10, 8, 6, 20]
    upper_coords = [lm_dict[lid] for lid in upper_seq if lid in lm_dict]

    # Lower contour points: 4 (or 11), 13, 15, 17, 9, 7, 5, 19
    lower_seq = [4, 13, 15, 17, 9, 7, 5, 19]
    lower_coords = [lm_dict[lid] for lid in lower_seq if lid in lm_dict]

    if len(upper_coords) > 1:
        ax_cart.plot([p[0] for p in upper_coords], [p[1] for p in upper_coords],
                     color='#ef4444', lw=2.2, label='Upper edge', zorder=3)
    if len(lower_coords) > 1:
        ax_cart.plot([p[0] for p in lower_coords], [p[1] for p in lower_coords],
                     color='#2563eb', lw=2.2, label='Lower edge', zorder=3)

    # Landmark 1 -> 12 Connection
    if 1 in lm_dict and 12 in lm_dict:
        ax_cart.plot([lm_dict[1][0], lm_dict[12][0]], [lm_dict[1][1], lm_dict[12][1]],
                     color='#16a34a', lw=2.2, label='Landmark 1->12 Connection', zorder=4)

    # Landmark 1 -> 11/4 Connection
    if 1 in lm_dict and 4 in lm_dict:
        ax_cart.plot([lm_dict[1][0], lm_dict[4][0]], [lm_dict[1][1], lm_dict[4][1]],
                     color='#15803d', lw=2.2, label='Landmark 1->11/4 Connection', zorder=4)

    # Landmark 12 -> 11/4 Vertical
    if 12 in lm_dict and 4 in lm_dict:
        ax_cart.plot([lm_dict[12][0], lm_dict[4][0]], [lm_dict[12][1], lm_dict[4][1]],
                     color='#64748b', lw=1.2, linestyle='--', label='Landmark 12->11/4 Vertical', zorder=2)

    # Wrist closure (20 -> 19)
    if 20 in lm_dict and 19 in lm_dict:
        ax_cart.plot([lm_dict[20][0], lm_dict[19][0]], [lm_dict[20][1], lm_dict[19][1]],
                     color='#16a34a', lw=2.0, zorder=4)

    # Transverse Calipers (dashed lines)
    caliper_pairs = [(14, 13), (16, 15), (18, 17), (10, 9), (8, 7), (6, 5)]
    for idx, (t_id, b_id) in enumerate(caliper_pairs):
        if t_id in lm_dict and b_id in lm_dict:
            c1, c2 = lm_dict[t_id], lm_dict[b_id]
            is_muac = (t_id == 16 and b_id == 15)
            c_color = '#e11d48' if is_muac else '#94a3b8'
            c_lw = 2.0 if is_muac else 1.0
            ax_cart.plot([c1[0], c2[0]], [c1[1], c2[1]], color=c_color, lw=c_lw, linestyle='--', zorder=2)

    # Plot anatomical joints
    if 21 in lm_dict:
        ax_cart.scatter(lm_dict[21][0], lm_dict[21][1], color='#dc2626', s=60, zorder=6)
        ax_cart.annotate('Shoulder', lm_dict[21], xytext=(0, 6), textcoords='offset points',
                         fontsize=8.5, fontweight='bold', color='#dc2626')
    if 2 in lm_dict:
        ax_cart.scatter(lm_dict[2][0], lm_dict[2][1], color='#ea580c', s=60, zorder=6)
        ax_cart.annotate('Elbow', lm_dict[2], xytext=(4, 4), textcoords='offset points',
                         fontsize=8.5, fontweight='bold', color='#ea580c')
    if 3 in lm_dict:
        ax_cart.scatter(lm_dict[3][0], lm_dict[3][1], color='#7c3aed', s=60, zorder=6)
        ax_cart.annotate('Wrist', lm_dict[3], xytext=(4, 4), textcoords='offset points',
                         fontsize=8.5, fontweight='bold', color='#7c3aed')

    # Plot all 22 landmarks
    for lm in landmark_data:
        pt_c = (lm['X_cartesian_cm'], lm['Y_cartesian_cm'])
        lid = lm['Landmark_ID']
        if lid in [21, 2, 3]:
            continue
        pt_color = '#16a34a' if lid in [12, 4, 11] else ('#dc2626' if lid == 1 else '#f59e0b')
        ax_cart.scatter(pt_c[0], pt_c[1], color=pt_color, s=32, zorder=5)
        lbl = "+1" if lid == 1 else ("11/4" if lid == 4 else str(lid))
        ax_cart.annotate(lbl, pt_c, xytext=(2, 4), textcoords='offset points',
                         fontsize=8, fontweight='bold', color=pt_color)

    ax_cart.set_title(f"Cartesian Landmark Plot\nShoulder->Wrist: {total_arm_cm:.3f} cm | MUAC: {muac_cm:.3f} cm" if total_arm_cm and muac_cm else "Cartesian Landmark Plot",
                      fontsize=10.5, fontweight='bold')
    ax_cart.set_xlabel('X (cm) -> right', fontsize=9.5)
    ax_cart.set_ylabel('Y (cm) -> up', fontsize=9.5)
    ax_cart.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax_cart.grid(True, linestyle=':', alpha=0.6)
    ax_cart.set_aspect('equal')
    plt.tight_layout()

    cartesian_plot_b64 = fig_to_base64(fig_cart)

    # 10. Render Side-by-Side Quality Check Composite (Arm Overlay + Cartesian Plot)
    fig_sbs, (ax_sbs1, ax_sbs2) = plt.subplots(1, 2, figsize=(15, 6), facecolor='#ffffff')
    
    # Left: Arm Overlay
    ax_sbs1.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
    ax_sbs1.set_title(f"Arm Overlay | scale: {px_per_cm:.2f} px/cm", fontsize=11, fontweight='bold')
    ax_sbs1.axis('off')

    # Right: Cartesian Landmark Plot
    ax_sbs2.set_facecolor('#ffffff')
    ax_sbs2.axhline(0, color='#cbd5e1', linewidth=0.8, linestyle=':')
    ax_sbs2.axvline(0, color='#cbd5e1', linewidth=0.8, linestyle=':')

    if len(upper_coords) > 1:
        ax_sbs2.plot([p[0] for p in upper_coords], [p[1] for p in upper_coords], color='#ef4444', lw=2.2, label='Upper edge')
    if len(lower_coords) > 1:
        ax_sbs2.plot([p[0] for p in lower_coords], [p[1] for p in lower_coords], color='#2563eb', lw=2.2, label='Lower edge')
    if 1 in lm_dict and 12 in lm_dict:
        ax_sbs2.plot([lm_dict[1][0], lm_dict[12][0]], [lm_dict[1][1], lm_dict[12][1]], color='#16a34a', lw=2.0, label='Landmark 1->12 Connection')
    if 1 in lm_dict and 4 in lm_dict:
        ax_sbs2.plot([lm_dict[1][0], lm_dict[4][0]], [lm_dict[1][1], lm_dict[4][1]], color='#15803d', lw=2.0, label='Landmark 1->11/4 Connection')
    if 12 in lm_dict and 4 in lm_dict:
        ax_sbs2.plot([lm_dict[12][0], lm_dict[4][0]], [lm_dict[12][1], lm_dict[4][1]], color='#64748b', lw=1.2, linestyle='--', label='Landmark 12->11/4 Vertical')
    if 20 in lm_dict and 19 in lm_dict:
        ax_sbs2.plot([lm_dict[20][0], lm_dict[19][0]], [lm_dict[20][1], lm_dict[19][1]], color='#16a34a', lw=2.0)

    for idx, (t_id, b_id) in enumerate(caliper_pairs):
        if t_id in lm_dict and b_id in lm_dict:
            c1, c2 = lm_dict[t_id], lm_dict[b_id]
            is_muac = (t_id == 16 and b_id == 15)
            ax_sbs2.plot([c1[0], c2[0]], [c1[1], c2[1]], color='#e11d48' if is_muac else '#94a3b8',
                         lw=1.8 if is_muac else 1.0, linestyle='--')

    if 21 in lm_dict:
        ax_sbs2.scatter(lm_dict[21][0], lm_dict[21][1], color='#dc2626', s=50, zorder=6)
        ax_sbs2.annotate('Shoulder', lm_dict[21], xytext=(0, 5), textcoords='offset points', fontsize=8, fontweight='bold', color='#dc2626')
    if 2 in lm_dict:
        ax_sbs2.scatter(lm_dict[2][0], lm_dict[2][1], color='#ea580c', s=50, zorder=6)
        ax_sbs2.annotate('Elbow', lm_dict[2], xytext=(4, 4), textcoords='offset points', fontsize=8, fontweight='bold', color='#ea580c')
    if 3 in lm_dict:
        ax_sbs2.scatter(lm_dict[3][0], lm_dict[3][1], color='#7c3aed', s=50, zorder=6)
        ax_sbs2.annotate('Wrist', lm_dict[3], xytext=(4, 4), textcoords='offset points', fontsize=8, fontweight='bold', color='#7c3aed')

    for lm in landmark_data:
        pt_c = (lm['X_cartesian_cm'], lm['Y_cartesian_cm'])
        lid = lm['Landmark_ID']
        if lid in [21, 2, 3]:
            continue
        pt_color = '#16a34a' if lid in [12, 4, 11] else ('#dc2626' if lid == 1 else '#f59e0b')
        ax_sbs2.scatter(pt_c[0], pt_c[1], color=pt_color, s=28, zorder=5)
        lbl = "+1" if lid == 1 else ("11/4" if lid == 4 else str(lid))
        ax_sbs2.annotate(lbl, pt_c, xytext=(2, 3), textcoords='offset points', fontsize=7.5, fontweight='bold', color=pt_color)

    ax_sbs2.set_title(f"Cartesian Landmark Plot\nShoulder->Wrist: {total_arm_cm:.3f} cm | MUAC: {muac_cm:.3f} cm" if total_arm_cm and muac_cm else "Cartesian Landmark Plot",
                      fontsize=10.5, fontweight='bold')
    ax_sbs2.set_xlabel('X (cm) -> right', fontsize=9)
    ax_sbs2.set_ylabel('Y (cm) -> up', fontsize=9)
    ax_sbs2.legend(loc='upper left', fontsize=7.5, framealpha=0.9)
    ax_sbs2.grid(True, linestyle=':', alpha=0.6)
    ax_sbs2.set_aspect('equal')
    plt.tight_layout()

    side_by_side_b64 = fig_to_base64(fig_sbs)

    return {
        "status": "success",
        "arm_side": arm_side,
        "px_per_cm": round(float(px_per_cm), 2),
        "total_arm_cm": float(total_arm_cm) if total_arm_cm else None,
        "upper_arm_cm": float(upper_arm_cm) if upper_arm_cm else None,
        "forearm_cm": float(forearm_cm) if forearm_cm else None,
        "muac_cm": float(muac_cm) if muac_cm else None,
        "landmarks": landmark_data,
        "raw_coordinates": raw_coords_dict,
        "contour_overlay_base64": contour_overlay_b64,
        "cartesian_plot_base64": cartesian_plot_b64,
        "side_by_side_base64": side_by_side_b64
    }

