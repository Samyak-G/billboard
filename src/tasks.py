# src/tasks.py
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import os
import time
import tempfile
from supabase import create_client
import psycopg2
import json
from ultralytics import YOLO
import torch
from PIL import Image
from PIL.ExifTags import TAGS

# --- Constants ---
CAR_AVG_WIDTH_M = 1.8 
REFERENCE_CLASSES = {'car', 'bus', 'truck'}
PRIMARY_TARGET_CLASS = 'billboard' # Assuming a custom-trained model, but will have fallbacks.

# --- Model Loading ---
print("Initializing YOLOv8 model...")
MODEL_PATH = 'yolov8n.pt'
try:
    model = YOLO(MODEL_PATH)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f"✅ YOLOv8 model '{MODEL_PATH}' loaded successfully on '{device}'.")
except Exception as e:
    print(f"🚨 Failed to load YOLOv8 model: {e}")
    model = None

# --- Database and Storage Clients ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

def _pg_conn():
    """Establishes a new PostgreSQL connection."""
    return psycopg2.connect(DATABASE_URL)

def download_from_storage(bucket, object_key, local_path):
    """Downloads a file from Supabase storage."""
    with open(local_path, "wb+") as f:
        res = supabase.storage.from_(bucket).download(object_key)
        f.write(res)

def run_detection(image_path):
    """Runs YOLOv8 model, returning all detections."""
    if not model:
        raise RuntimeError("YOLOv8 model is not available.")
    results = model.predict(image_path, conf=0.25, verbose=False)
    
    detections = []
    if results and (result := results[0]):
        if result.boxes:
            for box in result.boxes:
                detections.append({
                    "bbox": [round(coord) for coord in box.xyxy[0].tolist()],
                    "class": model.names[int(box.cls)],
                    "conf": round(float(box.conf), 4)
                })
    return detections

def estimate_size_from_references(detections, primary_target_bbox):
    """Estimates physical size using reference objects (e.g., cars)."""
    reference_detections = [d for d in detections if d['class'] in REFERENCE_CLASSES]
    if not reference_detections:
        return None

    best_reference = max(reference_detections, key=lambda x: x['conf'])
    ref_bbox = best_reference['bbox']
    ref_pixel_width = ref_bbox[2] - ref_bbox[0]
    
    target_pixel_width = primary_target_bbox[2] - primary_target_bbox[0]
    target_pixel_height = primary_target_bbox[3] - primary_target_bbox[1]

    if ref_pixel_width == 0: return None

    pixels_per_meter = ref_pixel_width / CAR_AVG_WIDTH_M
    estimated_width_m = target_pixel_width / pixels_per_meter
    estimated_height_m = target_pixel_height / pixels_per_meter
    
    return {
        "w": round(estimated_width_m, 2), "h": round(estimated_height_m, 2),
        "conf": round(best_reference['conf'] * 0.75, 2), "method": "reference_object"
    }

def estimate_size_from_exif(image_path, target_bbox):
    """Estimates object size using image EXIF data as a fallback."""
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()
        if not exif_data: return None

        focal_length_35mm = next((v for k, v in exif_data.items() if TAGS.get(k) == 'FocalLengthIn35mmFilm'), None)
        if not focal_length_35mm: return None
        
        SENSOR_HEIGHT_MM = 24
        img_height_px = image.height
        obj_height_px = target_bbox[3] - target_bbox[1]
        ASSUMED_DISTANCE_M = 15.0

        object_height_m = (ASSUMED_DISTANCE_M * obj_height_px) / (focal_length_35mm * (img_height_px / SENSOR_HEIGHT_MM))
        if object_height_m <= 0: return None

        aspect_ratio = (target_bbox[2] - target_bbox[0]) / obj_height_px
        object_width_m = object_height_m * aspect_ratio

        return {"w": round(object_width_m, 2), "h": round(object_height_m, 2), "conf": 0.2, "method": "exif_approximate"}
    except Exception:
        return None

def run_compliance_checks(cur, report_id, estimated_size):
    """Runs geospatial and permit checks against the database."""
    violations = []
    
    # Fetch report geometry
    cur.execute("SELECT geom FROM reports WHERE id = %s", (report_id,))
    report_geom = cur.fetchone()[0]
    if not report_geom:
        return {"violation": True, "reasons": ["missing_geolocation"]}

    # 1. Zone Compliance Check
    cur.execute("""
        SELECT name, rules FROM zones WHERE ST_Contains(geom, ST_GeomFromText(%s, 4326))
    """, (report_geom,))
    zone_result = cur.fetchone()

    if not zone_result:
        violations.append({"code": "outside_zoning", "message": "Billboard is not in a designated commercial or residential zone."})
        zone_rules = {}
    else:
        zone_name, zone_rules = zone_result
        if not zone_rules.get("allows_billboards", False):
            violations.append({"code": "prohibited_zone", "message": f"Billboards are prohibited in '{zone_name}'."})
        
        # Size check (only if size was estimated)
        if estimated_size and estimated_size.get('w') is not None:
            max_w = zone_rules.get("max_width_m")
            max_h = zone_rules.get("max_height_m")
            if max_w and estimated_size['w'] > max_w:
                violations.append({"code": "size_exceeded_width", "message": f"Exceeds max width of {max_w}m in zone '{zone_name}'. Estimated: {estimated_size['w']}m."})
            if max_h and estimated_size['h'] > max_h:
                violations.append({"code": "size_exceeded_height", "message": f"Exceeds max height of {max_h}m in zone '{zone_name}'. Estimated: {estimated_size['h']}m."})

    # 2. Permit Check
    cur.execute("""
        SELECT id, valid_to FROM permits 
        WHERE ST_DWithin(geom, ST_GeomFromText(%s, 4326)::geography, 50)
        ORDER BY ST_Distance(geom, ST_GeomFromText(%s, 4326)::geography)
        LIMIT 1;
    """, (report_geom, report_geom))
    permit_result = cur.fetchone()

    if not permit_result:
        violations.append({"code": "no_permit_found", "message": "No valid permit found within 50 meters of this location."})
    
    return {
        "violation": len(violations) > 0,
        "reasons": violations,
        "size_m": estimated_size,
        "zone_assessed": zone_result[0] if zone_result else "N/A"
    }

def process_report(report_id, storage_key=None):
    """Main background task to process a single report."""
    bucket, object_key = storage_key.split("/", 1)
    conn = _pg_conn()
    cur = conn.cursor()
    
    try:
        cur.execute("UPDATE reports SET status='processing' WHERE id=%s", (report_id,))
        conn.commit()

        with tempfile.NamedTemporaryFile(delete=True, suffix=".jpg") as tmp:
            download_from_storage(bucket, object_key, tmp.name)
            
            start_time = time.time()
            all_detections = run_detection(tmp.name)
            inference_time = time.time() - start_time
            
            # Find the primary target (a billboard)
            # Simple approach: find largest object not in reference list, or most confident.
            non_ref_detections = [d for d in all_detections if d['class'] not in REFERENCE_CLASSES]
            primary_detection = max(non_ref_detections, key=lambda x: (x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1])) if non_ref_detections else None

            audit_payload = {"inference_time_s": round(inference_time, 2), "detections_found": len(all_detections)}

            if not primary_detection:
                verdict = {"violation": False, "reasons": ["no_billboard_detected"]}
                cur.execute("UPDATE reports SET status='processed', verdict=%s WHERE id=%s", (json.dumps(verdict), report_id))
            else:
                # --- Size Estimation (Day 4) ---
                size_estimate = estimate_size_from_references(all_detections, primary_detection['bbox'])
                if not size_estimate:
                    size_estimate = estimate_size_from_exif(tmp.name, primary_detection['bbox'])
                primary_detection["size_m"] = size_estimate if size_estimate else {"w": None, "h": None, "conf": 0.0, "method": "none"}
                
                # --- Compliance Checks (Day 5) ---
                verdict = run_compliance_checks(cur, report_id, primary_detection["size_m"])
                
                # Save all findings
                cur.execute(
                    "INSERT INTO detections(report_id, bbox, class, conf, size_m) VALUES(%s, %s, %s, %s, %s)",
                    (report_id, json.dumps(primary_detection["bbox"]), primary_detection["class"], primary_detection["conf"], json.dumps(primary_detection["size_m"]))
                )
                cur.execute("UPDATE reports SET verdict=%s, status='processed' WHERE id=%s", (json.dumps(verdict), report_id))
                audit_payload["primary_detection"] = primary_detection
                audit_payload["final_verdict"] = verdict

        cur.execute(
            "INSERT INTO audit_logs(report_id, event_type, payload) VALUES(%s, %s, %s)",
            (report_id, "processing_complete", json.dumps(audit_payload))
        )
        conn.commit()

    except Exception as e:
        conn.rollback()
        cur.execute("UPDATE reports SET status='error' WHERE id=%s", (report_id,))
        cur.execute("INSERT INTO audit_logs(report_id, event_type, payload) VALUES(%s, %s, %s)",
                    (report_id, "processing_error", json.dumps({"error": str(e)})))
        conn.commit()
        raise
    finally:
        cur.close()
        conn.close()
