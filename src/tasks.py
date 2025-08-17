# src/tasks.py
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import os
import time
import tempfile
import requests
from supabase import create_client
import psycopg2
import json
from ultralytics import YOLO
import torch
from PIL import Image
from PIL.ExifTags import TAGS

# --- Constants ---
# Average width of a car in meters, used as a reference object.
CAR_AVG_WIDTH_M = 1.8 
# Acceptable classes for reference objects
REFERENCE_CLASSES = {'car', 'bus', 'truck'}

# --- Model Loading ---
# Load the YOLOv8 model. It's loaded once when the module is imported for efficiency.
print("Initializing YOLOv8 model...")
MODEL_PATH = 'yolov8n.pt'
try:
    model = YOLO(MODEL_PATH)
    # Move model to GPU if available, otherwise it runs on CPU
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
    """Runs the YOLOv8 model on a single image and returns all detections."""
    if not model:
        raise RuntimeError("YOLOv8 model is not available.")
    
    results = model.predict(image_path, conf=0.25, verbose=False)
    
    detections = []
    if results and (result := results[0]):
        if result.boxes:
            for box in result.boxes:
                class_id = int(box.cls)
                class_name = model.names[class_id]
                
                detections.append({
                    "bbox": [round(coord) for coord in box.xyxy[0].tolist()],
                    "class": class_name,
                    "conf": round(float(box.conf), 4)
                })
    
    return detections

def estimate_size_from_references(detections, primary_target_bbox):
    """
    Estimates the physical size of a target object using reference objects (e.g., cars)
    in the same image.
    """
    reference_detections = [d for d in detections if d['class'] in REFERENCE_CLASSES]
    if not reference_detections:
        return None

    # Find the best reference object (e.g., highest confidence car)
    best_reference = max(reference_detections, key=lambda x: x['conf'])
    
    # Get pixel dimensions
    ref_bbox = best_reference['bbox']
    ref_pixel_width = ref_bbox[2] - ref_bbox[0]
    
    target_pixel_width = primary_target_bbox[2] - primary_target_bbox[0]
    target_pixel_height = primary_target_bbox[3] - primary_target_bbox[1]

    if ref_pixel_width == 0:
        return None

    # Calculate pixels-per-meter ratio based on the reference object
    # This is a strong assumption: that the objects are at a similar depth
    pixels_per_meter = ref_pixel_width / CAR_AVG_WIDTH_M
    
    # Estimate physical size of the target
    estimated_width_m = target_pixel_width / pixels_per_meter
    estimated_height_m = target_pixel_height / pixels_per_meter
    
    # Confidence is based on the reference object's confidence.
    # Could be improved by also considering object sizes, overlap, etc.
    estimation_confidence = best_reference['conf'] * 0.75 # Heuristic penalty

    return {
        "w": round(estimated_width_m, 2),
        "h": round(estimated_height_m, 2),
        "conf": round(estimation_confidence, 2),
        "method": "reference_object"
    }

def estimate_size_from_exif(image_path, target_bbox):
    """
    Estimates object size using image EXIF data as a fallback.
    This method is highly approximate and relies on strong assumptions.
    """
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()
        if not exif_data:
            return None

        # Extract focal length, preferring the 35mm equivalent
        focal_length_35mm = None
        focal_length_mm = None
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            if tag == 'FocalLengthIn35mmFilm':
                focal_length_35mm = float(value)
            elif tag == 'FocalLength':
                focal_length_mm = float(value)

        if not focal_length_35mm and not focal_length_mm:
            return None
        
        # Assume a standard full-frame sensor height (24mm) for distance calculation
        # if we only have the native focal length.
        # The 35mm equivalent focal length already accounts for sensor size.
        SENSOR_HEIGHT_MM = 24
        F_eff_mm = focal_length_35mm if focal_length_35mm else focal_length_mm

        # Get image and object dimensions in pixels
        img_height_px = image.height
        obj_height_px = target_bbox[3] - target_bbox[1]

        # VERY ROUGH distance estimation assuming camera is held at 1.6m
        # and the bottom of the billboard is ~3m off the ground.
        # This is a major simplification for hackathon purposes.
        ASSUMED_DISTANCE_M = 15.0

        # Standard formula: object_height_m = (distance_m * object_height_px * sensor_height_mm) / (focal_length_mm * image_height_px)
        # Simplified with 35mm equivalent focal length:
        object_height_m = (ASSUMED_DISTANCE_M * obj_height_px) / (F_eff_mm * (img_height_px / SENSOR_HEIGHT_MM))

        if object_height_m <= 0:
            return None

        # Estimate width based on aspect ratio
        obj_width_px = target_bbox[2] - target_bbox[0]
        aspect_ratio = obj_width_px / obj_height_px
        object_width_m = object_height_m * aspect_ratio

        return {
            "w": round(object_width_m, 2),
            "h": round(object_height_m, 2),
            "conf": 0.2,  # Hardcoded low confidence
            "method": "exif_approximate"
        }

    except Exception:
        # Pillow can fail on images with no/corrupt EXIF data
        return None


def process_report(report_id, storage_key=None):
    """The main background task to process a single report."""
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
            
            # For now, assume the most confident non-reference object is the primary target billboard
            # This logic can be refined to find the largest non-reference object.
            non_ref_detections = [d for d in all_detections if d['class'] not in REFERENCE_CLASSES]
            primary_detection = max(non_ref_detections, key=lambda x: x['conf']) if non_ref_detections else None

            audit_payload = {
                "source": "yolo_v8n",
                "inference_time_s": round(inference_time, 2),
                "detections_found": len(all_detections)
            }

            if not primary_detection:
                cur.execute("UPDATE reports SET status='processed', verdict=%s WHERE id=%s", 
                            (json.dumps({"violation": False, "reasons": ["no_billboard_detected"]}), report_id))
                audit_payload["note"] = "No primary target detected by the model."
            else:
                # --- Size Estimation ---
                size_estimate = estimate_size_from_references(all_detections, primary_detection['bbox'])
                
                # If reference method fails, try EXIF-based fallback
                if not size_estimate:
                    size_estimate = estimate_size_from_exif(tmp.name, primary_detection['bbox'])

                primary_detection["size_m"] = size_estimate if size_estimate else {"w": None, "h": None, "conf": 0.0, "method": "none"}
                
                cur.execute(
                    "INSERT INTO detections(report_id, bbox, class, conf, size_m) VALUES(%s, %s, %s, %s, %s)",
                    (report_id, json.dumps(primary_detection["bbox"]), primary_detection["class"], primary_detection["conf"], json.dumps(primary_detection["size_m"]))
                )
                
                verdict = {"violation": True, "reasons": ["requires_manual_review"], "size_m": primary_detection["size_m"]}
                cur.execute("UPDATE reports SET verdict=%s, status='processed' WHERE id=%s", (json.dumps(verdict), report_id))
                
                audit_payload["primary_detection"] = primary_detection

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

if __name__ == '__main__':
    # Example for local testing.
    pass
