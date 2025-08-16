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

# --- Model Loading ---
# Load the YOLOv8 model. It's loaded once when the module is imported for efficiency.
# We use yolov8n (nano) for speed, which is great for a hackathon prototype.
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
    """Runs the YOLOv8 model on a single image and returns detections."""
    if not model:
        raise RuntimeError("YOLOv8 model is not available.")
    
    # Run inference
    results = model.predict(image_path, conf=0.25, verbose=False)
    
    # Process results
    detections = []
    if results:
        result = results[0]  # Get results for the first image
        if result.boxes:
            for box in result.boxes:
                class_id = int(box.cls)
                class_name = model.names[class_id]
                
                # For this project, we are primarily interested in billboards.
                # The default YOLO model might not have a 'billboard' class.
                # We'll treat the most confident detection as our target object for now.
                # In a real scenario, this would be filtered for specific class names.
                
                detections.append({
                    "bbox": [round(coord) for coord in box.xyxy[0].tolist()],
                    "class": class_name,
                    "conf": round(float(box.conf), 4)
                })
    
    # Return the most confident detection, if any
    if not detections:
        return None
    
    return max(detections, key=lambda x: x['conf'])

def process_report(report_id, storage_key=None):
    """The main background task to process a single report."""
    bucket, object_key = storage_key.split("/", 1)
    conn = _pg_conn()
    cur = conn.cursor()
    
    try:
        # 1. Mark report as 'processing'
        cur.execute("UPDATE reports SET status='processing' WHERE id=%s", (report_id,))
        conn.commit()

        # 2. Download image to a temporary file
        with tempfile.NamedTemporaryFile(delete=True, suffix=".jpg") as tmp:
            download_from_storage(bucket, object_key, tmp.name)

            # 3. Run detection model
            start_time = time.time()
            detection_result = run_detection(tmp.name)
            inference_time = time.time() - start_time
            
            audit_payload = {
                "source": "yolo_v8n",
                "inference_time_s": round(inference_time, 2),
            }

            if not detection_result:
                # No detection found, mark as processed with a note
                cur.execute("UPDATE reports SET status='processed', verdict=%s WHERE id=%s", 
                            (json.dumps({"violation": False, "reasons": ["no_billboard_detected"]}), report_id))
                audit_payload["note"] = "No objects detected by the model."
            else:
                # 4. A detection was found, save it
                # Placeholder for size estimation (Day 4)
                detection_result["size_m"] = {"w": None, "h": None, "conf": 0.0}
                
                cur.execute(
                    "INSERT INTO detections(report_id, bbox, class, conf, size_m) VALUES(%s, %s, %s, %s, %s)",
                    (report_id, json.dumps(detection_result["bbox"]), detection_result["class"], detection_result["conf"], json.dumps(detection_result["size_m"]))
                )
                
                # 5. Update report with a temporary verdict (to be refined on Day 5)
                # For now, any detection is marked as a potential violation for review.
                verdict = {"violation": True, "reasons": ["requires_manual_review"], "size_m": detection_result["size_m"]}
                cur.execute("UPDATE reports SET verdict=%s, status='processed' WHERE id=%s", (json.dumps(verdict), report_id))
                
                audit_payload["detection"] = detection_result

        # 6. Write audit log
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
        # Re-raise the exception to notify the worker (e.g., RQ) of the failure
        raise
    finally:
        cur.close()
        conn.close()

if __name__ == '__main__':
    # Example for local testing.
    # Ensure you have a report in your DB and a corresponding image in Supabase storage.
    # Example: process_report('your-report-uuid-here', 'reports/your-image-key.jpg')
    pass
