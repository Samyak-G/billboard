# src/tasks.py
import os
import time
import tempfile
import requests
from supabase import create_client
import psycopg2
import json

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

def _pg_conn():
    return psycopg2.connect(DATABASE_URL)

def download_from_storage(bucket, object_key, local_path):
    # Use Supabase Storage REST endpoint if client download isn't available
    url = f"{SUPABASE_URL}/storage/v1/object/{bucket}/{object_key}"
    headers = {"Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}"}
    r = requests.get(url, headers=headers, stream=True)
    r.raise_for_status()
    with open(local_path, "wb") as f:
        for chunk in r.iter_content(1024*64):
            f.write(chunk)

def process_report(report_id, storage_key=None):
    # storage_key expected as 'reports/<object>'
    bucket, object_key = storage_key.split("/", 1)

    conn = _pg_conn()
    cur = conn.cursor()
    try:
        # mark processing
        cur.execute("update reports set status='processing' where id=%s", (report_id,))
        conn.commit()

        # download image to temp file
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        tmp.close()
        download_from_storage(bucket, object_key, tmp.name)

        # ---- STUB: run detection here ----
        # For now simulate work and produce fake bbox + size
        time.sleep(2)  # simulate inference
        fake_detection = {
            "bbox": [100, 50, 300, 200],
            "class": "billboard",
            "conf": 0.94,
            "size_m": {"w": 6.0, "h": 3.0, "conf": 0.6}
        }

        # insert detection
        cur.execute(
            "insert into detections(report_id, bbox, class, conf, size_m) values(%s, %s, %s, %s, %s)",
            (report_id, json.dumps(fake_detection["bbox"]), fake_detection["class"], fake_detection["conf"], json.dumps(fake_detection["size_m"]))
        )
        # write audit log
        cur.execute(
            "insert into audit_logs(report_id, event_type, payload) values(%s, %s, %s)",
            (report_id, "processing_complete", json.dumps({"note": "stub detection", "detection": fake_detection}))
        )
        # update report verdict & status
        verdict = {"violation": True, "reasons": ["exceeds_area"], "size_m": fake_detection["size_m"]}
        cur.execute("update reports set verdict = %s, status = %s where id=%s", (json.dumps(verdict), "processed", report_id))

        conn.commit()
    except Exception as e:
        conn.rollback()
        cur.execute("update reports set status='error' where id=%s", (report_id,))
        conn.commit()
        # write error log
        cur.execute("insert into audit_logs(report_id, event_type, payload) values(%s, %s, %s)", (report_id, "processing_error", json.dumps({"error": str(e)})))
        conn.commit()
        raise
    finally:
        cur.close()
        conn.close()
