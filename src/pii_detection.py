"""
PII Detection and Redaction Module
Day 6 - Face and license plate detection with blurring
"""

import cv2
import numpy as np
import asyncpg
import json
import os
import uuid
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

# Environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

# Initialize Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# Initialize face detector (OpenCV Haar Cascade)
face_cascade = None
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    print("✅ Face detector initialized")
except Exception as e:
    print(f"⚠️ Face detector initialization failed: {e}")

def detect_faces(image: np.ndarray, min_size: Tuple[int, int] = (30, 30)) -> List[Dict]:
    """
    Detect faces in image using OpenCV Haar Cascade
    
    Args:
        image: OpenCV image (BGR format)
        min_size: Minimum face size to detect
        
    Returns:
        list: List of face detections with bbox and confidence
    """
    faces = []
    
    try:
        if face_cascade is None:
            return faces
            
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with stricter parameters to reduce false positives
        detected_faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.3,        # Larger scale factor (less sensitive)
            minNeighbors=6,         # More neighbors required (reduces false positives)
            minSize=(50, 50),       # Larger minimum size
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        for (x, y, w, h) in detected_faces:
            # Additional validation: check aspect ratio (faces should be roughly square)
            aspect_ratio = w / h
            if 0.7 <= aspect_ratio <= 1.3:  # Reject extreme rectangles
                faces.append({
                    "bbox": [int(x), int(y), int(x+w), int(y+h)],
                    "confidence": 0.8,  # Haar cascade doesn't provide confidence, use default
                    "type": "face"
                })
            
    except Exception as e:
        print(f"Face detection failed: {e}")
    
    return faces

def detect_license_plates(image: np.ndarray) -> List[Dict]:
    """
    Simple license plate detection using contours and aspect ratio
    (This is a basic implementation - in production, use specialized models)
    
    Args:
        image: OpenCV image (BGR format)
        
    Returns:
        list: List of potential license plate detections
    """
    plates = []
    
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Find edges
        edges = cv2.Canny(filtered, 30, 200)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours that could be license plates
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Skip small contours
            if area < 1000:
                continue
                
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check aspect ratio (license plates are typically wider than tall)
            aspect_ratio = w / h
            
            # License plates typically have aspect ratio between 2:1 and 5:1
            if 2.0 <= aspect_ratio <= 5.0:
                # Check if contour approximation has 4 corners (rectangle-like)
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                
                if len(approx) >= 4:
                    plates.append({
                        "bbox": [int(x), int(y), int(x+w), int(y+h)],
                        "confidence": 0.6,  # Basic confidence score
                        "type": "license_plate"
                    })
                    
    except Exception as e:
        print(f"License plate detection failed: {e}")
    
    return plates

def blur_region(image: np.ndarray, bbox: List[int], blur_strength: int = 51) -> np.ndarray:
    """
    Blur a specific region in the image
    
    Args:
        image: OpenCV image (BGR format)
        bbox: Bounding box [x1, y1, x2, y2]
        blur_strength: Gaussian blur kernel size (must be odd)
        
    Returns:
        np.ndarray: Image with blurred region
    """
    try:
        # Ensure blur strength is odd
        if blur_strength % 2 == 0:
            blur_strength += 1
            
        # Create a copy of the image
        blurred_image = image.copy()
        
        # Extract coordinates
        x1, y1, x2, y2 = bbox
        
        # Ensure coordinates are within image bounds
        h, w = image.shape[:2]
        x1 = max(0, min(x1, w))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h))
        y2 = max(0, min(y2, h))
        
        # Extract region of interest
        roi = blurred_image[y1:y2, x1:x2]
        
        if roi.size > 0:
            # Apply Gaussian blur
            blurred_roi = cv2.GaussianBlur(roi, (blur_strength, blur_strength), 0)
            
            # Replace the region in the original image
            blurred_image[y1:y2, x1:x2] = blurred_roi
            
    except Exception as e:
        print(f"Region blurring failed: {e}")
        return image
    
    return blurred_image

def detect_all_pii(image: np.ndarray) -> List[Dict]:
    """
    Detect all types of PII in image
    
    Args:
        image: OpenCV image (BGR format)
        
    Returns:
        list: List of all PII detections
    """
    pii_detections = []
    
    # Detect faces
    faces = detect_faces(image)
    pii_detections.extend(faces)
    
    # Detect license plates
    plates = detect_license_plates(image)
    pii_detections.extend(plates)
    
    return pii_detections

def create_redacted_image(image: np.ndarray, pii_detections: List[Dict]) -> np.ndarray:
    """
    Create redacted version of image with all PII blurred
    
    Args:
        image: Original image
        pii_detections: List of PII detections
        
    Returns:
        np.ndarray: Redacted image
    """
    redacted = image.copy()
    
    for detection in pii_detections:
        bbox = detection["bbox"]
        redacted = blur_region(redacted, bbox)
    
    return redacted

async def upload_redacted_image(redacted_image: np.ndarray, report_id: str) -> str:
    """
    Upload redacted image to Supabase Storage
    
    Args:
        redacted_image: Redacted image
        report_id: Report UUID
        
    Returns:
        str: Storage path of uploaded image
    """
    try:
        # Generate unique filename
        filename = f"redacted/{report_id}_{uuid.uuid4()}.jpg"
        
        # Encode image as JPEG
        _, buffer = cv2.imencode('.jpg', redacted_image)
        image_bytes = buffer.tobytes()
        
        # Upload to Supabase storage
        response = supabase.storage.from_("reports").upload(
            filename, 
            image_bytes,
            file_options={"content-type": "image/jpeg"}
        )
        
        if response.error:
            print(f"Storage upload failed: {response.error}")
            return None
            
        return filename
        
    except Exception as e:
        print(f"Failed to upload redacted image: {e}")
        return None

async def save_pii_artifacts(report_id: str, pii_detections: List[Dict], redacted_url: str = None):
    """
    Save PII artifacts to database
    
    Args:
        report_id: Report UUID
        pii_detections: List of PII detections
        redacted_url: URL of redacted image
    """
    try:
        if not pii_detections:
            return
            
        conn = await asyncpg.connect(DATABASE_URL)
        
        for detection in pii_detections:
            await conn.execute("""
                INSERT INTO pii_artifacts (report_id, artifact_type, bbox, redacted_url, confidence)
                VALUES ($1, $2, $3, $4, $5)
            """, 
            report_id, 
            detection["type"], 
            json.dumps(detection["bbox"]), 
            redacted_url,
            detection["confidence"]
            )
        
        await conn.close()
        
    except Exception as e:
        print(f"Failed to save PII artifacts: {e}")

async def process_pii_redaction(image: np.ndarray, report_id: str) -> Dict:
    """
    Complete PII detection and redaction pipeline
    
    Args:
        image: Original image
        report_id: Report UUID
        
    Returns:
        dict: Processing results
    """
    try:
        # Detect PII
        pii_detections = detect_all_pii(image)
        
        redacted_url = None
        
        # Create redacted image if PII found
        if pii_detections:
            redacted_image = create_redacted_image(image, pii_detections)
            redacted_url = await upload_redacted_image(redacted_image, report_id)
        
        # Save PII artifacts to database
        await save_pii_artifacts(report_id, pii_detections, redacted_url)
        
        return {
            "report_id": report_id,
            "pii_count": len(pii_detections),
            "pii_types": list(set([d["type"] for d in pii_detections])),
            "redacted_url": redacted_url,
            "status": "completed"
        }
        
    except Exception as e:
        print(f"PII processing failed: {e}")
        return {
            "report_id": report_id,
            "status": "failed",
            "error": str(e)
        }

# Test function
async def test_pii_detection():
    """Test PII detection pipeline"""
    print("🧪 Testing PII Detection Pipeline...")
    
    # Test with the sample image
    test_image_path = "tests/samyak.jpeg"
    
    if os.path.exists(test_image_path):
        # Load test image
        image = cv2.imread(test_image_path)
        
        if image is not None:
            print(f"📸 Testing with image: {test_image_path}")
            
            # Detect PII
            pii_detections = detect_all_pii(image)
            
            print(f"🔍 Found {len(pii_detections)} PII artifacts:")
            for detection in pii_detections:
                print(f"  - {detection['type']}: bbox={detection['bbox']}, conf={detection['confidence']:.2f}")
            
            # Test redaction
            if pii_detections:
                redacted = create_redacted_image(image, pii_detections)
                print("✅ Redacted image created successfully")
            else:
                print("ℹ️ No PII found - no redaction needed")
                
        else:
            print(f"❌ Could not load test image: {test_image_path}")
    else:
        print(f"❌ Test image not found: {test_image_path}")
    
    print("✅ PII detection test completed!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_pii_detection())
