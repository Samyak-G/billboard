"""
OCR and Content Moderation Module for Billboard Detection
Day 6 - Content validation pipeline
"""

import cv2
import numpy as np
import pytesseract
import easyocr
import re
import asyncpg
import json
from typing import List, Dict, Optional, Tuple
from PIL import Image
import os
from dotenv import load_dotenv

load_dotenv()

# Constants
PROFANITY_THRESHOLD = 0.6
DATABASE_URL = os.getenv("DATABASE_URL")

# Initialize EasyOCR reader (fallback for complex text)
try:
    easyocr_reader = easyocr.Reader(['en'], gpu=False)
    print("✅ EasyOCR initialized successfully")
except Exception as e:
    print(f"⚠️ EasyOCR initialization failed: {e}")
    easyocr_reader = None

# Custom profanity word list (simple implementation)
PROFANITY_WORDS = {
    'inappropriate', 'offensive', 'adult', 'explicit', 'profane',
    # Add more words as needed
}

# Political/misinformation keywords
POLITICAL_KEYWORDS = {
    'election', 'vote', 'candidate', 'poll', 'ballot', 'fake news',
    'conspiracy', 'scam', 'fraud', 'hoax'
}

# Sexual content keywords
SEXUAL_KEYWORDS = {
    'adult', 'explicit', 'mature', 'sexual', 'xxx'
}

def run_tesseract_ocr(image_crop: np.ndarray, config: str = '--psm 6') -> Tuple[str, float]:
    """
    Run Tesseract OCR on image crop
    
    Args:
        image_crop: OpenCV image (BGR format)
        config: Tesseract configuration
        
    Returns:
        tuple: (extracted_text, confidence)
    """
    try:
        # Convert to grayscale for better OCR
        if len(image_crop.shape) == 3:
            gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_crop
            
        # Apply some preprocessing for better OCR
        # Increase contrast
        alpha = 1.5  # Contrast control
        beta = 0     # Brightness control
        enhanced = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
        
        # Denoise
        denoised = cv2.medianBlur(enhanced, 3)
        
        # Extract text
        text = pytesseract.image_to_string(denoised, config=config).strip()
        
        # Get confidence (rough estimate)
        data = pytesseract.image_to_data(denoised, output_type=pytesseract.Output.DICT)
        confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return text, avg_confidence / 100.0  # Convert to 0-1 scale
        
    except Exception as e:
        print(f"Tesseract OCR failed: {e}")
        return "", 0.0

def run_easyocr_ocr(image_crop: np.ndarray) -> Tuple[str, float]:
    """
    Run EasyOCR on image crop (fallback)
    
    Args:
        image_crop: OpenCV image (BGR format)
        
    Returns:
        tuple: (extracted_text, confidence)
    """
    try:
        if easyocr_reader is None:
            return "", 0.0
            
        # EasyOCR expects RGB format
        if len(image_crop.shape) == 3:
            rgb_image = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = cv2.cvtColor(image_crop, cv2.COLOR_GRAY2RGB)
            
        # Run EasyOCR
        results = easyocr_reader.readtext(rgb_image)
        
        # Combine all detected text
        texts = []
        confidences = []
        for (bbox, text, confidence) in results:
            if confidence > 0.3:  # Filter low confidence detections
                texts.append(text)
                confidences.append(confidence)
        
        combined_text = " ".join(texts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return combined_text, avg_confidence
        
    except Exception as e:
        print(f"EasyOCR failed: {e}")
        return "", 0.0

def extract_text_from_detection(image: np.ndarray, bbox: List[float], use_easyocr: bool = False) -> Tuple[str, float, str]:
    """
    Extract text from a detection bounding box
    
    Args:
        image: Full image (OpenCV format)
        bbox: Bounding box [x1, y1, x2, y2]
        use_easyocr: Whether to use EasyOCR instead of Tesseract
        
    Returns:
        tuple: (text, confidence, ocr_engine)
    """
    try:
        # Extract crop from bounding box
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        crop = image[y1:y2, x1:x2]
        
        if crop.size == 0:
            return "", 0.0, "none"
        
        # Choose OCR engine
        if use_easyocr and easyocr_reader is not None:
            text, confidence = run_easyocr_ocr(crop)
            engine = "easyocr"
        else:
            text, confidence = run_tesseract_ocr(crop)
            engine = "tesseract"
            
        # Fallback to EasyOCR if Tesseract confidence is low
        if confidence < 0.3 and not use_easyocr and easyocr_reader is not None:
            fallback_text, fallback_conf = run_easyocr_ocr(crop)
            if fallback_conf > confidence:
                text, confidence, engine = fallback_text, fallback_conf, "easyocr"
        
        return text, confidence, engine
        
    except Exception as e:
        print(f"Text extraction failed: {e}")
        return "", 0.0, "error"

def check_profanity(text: str) -> Dict:
    """
    Simple profanity detection
    
    Args:
        text: Text to check
        
    Returns:
        dict: Flag details if profanity found
    """
    if not text:
        return None
        
    text_lower = text.lower()
    found_words = []
    
    for word in PROFANITY_WORDS:
        if word in text_lower:
            found_words.append(word)
    
    if found_words:
        score = min(1.0, len(found_words) * 0.3)  # Simple scoring
        return {
            "flag_type": "profanity",
            "score": score,
            "details": {"found_words": found_words}
        }
    
    return None

def check_political_content(text: str) -> Dict:
    """
    Check for political/misinformation content
    
    Args:
        text: Text to check
        
    Returns:
        dict: Flag details if political content found
    """
    if not text:
        return None
        
    text_lower = text.lower()
    found_keywords = []
    
    for keyword in POLITICAL_KEYWORDS:
        if keyword in text_lower:
            found_keywords.append(keyword)
    
    if found_keywords:
        score = min(1.0, len(found_keywords) * 0.4)
        return {
            "flag_type": "political_or_misinformation",
            "score": score,
            "details": {"found_keywords": found_keywords}
        }
    
    return None

def check_sexual_content(text: str) -> Dict:
    """
    Check for sexual/adult content
    
    Args:
        text: Text to check
        
    Returns:
        dict: Flag details if sexual content found
    """
    if not text:
        return None
        
    text_lower = text.lower()
    found_keywords = []
    
    for keyword in SEXUAL_KEYWORDS:
        if keyword in text_lower:
            found_keywords.append(keyword)
    
    if found_keywords:
        score = min(1.0, len(found_keywords) * 0.5)
        return {
            "flag_type": "sexual_content",
            "score": score,
            "details": {"found_keywords": found_keywords}
        }
    
    return None

def moderate_text(text: str) -> List[Dict]:
    """
    Run complete text moderation pipeline
    
    Args:
        text: Text to moderate
        
    Returns:
        list: List of flags found
    """
    flags = []
    
    # Check each moderation category
    moderation_checks = [
        check_profanity,
        check_political_content,
        check_sexual_content
    ]
    
    for check_func in moderation_checks:
        flag = check_func(text)
        if flag:
            flags.append(flag)
    
    return flags

async def save_ocr_result(detection_id: str, text: str, confidence: float, engine: str):
    """
    Save OCR result to database
    
    Args:
        detection_id: Detection UUID
        text: Extracted text
        confidence: OCR confidence
        engine: OCR engine used
    """
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        
        await conn.execute("""
            INSERT INTO ocr_texts (detection_id, text, confidence, ocr_engine)
            VALUES ($1, $2, $3, $4)
        """, detection_id, text, confidence, engine)
        
        await conn.close()
        
    except Exception as e:
        print(f"Failed to save OCR result: {e}")

async def save_content_flags(detection_id: str, flags: List[Dict]):
    """
    Save content moderation flags to database
    
    Args:
        detection_id: Detection UUID
        flags: List of moderation flags
    """
    try:
        if not flags:
            return
            
        conn = await asyncpg.connect(DATABASE_URL)
        
        for flag in flags:
            await conn.execute("""
                INSERT INTO content_flags (detection_id, flag_type, score, details)
                VALUES ($1, $2, $3, $4)
            """, detection_id, flag["flag_type"], flag["score"], json.dumps(flag["details"]))
        
        await conn.close()
        
    except Exception as e:
        print(f"Failed to save content flags: {e}")

async def process_detection_ocr_and_moderation(image: np.ndarray, detection: Dict) -> Dict:
    """
    Complete OCR and moderation pipeline for a single detection
    
    Args:
        image: Full image (OpenCV format)
        detection: Detection dictionary with bbox and id
        
    Returns:
        dict: Processing results
    """
    try:
        detection_id = detection['id']
        bbox = detection['bbox']
        
        # Extract text
        text, confidence, engine = extract_text_from_detection(image, bbox)
        
        # Run moderation
        flags = moderate_text(text)
        
        # Save to database
        await save_ocr_result(detection_id, text, confidence, engine)
        await save_content_flags(detection_id, flags)
        
        return {
            "detection_id": detection_id,
            "text": text,
            "confidence": confidence,
            "engine": engine,
            "flags": flags,
            "status": "completed"
        }
        
    except Exception as e:
        print(f"OCR and moderation processing failed: {e}")
        return {
            "detection_id": detection.get('id', 'unknown'),
            "status": "failed",
            "error": str(e)
        }

# Test function
async def test_ocr_moderation():
    """Test the OCR and moderation pipeline"""
    print("🧪 Testing OCR and Moderation Pipeline...")
    
    # Test text moderation
    test_texts = [
        "Fresh vegetables for sale",
        "Vote for candidate X - election 2024",
        "Adult content warning",
        "Quality products at best prices"
    ]
    
    for text in test_texts:
        flags = moderate_text(text)
        print(f"Text: '{text}' -> Flags: {len(flags)}")
        for flag in flags:
            print(f"  - {flag['flag_type']}: {flag['score']:.2f}")
    
    print("✅ OCR and moderation test completed!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_ocr_moderation())
