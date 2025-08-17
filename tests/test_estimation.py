# tests/test_estimation.py
import pytest
from src.tasks import estimate_size_from_references, estimate_size_from_exif
from unittest.mock import patch, MagicMock

# --- Test for Reference Object Method ---

def test_estimate_size_from_references_success():
    """ 
    Tests the reference-based estimation with ideal mock data.
    """
    # Mock detections: a billboard and a car at roughly the same scale
    primary_target_bbox = [100, 100, 500, 300] # 400px wide, 200px high
    detections = [
        {"bbox": primary_target_bbox, "class": "tv", "conf": 0.9},
        {"bbox": [150, 350, 350, 450], "class": "car", "conf": 0.88} # 200px wide
    ]

    # Calculation check:
    # Car is 200px wide, real width is 1.8m -> 111.11 pixels/meter
    # Billboard is 400px wide -> 400 / 111.11 = 3.6m
    # Billboard is 200px high -> 200 / 111.11 = 1.8m
    expected_w = 3.6
    expected_h = 1.8

    result = estimate_size_from_references(detections, primary_target_bbox)

    assert result is not None
    assert result['method'] == "reference_object"
    assert abs(result['w'] - expected_w) < 0.01
    assert abs(result['h'] - expected_h) < 0.01
    assert result['conf'] > 0.6

def test_estimate_size_from_references_no_reference():
    """
    Tests that the function returns None when no reference objects are detected.
    """
    primary_target_bbox = [100, 100, 500, 300]
    detections = [
        {"bbox": primary_target_bbox, "class": "person", "conf": 0.9},
        {"bbox": [600, 150, 700, 250], "class": "motorcycle", "conf": 0.8}
    ]
    result = estimate_size_from_references(detections, primary_target_bbox)
    assert result is None

# --- Test for EXIF Fallback Method ---

@patch('src.tasks.Image.open')
def test_estimate_size_from_exif_success(mock_image_open):
    """
    Tests the EXIF-based estimation by mocking the Pillow Image object
    and its EXIF data.
    """
    # Create a mock image object
    mock_image = MagicMock()
    mock_image.height = 4000 # Mock image height
    # Mock the _getexif method to return fake EXIF data
    # 37386 is the tag for FocalLength, 41989 for FocalLengthIn35mmFilm
    mock_image._getexif.return_value = {
        37386: 50.0, # FocalLength
        41989: 50.0  # FocalLengthIn35mmFilm
    }
    mock_image_open.return_value = mock_image

    target_bbox = [100, 100, 500, 300] # 400px wide, 200px high
    result = estimate_size_from_exif('fake_path.jpg', target_bbox)

    assert result is not None
    assert result['method'] == "exif_approximate"
    assert result['conf'] == 0.2
    assert 'w' in result and 'h' in result
    assert result['w'] > 0 and result['h'] > 0

@patch('src.tasks.Image.open')
def test_estimate_size_from_exif_no_data(mock_image_open):
    """
    Tests that the function returns None if the image has no EXIF data.
    """
    mock_image = MagicMock()
    mock_image._getexif.return_value = None # No EXIF data
    mock_image_open.return_value = mock_image

    result = estimate_size_from_exif('fake_path.jpg', [100, 100, 500, 300])
    assert result is None

@patch('src.tasks.Image.open')
def test_estimate_size_from_exif_no_focal_length(mock_image_open):
    """
    Tests that the function returns None if EXIF data lacks focal length.
    """
    mock_image = MagicMock()
    # 271 is the tag for 'Make' (e.g., camera manufacturer)
    mock_image._getexif.return_value = { 271: 'Sony' }
    mock_image_open.return_value = mock_image

    result = estimate_size_from_exif('fake_path.jpg', [100, 100, 500, 300])
    assert result is None
