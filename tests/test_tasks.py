# tests/test_tasks.py
import os
import time
import requests
import pytest

# --- Model Download Pre-flight Check ---
MODEL_NAME = 'yolov8n.pt'
MODEL_URL = f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{MODEL_NAME}"

def ensure_model_is_local():
    """
    Checks if the YOLO model is present locally. If not, downloads it 
    with a detailed progress bar.
    """
    if os.path.exists(MODEL_NAME):
        print(f"\n✅ Model '{MODEL_NAME}' already exists locally.")
        return

    print(f"\n⏳ Model '{MODEL_NAME}' not found. Starting download from {MODEL_URL}...")
    try:
        with requests.get(MODEL_URL, stream=True) as r:
            r.raise_for_status()
            total_size_in_bytes = int(r.headers.get('content-length', 0))
            total_size_in_mb = total_size_in_bytes / (1024 * 1024)
            block_size = 8192
            downloaded_bytes = 0
            start_time = time.time()

            with open(MODEL_NAME, 'wb') as f:
                for chunk in r.iter_content(chunk_size=block_size):
                    f.write(chunk)
                    downloaded_bytes += len(chunk)
                    
                    # Calculate progress
                    progress = downloaded_bytes / total_size_in_bytes
                    percentage = progress * 100
                    downloaded_mb = downloaded_bytes / (1024 * 1024)
                    elapsed_time = time.time() - start_time
                    
                    # Print progress bar
                    print(
                        f"\r    Downloading: [{int(percentage)}%] "
                        f"{downloaded_mb:.2f}/{total_size_in_mb:.2f} MB | "
                        f"Elapsed: {elapsed_time:.1f}s",
                        end=''
                    )
            print("\n✅ Download complete.")

    except Exception as e:
        print(f"\n🚨 Failed to download model. Error: {e}")
        # If download fails, we should skip the tests that need the model.
        pytest.skip(f"Failed to download required model '{MODEL_NAME}'.", allow_module_level=True)

# Run the check when the test module is loaded
ensure_model_is_local()

# Now that the model is guaranteed to be local (or tests are skipped),
# we can safely import the tasks module which loads the model.
from src.tasks import run_detection, model

# --- Pytest Fixtures and Tests ---

# Get the absolute path to the test image
# This makes the test runnable from any directory
TEST_IMAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'samyak.jpeg'))

@pytest.mark.skipif(not os.path.exists(TEST_IMAGE_PATH), reason="Test image not found")
@pytest.mark.skipif(model is None, reason="YOLO model failed to load or was not downloaded")
def test_run_detection_returns_valid_format():
    """
    Tests the run_detection function to ensure it returns the correct data format.
    It does not check for detection accuracy, only for format correctness.
    """
    # Ensure the test image exists before running
    assert os.path.exists(TEST_IMAGE_PATH), f"Test image not found at {TEST_IMAGE_PATH}"

    # Run the detection function
    detection_result = run_detection(TEST_IMAGE_PATH)

    # The result can be None if no object is detected with sufficient confidence
    if detection_result is not None:
        # If a detection is found, validate its structure
        assert isinstance(detection_result, dict)
        
        # Check for required keys that run_detection is responsible for.
        # The 'size_m' key is added later in the pipeline.
        expected_keys = ["bbox", "class", "conf"]
        assert all(key in detection_result for key in expected_keys)

        # Check bounding box format
        bbox = detection_result['bbox']
        assert isinstance(bbox, list)
        assert len(bbox) == 4
        assert all(isinstance(coord, int) for coord in bbox)

        # Check other field types
        assert isinstance(detection_result['class'], str)
        assert isinstance(detection_result['conf'], float)

    else:
        # If no detection, the test passes as this is a valid outcome
        print("\nNote: No objects detected in the test image, which is a valid result.")
        pass
