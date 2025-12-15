# run_inference_stub.py
# Minimal stub for MQTT simulation before wiring in PyTorch detector

def run_inference(img):
    """
    Fake inference function.
    Returns a hardcoded detection label for testing MQTT subscriber flow.
    """
    detections = [
        ("helmet", 0.94)  # (label, confidence)
    ]
    return detections