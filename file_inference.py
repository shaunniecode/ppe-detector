# file_inference.py
from ultralytics import YOLO

# Load your trained checkpoint (best.pt from your last training run)
model = YOLO("runs/detect/train6/weights/best.pt")

# Path to an unseen image
IMAGE_PATH = r"images\sim2.png" 

# Run inference
results = model(IMAGE_PATH)

# Print detections in a readable form
for r in results:
    boxes = r.boxes
    names = model.names
    dets = []
    for b in boxes:
        cls_id = int(b.cls.item())
        conf = float(b.conf.item())
        label = names[cls_id]
        dets.append((label, round(conf, 3)))
    print("Detections:", dets)