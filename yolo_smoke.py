from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Print the COCO class mapping once
print(model.names)

# Run inference on your video
results = model("raw_videos/firefly-multi1.mp4")

# Loop through detections and print class names + confidence
for r in results:
    for box in r.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        cls_name = model.names[cls_id]
        print(f"Detected {cls_name} with confidence {conf:.2f}")

        