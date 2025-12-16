# visualize_inference.py
from ultralytics import YOLO
import cv2

# Load your trained checkpoint
model = YOLO(r"runs\detect\train6\weights\best.pt")

# Path to an unseen image
IMAGE_PATH = r"images\sim2.png"
OUTPUT_PATH = r"annotated_output2.jpg"

# Run inference
results = model(IMAGE_PATH)

# Iterate through results
for r in results:
    boxes = r.boxes
    names = model.names

    # Load image for annotation
    img = cv2.imread(IMAGE_PATH)

    for b in boxes:
        cls_id = int(b.cls.item())
        conf = float(b.conf.item())
        label = names[cls_id]

        # Bounding box coordinates
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())

        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Put label + confidence
        text = f"{label} {conf:.2f}"
        cv2.putText(img, text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save annotated image
    cv2.imwrite(OUTPUT_PATH, img)
    print("Annotated image saved:", OUTPUT_PATH)

    # Show annotated image
    cv2.imshow("Inference Visualization", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Print detections in readable form
    dets = [(names[int(b.cls.item())], round(float(b.conf.item()), 3)) for b in boxes]
    print("Detections:", dets)