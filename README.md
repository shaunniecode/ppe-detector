# ppe-detector
AI model for detecting missing PPE on construction workers using PyTorch and edge devices

# PPE Dataset Preparation

## Overview
This repository contains the initial dataset preparation for a **Personal Protective Equipment (PPE) detection project**. 
It also documents my reproducible AI workflow setup, ensuring clean version control and cross-platform copatibility. 
It is recorded in chronological order, with latest changes at the top, with date format dd/mm/yy

16/12/25

# PPE Detector

This simulates an ESP32-CAM publishing frames to an MQTT broker, with a subscriber decoding and running inference. It is part of the same PPE Detector pipeline that includes dataset training and inference experiments.

## Components
- **publisher_sim.py** — Publishes a test image (`images/sim1.png`) to MQTT.  
- **subscriber.py** — Subscribes to topic, decodes payload, reconstructs image, runs inference.  
- **run_inference_stub.py** — Temporary stub returning static detection result (`helmet (94.8%)`).  
- **Mosquitto Broker** — Local broker (`localhost:1883`).

## Usage
1. Start Mosquitto broker: mosquitto -v
2. Run Subscriber: python subscriber.py
3. In another terminal, run publisher: python publisher_sim.py
4. Example output: Message received on ppe/camera/esp32cam/frame Detections: helmet (94.8%)

Notes- Publisher requires client.loop(1) after publish() to flush message.
- Subscriber must be running before publisher to capture QoS 0 messages.
- Current inference is stubbed; replace with YOLO model for real PPE detection.

Next Steps- Integrate YOLOv8 for actual inference.
- Expand dataset for weak classes (gloves, boots, goggles).
- Add logging and visualization of detections.
- Configure broker for remote edge devices (ESP32, Jetson Nano)







13/12/25

PPE Detector — Training Session (Dec 2025)
Overview
I ran a 30‑minute YOLOv8n training session on my PPE dataset. The run was configured for epochs=300 at imgsz=640, but training stopped early at 175 epochs due to no improvement in the last 100 epochs. The best checkpoint was saved at epoch 75 (best.pt).

Dataset
- Validation set: 10 images, 133 labeled objects.
- Classes: helmet, safety vest, right glove, left glove, right boot, left boot, safety goggle.
- Instances per class (val):
- Helmet: 37
- Safety vest: 40
- Right glove: 12
- Left glove: 18
- Right boot: 8
- Left boot: 13
- Safety goggle: 5

Results (best.pt at epoch 75)
- Overall metrics:
- Precision: 0.892
- Recall: 0.582
- mAP@50: 0.692
- mAP@50‑95: 0.38
- Per class performance:
- Helmet → P=0.901, R=0.892, mAP@50=0.939
- Safety vest → P=0.979, R=1.0, mAP@50=0.995
- Right glove → P=0.907, R=0.81, mAP@50=0.795
- Left glove → P=1.0, R=0.211, mAP@50=0.635
- Right boot → P=0.764, R=0.625, mAP@50=0.74
- Left boot → P=0.692, R=0.538, mAP@50=0.686
- Safety goggle → P=1.0, R=0.0, mAP@50=0.056

Observations
- Helmets and vests are performing very well — strong precision and recall.
- Gloves and boots show moderate learning, but recall is inconsistent (left glove recall is very low).
- Safety goggles are the weakest class: precision is high but recall is zero, meaning the model predicts them only when extremely confident, and misses most instances.
- Early stopping suggests the model plateaued quickly, which is common with small datasets and CPU training.

Recommendations
- Safety goggles: Add at least 30–50 more diverse images (different angles, lighting, contexts).
- Gloves: Expand to 40–50 images each to balance recall.
- Boots: Increase to 30–40 images each to stabilize detection.
- Helmets and vests: Already strong, but adding another 20–30 images each will help generalization.
- Overall dataset size: To push mAP@50‑95 above 0.6, I should aim for ~300–400 total labeled instances, balanced across all classes.


11/12/25

# PPE-DETECTOR Training Session (11 Dec 2025)

## Dataset Setup
- Created dataset/ folder with YOLO structure:
  dataset/
    train/images (12 PNGs)
    train/labels (12 TXTs)
    val/images (3 PNGs)
    val/labels (3 TXTs)
    test/images (2 PNGs)
    test/labels (2 TXTs)

- Verified .txt labels contained real bounding boxes.
- Confirmed dataset.yml points to dataset/train/images, dataset/val/images, dataset/test/images.

## Classes
- Defined 7 PPE classes:
  0 helmet
  1 safetyvest
  2 right_glove
  3 left_glove
  4 right_boot
  5 left_boot
  6 safetygoggle

## Validation
- Ran: yolo val model=yolov8n.pt data=dataset.yml imgsz=640 batch=1
- Output showed COCO classes (person, car, etc.) because pretrained weights were COCO-trained.
- Dataset parsed correctly (no corrupt files, labels found).

## Training
- Ran smoke test: yolo train model=yolov8n.pt data=dataset.yml epochs=2 imgsz=640 device=cpu
- Training completed successfully:
  - Loss values decreased slightly.
  - Metrics (Precision, Recall, mAP) = 0 (expected with tiny dataset + 2 epochs).
- Validation after training listed PPE classes correctly but showed 0 detections.

## Key Findings
- Pipeline is wired up correctly: dataset, labels, config, training loop all functional.
- Safety goggles not detected because only 1 annotated instance exists.
- With such a small dataset (17 images), YOLO cannot generalize yet.

04/12/25
Dataset expansion, I had used Adobe Firefly to generate several more images. Some images include multiple subjects (people), naming and labelling conventions has left me thinking how to develop what we already have for something new/er. Tempted to change the naming convention completely



03/12/25
Development Environment Setup
I installed Git for Windows (Git Bash) to manage version control and ensure reproducible workflows.
Key choices during installation:
- Editor: Visual Studio Code
- Default branch: main
- PATH: Git available from command line and 3rd‑party tools
- SSH: Bundled OpenSSH
- HTTPS: OpenSSL library
- Line endings: Checkout Windows‑style (CRLF), commit Unix‑style (LF)
- Terminal: MinTTY
- Credential Manager: Enabled
This setup ensures cross‑platform compatibility (Windows/macOS/Linux), secure GitHub integration, and clean commit history for collaborative projects.




02/12/25
The dataset currently includes 7 images of construction workers in **full PPE**, captured from multiple angles and perspectives.

## Current Files
- `images/`  
  - `front.png`  
  - `front long.png`  
  - `left.png`  
  - `rear.png`  
  - `right.png`  
  - `right long.png`  
  - `right long 2.png`
- `labels.csv`  
  - A structured file mapping each image to its **angle** and **gear_combo** (`full_ppe`).

## Why We Created `labels.csv` First
Before setting up a Python environment (`.venv`), we prioritized **dataset organization**:
- 🔹 **Clarity**: Ensures every image has a clear label (angle + gear).  
- 🔹 **Reproducibility**: Reviewers and collaborators can immediately understand the dataset without needing to run code.  
- 🔹 **Scalability**: Automating CSV generation now makes it easier to expand to thousands of images later.  
- 🔹 **Separation of concerns**: Dataset preparation is independent of environment setup. By locking in filenames and labels first, we avoid mismatches when training scripts are introduced.

## Next Steps
1. **Environment Setup (`.venv`)**  
   - Create a clean Python virtual environment for reproducibility.  
   - Install required packages (e.g., PyTorch, NumPy, Pandas).  
   - Keep dependencies isolated from system Python.

2. **Dataset Expansion**  
   - Add more images with consistent naming conventions.  
   - Use automation scripts to update `labels.csv`.

3. **Model Training Pipeline**  
   - Load images and labels into preprocessing scripts.  
   - Train a PPE detection model using PyTorch.  
   - Validate with test sets and refine labeling if needed.

## Notes
- All images currently represent **full PPE**.  
- Angles are inferred directly from filenames (`front`, `left`, `rear`, `right`).  
- Environment setup will follow once dataset preparation is stable, ensuring a clean slate for reproducible training.

---
This README serves as a **living document**. Update it as the dataset grows and the pipeline evolves.
