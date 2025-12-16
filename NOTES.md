# Notes on Code for PPE Dataset

This file documents every code line we wrote so far, with simple explanations for non‑programmers. It also explains decisions and thought processes throughtout the project. It is in chronological order, i.e. most recent changes are at the top of the file.

17/12/25

🧾 sim2.png Results
Raw detections (15 total):
- Helmets: 0.959, 0.952, 0.948, 0.938, 0.833, 0.817, 0.665, 0.378
- Vests: 0.932, 0.928, 0.925, 0.857, 0.790, 0.772, 0.755
Outlier removed:
- Double detection helmet (0.378) → true count is 7 helmets, 7 vests.
Remaining confidences (14 total):
- Helmets: 0.959, 0.952, 0.948, 0.938, 0.833, 0.817, 0.665
- Vests: 0.932, 0.928, 0.925, 0.857, 0.790, 0.772, 0.755
Sorted confidences:
0.665, 0.755, 0.772, 0.790, 0.817, 0.833, 0.857, 0.925, 0.928, 0.932, 0.938, 0.948, 0.952, 0.959
Median (14 values):
Average of 7th and 8th values → (0.857 + 0.925) / 2 = 0.891

🧾 sim1.png Results
Raw detections (6 total):
- Helmets: 0.790, 0.633
- Vests: 0.772, 0.761, 0.707, 0.318
Outliers removed:
- Missing helmet (not detected at all).
- Obscured vest (0.318, very low confidence).
Remaining confidences (4 total):
- Helmets: 0.790, 0.633
- Vests: 0.772, 0.761
Sorted confidences:
0.633, 0.761, 0.772, 0.790
Median (4 values):
Average of 2nd and 3rd values → (0.761 + 0.772) / 2 = 0.767

### Median Confidence Analysis

- **sim2.png (true 7 helmets, 7 vests):**
  - Outlier removed: duplicate helmet (0.378).
  - Median confidence = **0.891**.
  - Interpretation: detections are generally strong, clustered around high confidence.

- **sim1.png (true 3 helmets, 4 vests):**
  - Outliers removed: missing helmet, obscured vest (0.318).
  - Median confidence = **0.767**.
  - Interpretation: detections are weaker overall, with lower confidence spread compared to sim2.png.

17/12/25

## Inference Results — sim2.png

### Observations
- Model initially reported 8 helmets and 7 safety vests.
- On review, one helmet was **double detected** because of a yellow pillar directly behind it.
- True count: **7 helmets** and **7 safety vests**.
- Annotated image saved as `annotated_output2.jpg`.

### My Thoughts
- The duplicate detection shows how background structures (like the yellow pillar) can confuse the model into proposing multiple bounding boxes for the same object.
- This highlights a limitation in non‑maximum suppression (NMS): if two overlapping proposals differ enough in confidence or position, both can survive.
- Compared to sim1.png, detections here were stronger overall, but the double detection reminds me that background context can still mislead the model.

### My Reasoning
- **Background interference:** The yellow pillar provided strong edges and color contrast, which the model may have misinterpreted as part of another helmet.
- **NMS behavior:** If two bounding boxes overlap but aren’t close enough in IoU, NMS may keep both, resulting in duplicates.
- **Dataset bias:** Training data may not have enough examples of helmets against similar colored or structured backgrounds, so the model struggles to distinguish helmet vs. pillar.
- **Confidence spread:** The duplicate helmet had lower confidence (0.378), suggesting the model was uncertain but still proposed it.

### Takeaway
- True detections: 7 helmets, 7 vests.
- Double detection occurred due to background confusion with a yellow pillar.
- Next step: expand dataset with helmets against varied backgrounds (including pillars, poles, and bright structures) to reduce false duplicates.



17/12/25

Notes update on inference results
- Observation: On images\sim1.png, the model detected 2 helmets and 4 safety vests. It missed a very clear white helmet in the center, yet picked up a small, partially obscured safety vest behind it with low confidence (0.318).
- Immediate thought: It seems plausible that the bright, saturated vest (even though smaller) was more recognizable than the white helmet, which may have blended with the sky due to similar color/brightness.
- Screenshot artifact: Annotated output saved as annotated_output.jpg confirms bounding boxes align with the printed detections.

My reasoning on why this happened
- Color and contrast cues: I suspect the white helmet had low contrast against the bright sky, reducing edge definition and gradient cues the model relies on. In contrast, the vest’s saturated, high-contrast color likely produced stronger feature activations, even at a smaller scale.
- Occlusion and shape cues: The helmet’s contour may have been partially occluded by the person behind it, disrupting its canonical shape; the vest’s rectangular, high-saturation panel remained visible enough to trigger detection.
- Scale sensitivity: At input size 384×640, small objects can be retained while mid-size objects with weak contrast are suppressed during feature downsampling. The vest’s compact, high-contrast patch may have survived downsampling better than the low-contrast helmet.
- NMS and confidence dynamics: If overlapping or adjacent boxes were proposed, non-maximum suppression may have favored the vest proposals due to slightly higher local confidence, while borderline helmet proposals were suppressed.
- Dataset bias and class balance: My training data may have more varied, high-visibility vests than white helmets against bright backgrounds, biasing the model toward vest features. Helmets in my dataset might skew toward darker colors or indoor lighting, reducing generalization to white helmets under outdoor sky.
- Augmentation coverage: If augmentations didn’t sufficiently cover high-brightness backgrounds, glare, or washed-out edges, the model could underperform on white objects against sky.
- Threshold and postprocessing: The helmet proposals could have landed just below the confidence threshold, while the vest—despite being small—cleared the threshold due to stronger color cues.

- My finding: I noticed the model completely missed a clear white helmet in the center but detected a very small, partially obscured safety vest behind it at 0.318 confidence.
- My hypothesis: The vest’s bright, saturated color is more distinctive than the white helmet against the sky. The helmet likely blended with the background, reducing edge and contrast cues, while the vest’s high-contrast patch triggered stronger features even at a smaller scale.
- My reasoning: Occlusion disrupted the helmet’s shape, and at 384×640 the vest’s compact, high-contrast region survived downsampling better than the helmet’s low-contrast edges. NMS may have suppressed weak helmet proposals. My dataset might also be biased toward vests or darker helmets, and my augmentations may not sufficiently cover bright-sky scenarios.
- My takeaway: High-visibility PPE (vests) can be easier for the model than low-contrast helmets against bright backgrounds. I’ll need to harden the model against white-on-bright conditions and partial occlusions.

- Next test: switched to images\sim2.png to compare detection behavior against sim1.png.
- Reason: sim1.png showed a missed helmet and a low-confidence vest detection; sim2.png will help confirm whether the issue is dataset bias, color/contrast, or occlusion sensitivity.



# NOTES.md — 17 Dec 2025

## Context
Working on PPE detector pipeline with YOLO. Built `file_inference.py` first to confirm detections, then extended into `visualize_inference.py` for annotated outputs.

---

## Decisions

### 1. Building `file_inference.py`
- **Goal:** Establish a minimal, reproducible script to load my trained checkpoint (`best.pt`) and run inference on an unseen image (`images\sim1.png`).
- **Reasoning:**  
  - Needed a clean baseline to verify that the model loads correctly and produces detections.  
  - Printing `(label, confidence)` pairs gave me a human‑readable log of what the model saw.  
  - This atomic step ensured reproducibility before adding complexity.

---

### 2. Extending to `visualize_inference.py`
- **Goal:** Move beyond printed detections to visual confirmation of bounding boxes and labels.  
- **Reasoning:**  
  - Visualization provides intuitive validation — I can *see* whether detections align with PPE items.  
  - Annotated outputs (`annotated_output.jpg`) serve as artifacts for reviewers and Git commits.  
  - Separating into a new file keeps the workflow modular: `file_inference.py` for textual logs, `visualize_inference.py` for graphical outputs.

---

### 3. Choosing `cv2.destroyAllWindows()`
- **Goal:** Ensure clean closure of OpenCV windows after visualization.  
- **Reasoning:**  
  - Without `destroyAllWindows()`, OpenCV windows can linger, causing resource leaks or blocking subsequent runs.  
  - Explicitly closing windows maintains reproducibility and prevents confusion during iterative testing.  
  - This choice reflects my workflow discipline: every process should end cleanly, leaving no hidden state.

---

## Thought Process
- Started with questions: *Can my model load and detect reliably?*  
- Answered by building `file_inference.py` — print‑only, atomic, reproducible.  
- Next question: *How do I confirm detections visually?*  
- Answered by creating `visualize_inference.py` — bounding boxes, labels, annotated outputs.  
- Final consideration: *How do I prevent clutter or leaks when showing images?*  
- Answered by using `cv2.destroyAllWindows()` — ensures clean teardown after each visualization.
- Important detail: the window only closes after a keypress is registered inside the OpenCV image window (focus matters — e.g. pressing space while the cursor is in the window), otherwise it stays open until you interact with it.

---

## Next Steps
1. Run `visualize_inference.py` on `images\sim1.png`.  
2. Save and commit `annotated_output.jpg` + script to Git.  
3. Journal milestone in NOTES.md.  
4. Prepare for MQTT transmission of annotated images.

16/12/25

# Project Notes – PPE Detector Simulation

## MQTT Simulation Flow
- **Publisher (publisher_sim.py)**
  - Reads `images/sim1.png`, encodes to base64, publishes to topic `ppe/camera/esp32cam/frame`.
  - Added `client.loop(1)` after `publish()` to flush the message before disconnecting.
  - Verified in Mosquitto logs: `Received PUBLISH` now appears.

- **Subscriber (subscriber.py)**
  - Subscribes to `ppe/camera/esp32cam/frame`.
  - Decodes base64 payload, reconstructs image with OpenCV, passes to `run_inference_stub`.
  - Confirmed detection output: `Detections: helmet (94.8%)`.

- **Broker (Mosquitto)**
  - Running in local-only mode on port 1883.
  - Logs confirm subscriber receives messages after publisher fix.

## Inference Stub
- File: `run_inference_stub.py`
- Provides a fake detection result for testing:
  ```python
  def run_inference(img):
      return [("helmet", 0.948)]

- This ensures the MQTT flow can be validated without requiring a full YOLO model.

Key Lessons
- publish() is asynchronous; without loop() or a delay, messages may not flush.
- Always run subscriber before publisher to avoid missing QoS 0 messages.
- Absolute paths for images prevent ambiguity when running scripts from different directories.

Next Steps
- Replace run_inference_stub.py with actual YOLO inference.
- Add multi-topic subscriber support for multiple edge devices (ESP32, Jetson).
- Implement logging/timestamping for detections.



16/12/25

## Smoke Test – 16 Dec 2025
- Ran smoke_test.py inside Python 3.8 `.venv`.
- Torch version: 2.4.1+cpu, CUDA not available (CPU inference only).
- Verified matrix multiply result printed successfully.
- Confirms PyTorch environment is stable and reproducible.
- Next step: integrate run_inference into MQTT subscriber for simulation pipeline.

15/12/25

Training Session — Pipeline Smoke Test (Dec 2025)

Purpose

Ran a 300-epoch training cycle to validate pipeline stability.

Training stopped early at epoch 175 due to patience (no improvement for 100 epochs).

Goal was to confirm end-to-end workflow, not to improve results.

Setup

Model: YOLOv8n

Epochs: 300 (early stopped at 175)

Image size: 640

Device: CPU (Ryzen AI 9 365)

Environment: .venv (Python 3.8, Torch 2.4.1+cpu)

Observations

Results matched the earlier 30-minute run.

Best checkpoint saved at epoch 75 (best.pt).

Early stopping confirmed pipeline is functioning correctly.

No new insights gained; this was a reproducibility check.

Notes

Pipeline runs cleanly from activation of .venv through training and logging.

Git commit optional: no dataset or config changes.

Documented here for traceability without cluttering commit history.

Next Steps

Focus on dataset expansion (goggles, gloves, boots) to extend learning.

Consider GPU training for faster convergence and larger image sizes.

Adjust patience if longer plateau exploration is desired.

Continue logging smoke tests separately from major training runs.

13/12/25

Training Session Log — PPE Detector (30‑min run)
I ran a 30‑minute training session using YOLOv8n with my PPE dataset.
The run was configured for epochs=300 at imgsz=640, but training stopped early at 175 epochs due to no improvement in the last 100 epochs. The best checkpoint was saved at epoch 75.

Dataset
- Validation set: 10 images, 133 labeled objects.
- Classes: helmet, safety vest, right glove, left glove, right boot, left boot, safety goggle.
- Total instances per class (val):
- Helmet: 37
- Safety vest: 40
- Right glove: 12
- Left glove: 18
- Right boot: 8
- Left boot: 13
- Safety goggle: 5

Results (best.pt at epoch 75)
- Overall:
- Precision: 0.892
- Recall: 0.582
- mAP@50: 0.692
- mAP@50‑95: 0.38
- Per class:
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
- Safety goggles: I need at least 30–50 more diverse images (different angles, lighting, contexts) to improve recall.
- Gloves: Both left and right gloves should be expanded to 40–50 images each to balance recall.
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

11/12/25

[2025-12-11] Baseline environment locked:
Python 3.13.11, NumPy 2.3.5, Torch 2.9.1+cpu, TorchVision 0.24.1+cpu, TorchAudio 2.9.1+cpu.
Verified torch.cuda.is_available() = False.



11/12/25

# Environment Notes — December 2025

## PyTorch + CUDA Compatibility
- Installed: PyTorch 2.1.2+cu121
- GPU detected: NVIDIA GeForce RTX 5070 Laptop (CUDA capability sm_120)
- Warning: Current PyTorch build supports sm_50–sm_90 only. sm_120 is **not yet supported**.
- Result: GPU kernels fail with `RuntimeError: no kernel image is available for execution on the device`.

## Decision
- For reproducibility and stability, we will **run all training on CPU** until PyTorch adds sm_120 support.
- `smoke_test.py` updated to force CPU execution (no CUDA calls).
- NumPy pinned to 1.26.4 for compatibility with PyTorch 2.1.2.

## Next Steps
- Monitor PyTorch nightly/stable releases for sm_120 support.
- Once supported, re‑enable GPU training and update NOTES.md.

11/12/25

PyTorch 2.1.2+cu121 installed
NumPy pinned to 1.26.4 for compatibility
GPU detected: NVIDIA GeForce RTX 5070 Laptop (sm_120)
Warning: sm_120 not yet supported in PyTorch 2.1.2 — falls back to PTX


11/12/25

Environment Strategy: Python 3.8 vs 3.9

Python 3.8 — ppe-labelimg38

Purpose: Stable baseline for annotation and reproducibility

Tools: LabelImg, dataset preparation, schema documentation

Strengths:

Proven compatibility with older packages

Documented in NOTES.md for reproducibility

Safe fallback if newer environments break

Usage:

Annotation workflows

Reviewer-friendly reproducibility

Legacy compatibility testing

Python 3.9 — ppe-train39

Purpose: Training environment with GPU acceleration

Tools: PyTorch nightly, YOLO training, CUDA-enabled workflows

Strengths:

Supports RTX 5070 (sm_120) via nightly builds

Future-proof for modern AI frameworks

Clean separation from annotation env

Usage:

YOLO smoke tests

GPU-accelerated training

Experimental workflows

Workflow Diagram

                ┌───────────────────────────────┐
                │   ppe-labelimg38 (Python 3.8) │
                │   --------------------------- │
                │   • Stable baseline           │
                │   • LabelImg annotation       │
                │   • Reproducibility (NOTES.md)│
                │   • Legacy compatibility      │
                └───────────────────────────────┘
                           │
                           │  (documented, safe fallback)
                           ▼
                ┌───────────────────────────────┐
                │   ppe-train39 (Python 3.9)    │
                │   --------------------------- │
                │   • GPU training (RTX 5070)   │
                │   • PyTorch nightly support   │
                │   • YOLO smoke tests          │
                │   • Future-proof experiments  │
                └───────────────────────────────┘

Summary

Keep 3.8 env for annotation and reproducibility.

Use 3.9 env for GPU-accelerated training.

Document both in NOTES.md to show clear separation of roles and reproducibility strategy.

11/12/25

# Dataset Notes

I defined my PPE categories in `classes.txt` as: helmet, safetyvest, right_glove, left_glove, right_boot, left_boot, safetygoggle. YOLO automatically assigns IDs starting from 0 in the order above, so 0 = helmet, 1 = safetyvest, 2 = right_glove, 3 = left_glove, 4 = right_boot, 5 = left_boot, and 6 = safetygoggle. This mapping connects the numbers in my annotation files to the actual PPE items. For every image (`.png`), I have a matching `.txt` file. Each line in the `.txt` file follows the YOLO format: `class_id center_x center_y width height`. The class_id is the PPE item ID from my class list, center_x and center_y are normalized coordinates of the bounding box center (0–1), and width and height are normalized size of the bounding box (0–1). Here’s an example from my dataset (`multi-9.txt`):

0 0.035473 0.512731 0.027027 0.032407   # helmet  
1 0.018159 0.558449 0.026182 0.054398   # safetyvest  
3 0.376689 0.799769 0.050676 0.062500   # left_glove  
2 0.301098 0.817708 0.054899 0.061343   # right_glove  
5 0.279561 0.887731 0.060811 0.055556   # left_boot  
4 0.195101 0.890625 0.038851 0.065972   # right_boot  

This shows multiple PPE items in one image. Each line is one bounding box, and if there are multiple people in an image, I simply have more lines — YOLO handles that naturally. I organized my dataset into three folders: `train/` for the majority of images and annotations, `val/` for the validation subset, and `test/` for the final evaluation subset. Each folder contains `.png` images and their paired `.txt` files, with no duplicates across folders so each split is unique. My setup is correct because I have a clear class list (`classes.txt`), my `.txt` files use the proper YOLO format (class ID + normalized coordinates), each image can have multiple PPE items and I annotated them correctly, and my dataset is split into train/val/test exactly as YOLO expects. This structure ensures my dataset is valid, reproducible, and ready for training.

7/12/25

PPE Detector Annotation Notes

Environment Setup

Created new Conda environment: ppe-labelimg38 with Python 3.8 for stability.

Installed LabelImg successfully with dependencies (PyQt5, lxml, sip).

Verified environment stability compared to previous 3.11 crashes.

Annotation Workflow

Images used: .png format.

Annotation format: YOLO .txt files.

Each image has a paired .txt file containing bounding box coordinates and class IDs.

Skipped images without PPE items → no .txt file generated (treated as background).

Rule: Annotate all visible PPE items; skip only if no PPE present.

Label Schema

helmet

safetyvest

right_glove

left_glove

right_boot

left_boot

safetygoggle

Conventions

Labels are lowercase with underscores.

Left/right distinction (if used) is based on image orientation, not annotator perspective.

Consistency enforced across all annotations.

Verification Checklist

Each .png has a matching .txt unless intentionally skipped.

Spot-checked .txt files for correct class IDs and normalized coordinates.

Reopened annotated images in LabelImg to confirm bounding boxes and labels.

Next Steps

Organize dataset into train/, val/, and test/ folders.

Maintain paired image + annotation files in each split.

Document schema and background rules for reproducibility.

Prepare dataset for YOLOv8 training pipeline.

Summary: Annotation completed, dataset saved in YOLO format with consistent schema and reproducible conventions.

6/12/25

## Environment Activation Setup (PowerShell)

- **Issue:** Prompt did not show `(base)` or `(ppe-detector)` after running `conda activate`.
- **Cause:** Custom `profile.ps1` contained a `function prompt` override that hid Conda’s environment name.
- **Fix:** Commented out the custom prompt override in `profile.ps1` so Conda’s default hook could display environment names.
- **Verification:**
  - Restarted VS Code terminal → `(base)` appeared by default.
  - Ran `conda activate ppe-detector` → prompt changed to `(ppe-detector)`.
  - Confirmed with `python --version` and `conda list` inside the environment.
- **Outcome:** PowerShell now shows the active Conda environment clearly, ensuring reproducibility for reviewers.

6/12/25

# Checkpoint: Filenames Normalized

## Context
- We installed Miniconda (Conda 25.9.1) and created a reproducible environment named `ppe-detector`.
- Verified pandas installation inside the environment.
- Ran `rename_script.py` to normalize filenames in the dataset.

## Actions Taken
1. **Environment Setup**
   ```bash
   conda create --name ppe-detector python=3.14   # isolated env for this project
   conda activate ppe-detector                    # switch into the env
   conda install pandas                           # install pandas for CSV handling

2. Script Execution
    ```bash
    python rename_script.py
# - Explanation: Script reads labels.csv, normalizes filenames (lowercase + hyphens), renames actual files in images/, and saves results to labels_normalized.csv.

3. Debugging
# Added inline debug prints to confirm each file was found and renamed:
Row 0: old='images/front.png' -> new='images/front.png' | exists=True
# - Explanation: exists=True confirmed the script was correctly pointing to the images/ subfolder.

4. Verification
ls images                     # list renamed files
head labels_normalized.csv    # preview normalized CSV
wc -l labels_normalized.csv   # count rows in CSV
ls images | wc -l             # count files in folder

# Explanation: Counts matched between CSV rows and image files, proving all files were renamed and logged

Outcome: 
# All dataset images/ were renamed to a clean, reproducible format. labels_normalized.csv created with filenames. 

Next Milestone
# Stage 1: Cropping images based on labels_normalized.csv
# This will ensure dataset entries align with normalized schema before moving into model training.

6/12/25
# --- Conda Environment Setup for PPE Detector ---
# This section documents how to create and manage the reproducible environment.

# 1. Create a new Conda environment named "ppe-detector" with Python 3.14
#    - Each project gets its own isolated environment
#    - Ensures dependencies don't conflict across projects
conda create --name ppe-detector python=3.14

# 2. Activate the environment
#    - Switches your shell into the "ppe-detector" environment
#    - All installs and runs now happen inside this env
conda activate ppe-detector

# 3. Install pandas inside the environment
#    - pandas is required for CSV handling in rename_script.py
#    - Installing inside the env keeps it reproducible
conda install pandas

# 4. Verify pandas is installed
#    - Lists all packages in the environment
#    - Confirms pandas version for reproducibility
conda list pandas

# 5. Run the CSV renaming script
#    - Reads labels.csv and outputs labels_normalized.csv
#    - Uses pandas inside the env
python rename_script.py

# --- Notes ---
# - Miniconda was installed once globally; no need to reinstall for each project.
# - Each new project should have its own environment created with "conda create".
# - Export environment for reviewers with:
#     conda env export > environment.yml
# - Reviewers can recreate your exact setup with:
#     conda env create -f environment.yml

3/12/25
Why I Installed Git (Windows, Git Bash)
- Version control: Git tracks every change I make to my code, so I can roll back safely, compare history, and collaborate without losing work.
- Reproducibility: My projects (like the PPE pipeline) need clean, reviewer‑friendly workflows. Git ensures every step is documented and reproducible.
- Collaboration: GitHub integration lets me share code, receive feedback, and contribute to others’ projects.
- Cross‑platform consistency: By setting line endings and SSH/HTTPS defaults, I’ve future‑proofed my repos to work seamlessly on Windows, macOS, and Linux.
- Professional portfolio: Commits tagged with my identity build a verifiable record of contributions, strengthening my credibility for career transition.
- Automation & safety: Git ties directly into CI/CD pipelines, testing, and deployment — critical for scaling my AI and edge device experiments.

⚡ In short: I installed Git to make my projects reproducible, collaborative, and professional, while keeping workflows consistent across platforms.

Git Installation Decisions (Windows, Git Bash)
Installer Choices
- Editor: Visual Studio Code
- Initial branch name: main
- PATH environment: Use Git from the command line and also from 3rd‑party software
- SSH executable: Bundled OpenSSH
- HTTPS backend: OpenSSL library (instead of Windows Secure Channel)
- Line ending conversions: Checkout Windows‑style (CRLF), commit Unix‑style (LF)
- Terminal emulator: MinTTY (default, recommended)
- git pull behavior: Merge (default)
- Credential helper: Git Credential Manager (enabled)
- Extra options: File system caching (enabled), Symbolic links (disabled)
Notes
- .gitignore and .gitattributes files in repos remain untouched — installer only sets defaults.
- Global Git identity (user.name, user.email) is important for GitHub attribution, but can be set later.
- Using SSH keys avoids repeated password prompts; HTTPS + Credential Manager caches credentials after first use.
- Repo commits are stored with LF endings, ensuring cross‑platform compatibility (Windows, macOS, Linux).

Git Revert / Reset Cheat Sheet

View History = git log --oneline
- Shows commit history with short hashes, dates, and messages.

Undo the latest commit (keep changes) = git reset --soft HEAD~1
- Removes the commit but keeps your edits staged.

Undo the latest commit (discard changes) = git reset --hard HEAD~1
- Removes the commit and wipes the changes.

Go back to a specific commit (temporary checkout) = git checkout <commit-hash>
- Switches to that snapshot without deleting history.

Permanently reset to a specific commit = git reset --hard <commit-hash>
- Resets your project to that commit, discarding later changes.

Safely undo a commit (preserve history) = git revert <commit-hash>
- Creates a new commit that undoes the changes, keeping history intact.




---

## Creating the CSV file

## 1. Printing a message
```python
print("Hello Shaun")
#print(...) tells Python to show something on the screen.
#Here, it shows the text Hello Shaun.
#This is the simplest way to check that Python is working.


## 2. Importing the CSV module
import csv
#import means "bring in extra tools."
#csv is a built‑in tool for working with comma‑separated values files (CSV).
#We use it to create and read our labels.csv.

## 3. Defining the list of filenames

['filename', 'angle', 'gear_combo']
['front.png', 'front', 'full_ppe']
['front long.png', 'front', 'full_ppe']
['left.png', 'left', 'full_ppe']
['rear.png', 'back', 'full_ppe']
['right.png', 'right', 'full_ppe']
['right long.png', 'right', 'full_ppe']
['right long 2.png', 'right', 'full_ppe']

#filenames = [...] creates a list (like a shopping list).
#Each item in the list is the name of one image file.
#We use this list to loop through all images later.

## 4. Writing the CSV file

with open("labels.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "angle", "gear_combo"])
    for fname in filenames:
        if "front" in fname:
            angle = "front"
        elif "left" in fname:
            angle = "left"
        elif "rear" in fname:
            angle = "back"
        elif "right" in fname:
            angle = "right"
        else:
            angle = "unknown"
        
        writer.writerow([fname, angle, "full_ppe"])

#Line‑by‑line explanation:
##with open("labels.csv", "w", newline="") as f:
##Opens a new file called labels.csv.
##"w" means write mode (we are creating or overwriting the file).
##newline="" avoids extra blank lines in the file.
##as f gives us a shortcut name (f) to refer to the file.
##writer = csv.writer(f)
##Creates a writer object that knows how to put rows into the CSV file.
##writer.writerow(["filename", "angle", "gear_combo"])
##Writes the header row (column names).
##This makes the CSV easy to understand later.
##for fname in filenames:
##Loops through each filename in our list.
##Example: first loop → fname = "front.png", second loop → fname = "front long.png", etc.
##if "front" in fname:
##Checks if the word "front" is inside the filename.
##If yes, we set angle = "front".
##elif "left" in fname:
##If not front, check if "left" is in the filename.
##If yes, set angle = "left".
##elif "rear" in fname:
##If not front or left, check if "rear" is in the filename.
##If yes, set angle = "back".
##elif "right" in fname:
##If not front, left, or rear, check if "right" is in the filename.
##If yes, set angle = "right".
##else: angle = "unknown"
##If none of the words match, mark the angle as "unknown".
##writer.writerow([fname, angle, "full_ppe"])
##Writes one row into the CSV with three values:
##The filename (e.g. front.png)
##The angle (e.g. front)
##The gear combo (always "full_ppe" for now).

## 5. Verifying the CSV contents
with open("labels.csv", "r") as f:
    print(f.read())

#Opens the file in read mode.
#Prints the entire file contents to confirm it looks correct

7. Why CSV First, Environment Later
I created the CSV before setting up a Python environment (.venv) because:

- Dataset organization comes first: Labels must be correct and reproducible before training begins.

- No external packages needed: The CSV script uses only Python’s built‑in tools, so no environment setup is required yet.

- Avoids mismatches: By locking in filenames and labels early, we prevent errors when training scripts are introduced.

- Separation of concerns: Dataset prep is independent of environment setup. .venv will be used later for heavier tasks like PyTorch training.

8. How This Connects to .venv and Training

- Once the dataset is stable, we will create a .venv (virtual environment) to keep dependencies isolated and reproducible.

- Inside .venv, we’ll install packages like:

- NumPy / Pandas for data handling.

- PyTorch for model training.

- Matplotlib for visualization.

- The labels.csv will be loaded into these scripts to:

- Match each image to its label.

- Feed data into the training pipeline.

- Evaluate model accuracy against labeled angles and PPE status.

Summary

- We tested Python with a simple print.

- Imported the CSV tool.

- Listed filenames.

- Wrote a CSV with labels.

- Verified the file contents.

- Documented why dataset prep comes before environment setup.

- Next step: build .venv for training and connect the dataset to the model pipeline     











