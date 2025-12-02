# ppe-detector
AI model for detecting missing PPE on construction workers using PyTorch and edge devices

# PPE Dataset Preparation

## Overview
This repository contains the initial dataset preparation for a **Personal Protective Equipment (PPE) detection project**. 
It also documents my reproducible AI workflow setup, ensuring clean version control and cross-platform copatibility. 
It is recorded in chronological order, with latest changes at the top, with date format dd/mm/yy

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
