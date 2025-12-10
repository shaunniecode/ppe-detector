# Notes on Code for PPE Dataset

This file documents every code line we wrote so far, with simple explanations for non‑programmers. It also explains decisions and thought processes throughtout the project. It is in chronological order, i.e. most recent changes are at the top of the file.

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











