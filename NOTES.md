# Notes on Code for PPE Dataset

This file documents every code line we wrote so far, with simple explanations for non‑programmers. It also explains decisions and thought processes throughtout the project. It is in chronological order, i.e. most recent changes are at the top of the file.

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











