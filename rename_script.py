import os
import pandas as pd

# Step 1: Load your labels.csv file
df = pd.read_csv("labels.csv")  # reads the CSV into a table-like format

# Step 2: Define a function to clean up filenames
def normalize_filename(name):
    # Replace spaces with hyphens and make everything lowercase
    return name.lower().replace(" ", "-")

# 🔹 Define your image subfolder here
image_dir = "images"   # change this if your folder has a different name

# Step 3: Go through each row in the CSV
for i, row in df.iterrows():
    old_name = row["filename"]              # get the original filename
    new_name = normalize_filename(old_name) # create the cleaned-up version

    # Build full paths inside the subfolder
    old_path = os.path.join(image_dir, old_name)
    new_path = os.path.join(image_dir, new_name)

    # Debug print: shows what the script is seeing
    print(f"Row {i}: old='{old_path}' -> new='{new_path}' | exists={os.path.exists(old_path)}")

    # Step 4: Rename the actual image file if it exists
    if os.path.exists(old_path):
        os.rename(old_path, new_path)

    # Step 5: Update the filename in the CSV
    df.at[i, "filename"] = new_name

# Step 6: Save the updated CSV as a new file
df.to_csv("labels_normalized.csv", index=False)
