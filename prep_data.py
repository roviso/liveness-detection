import os
import json
import shutil

# Configuration
data_dir = "CelebA_Spoof"
backup_dir = "CelebA_Spoof_small"
image_dirs = ["Data/train", "Data/test"]
meta_dirs = ["metas/intra_test", "metas/protocol1", "metas/protocol2"]
expected_file_types = ['jpg', 'txt', 'json']

# Backup Original Data
if not os.path.exists(backup_dir):
    shutil.copytree(data_dir, backup_dir)


# Helper Function to Check File Existence
def file_exists(file_path):
    return os.path.exists(file_path)



# Helper Functions
def verify_files(directory, file_extension):
    missing_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(file_extension):
                file_path = os.path.join(root, file)
                if not os.path.exists(file_path):
                    missing_files.append(file_path)
    return missing_files

# File Verification
missing_files = {}
for dir in image_dirs + meta_dirs:
    for file_type in expected_file_types:
        missing = verify_files(os.path.join(data_dir, dir), file_type)
        if missing:
            missing_files[file_type] = missing

# Log Missing Files
if missing_files:
    with open(os.path.join(data_dir, "missing_files_log.txt"), "w") as log:
        for file_type, files in missing_files.items():
            log.write(f"Missing {file_type} files:\n")
            for file in files:
                log.write(file + "\n")

# Update JSON Files
for meta_dir in meta_dirs:
    json_files = [f for f in os.listdir(os.path.join(data_dir, meta_dir)) if f.endswith('.json')]
    for json_file in json_files:
        json_path = os.path.join(data_dir, meta_dir, json_file)
        with open(json_path, 'r') as file:
            data = json.load(file)

        updated_data = {}
        for key, value in data.items():
            image_path = os.path.join(data_dir, key)
            bbox_path = image_path.replace('.jpg', '_BB.txt')

            if file_exists(image_path) and file_exists(bbox_path):
                updated_data[key] = value
            else:
                print(f"Missing file for {key}")

        with open(json_path, 'w') as file:
            json.dump(updated_data, file, indent=4)




print("JSON files have been updated.")
print("Dataset verification and update complete.")
