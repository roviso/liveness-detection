import os
import json

# Configuration
data_dir = "CelebA_Spoof_small"  # Adjust to your specific path
json_files = ["test_label.json", "train_label.json"]  # List of JSON files to check

# Function to check if the file for a given key in JSON exists
def check_file_exists(json_key):
    file_path = os.path.join(data_dir, json_key)
    return os.path.isfile(file_path)

# Process each JSON file
for json_file in json_files:
    json_path = os.path.join(data_dir, 'metas', 'intra_test', json_file)
    if os.path.exists(json_path):
        with open(json_path, 'r') as file:
            data = json.load(file)

        # Check each key and remove if the corresponding .png does not exist
        data_keys = list(data.keys())  # Create a list of keys to iterate over
        for key in data_keys:
            if not check_file_exists(key):
                del data[key]  # Remove the key if the file does not exist

        # Write the updated data back to the JSON file
        with open(json_path, 'w') as file:
            json.dump(data, file, indent=4)
    else:
        print(f"File {json_file} does not exist in the directory structure.")

print("JSON files have been updated.")
