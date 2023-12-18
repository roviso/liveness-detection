import os
import shutil


# Configuration
data_dir = "CelebA_Spoof"
image_dirs = ["Data/train", "Data/test"]  # Adjust as per your dataset structure
meta_dirs = ["metas/intra-test", "metas/protocol 1", "metas/protocol 2"]



# Helper Functions
def file_exists(file_path):
    return os.path.exists(file_path)

def check_and_delete_unpaired_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.png'):
                base_name = file[:-4]  # Remove '.png' extension
                txt_file = f"{base_name}_BB.txt"
                txt_path = os.path.join(root, txt_file)
                if not file_exists(txt_path):
                    png_path = os.path.join(root, file)
                    print(f"Deleting unpaired file: {png_path}")
                    os.remove(png_path)

def is_folder_empty(folder_path):
    return not any(os.listdir(folder_path))

def remove_empty_folders(directory):
    for root, dirs, files in os.walk(directory, topdown=False):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            if is_folder_empty(dir_path):
                print(f"Removing empty folder: {dir_path}")
                shutil.rmtree(dir_path)


# Check and Delete Unpaired Files
for dir in image_dirs:
    check_and_delete_unpaired_files(os.path.join(data_dir, dir))
    remove_empty_folders(os.path.join(data_dir, dir))

print("Unpaired file cleanup complete.")
