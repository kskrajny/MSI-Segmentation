'''UNUSED
import os

if __name__ == '__main__':
    def clean_directory_except_clr_feat(directory_path):
        for root, dirs, files in os.walk(directory_path, topdown=False):
            # Remove unwanted files
            print(files)
            for file in files:
                if file != "CLR_feat.npy":
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                        print(f"Deleted file: {file_path}")
                    except Exception as e:
                        print(f"Error deleting file {file_path}: {e}")

            # Remove empty directories
            for dir_ in dirs:
                dir_path = os.path.join(root, dir_)
                try:
                    if not os.listdir(dir_path):  # Check if directory is empty
                        os.rmdir(dir_path)
                        print(f"Deleted empty directory: {dir_path}")
                except Exception as e:
                    print(f"Error deleting directory {dir_path}: {e}")

    # Replace '/path/to/results' with the path to your 'results' directory
    results_path = "/home/jakub/results"
    clean_directory_except_clr_feat(results_path)
'''