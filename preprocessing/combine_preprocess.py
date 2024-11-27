import os
from preprocessing.bad_np_to_good_np import convert_to_good_numpy_format
from preprocessing.bad_to_good_np_new_only import convert_to_good_numpy_format_new_only
from preprocessing.convolve_and_save import CONFIG, process_and_save_data
from utils.utils import DatasetName


current_dir = os.getcwd()
if not current_dir.endswith('MSI-Segmentation'):
    os.chdir(os.path.abspath(os.path.join(current_dir, "..")))
    print(f"Working directory changed to: {os.getcwd()}")
else:
    print(f"Working directory remains: {current_dir}")


combine_ver = 0

if __name__ =='__main__':

    if combine_ver == 0:
        letter = 'H'
        for version in [
            ('', f'{letter}_1'), ('', f'{letter}_2'), ('', f'{letter}_3'),
        ]:
            convert_to_good_numpy_format_new_only(version)
            config = CONFIG[DatasetName.new]
            postfix = f"{version[0]}_{version[1]}"
            config["postfix"] = postfix
            config["file_name"] = f"dane/nowe/root/filtered_intsy{postfix}.npy"
            process_and_save_data(**config)

    if combine_ver == 1:
        dataset_name = DatasetName.liver
        convert_to_good_numpy_format(dataset_name)
        config = CONFIG[dataset_name]
        process_and_save_data(**config)
