from preprocessing.bad_np_to_good_np import convert_to_good_numpy_format
from preprocessing.bad_to_good_np_new_only import convert_to_good_numpy_format_new_only
from preprocessing.convolve_and_save import CONFIG, process_and_save_data
from utils.utils import DatasetName

combine_ver = 1

if __name__ =='__main__':

    if combine_ver == 0:
        for version in [
            ('_3', 'P_1')
        ]:
            convert_to_good_numpy_format_new_only(version)
            config = CONFIG[DatasetName.new]
            # change filename and postfix
            process_and_save_data(**config)

    if combine_ver == 1:
        dataset_name = DatasetName.liver
        convert_to_good_numpy_format(dataset_name)
        config = CONFIG[dataset_name]
        process_and_save_data(**config)
