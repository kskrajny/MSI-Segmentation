import os
import warnings
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.utils import DatasetName

current_dir = os.getcwd()
if not current_dir.endswith('MSI-Segmentation'):
    os.chdir(os.path.abspath(os.path.join(current_dir, "..")))
    print(f"Working directory changed to: {os.getcwd()}")
else:
    print(f"Working directory remains: {current_dir}")

bad_format_intensities_files = {
    DatasetName.artificial: {
        'intsy_files': ["dane/sztuczne/root/intsy_sztuczne_magisterka.npy"],
        'mz_files': ["dane/sztuczne/root/mz_sztuczne_magisterka.npy"],
    },
    DatasetName.bladder: {
        'intsy_files': ["dane/pecherz/root/intsy_pecherz_magisterka.npy"],
        'mz_files': ["dane/pecherz/root/mz_pecherz_magisterka.npy"]
    },
    DatasetName.liver: {
        'intsy_files': [
            "dane/watroba/root/intsy_from_0_to_2999_magisterka.npy",
            "dane/watroba/root/intsy_from_3000_to_6000_magisterka.npy",
            "dane/watroba/root/intsy_from_6001_to_9000_magisterka.npy",
            "dane/watroba/root/intsy_from_9001_to_10011_magisterka.npy"
        ],
        'mz_files': [
            "dane/watroba/root/mz_from_0_to_2999_magisterka.npy",
            "dane/watroba/root/mz_from_3000_to_6000_magisterka.npy",
            "dane/watroba/root/mz_from_6001_to_9000_magisterka.npy",
            "dane/watroba/root/mz_from_9001_to_10011_magisterka.npy"
        ],
    }
}


def process_sample(sample_idx, intsy, mz, decimal):
    """Process a single sample (intensity and m/z pair)."""
    mz_val = np.around(mz[sample_idx], decimals=decimal)
    vals = pd.DataFrame({"mz": mz_val, "int": intsy[sample_idx]}).groupby("mz").sum()
    if not vals.empty:
        return pd.DataFrame([vals.int], columns=vals.index)
    else:
        return None  # Return None for empty results to handle separately


def process_intensities_multithread(intsy, mz, decimal, max_workers=4):
    """Process all samples in a single intensity/mz pair using multithreading."""
    df = pd.DataFrame()

    # Use ThreadPoolExecutor to parallelize sample processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_sample, i, intsy, mz, decimal): i for i in range(len(intsy))}
        for future in tqdm(futures, desc="Processing samples"):
            try:
                result = future.result()
                if result is not None:
                    df = pd.concat([df, result])
                else:  # Handle empty results by appending a row of zeros
                    df = pd.concat([df, pd.DataFrame([[0] * len(df.columns)], columns=df.columns)])
            except Exception as e:
                print(f"Error processing sample {futures[future]}: {e}")

    return df


def gather_intensities_values(intsy_files, mz_files, decimal, new_mz):
    """Run the intensity processing using multithreading."""
    df = pd.DataFrame(columns=new_mz)

    for ints_file, mz_file in zip(intsy_files, mz_files):
        intsy = np.load(ints_file, allow_pickle=True)
        mz = np.load(mz_file, allow_pickle=True)
        df = pd.concat([df, process_intensities_multithread(intsy, mz, decimal, 6)])

    df.fillna(0, inplace=True)

    return df



def get_dataframe_and_mz(intsy_files, mz_files, decimal):
    warnings.filterwarnings(
        "ignore", category=FutureWarning,
        message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated"
    )
    warnings.filterwarnings(
        "ignore", category=FutureWarning,
        message="Downcasting object dtype arrays on .fillna"
    )

    new_mz = []
    for mz_file in mz_files:
        mz_file = np.load(mz_file, allow_pickle=True)
        for mz in tqdm(mz_file, desc="Processing mz"):
            new_mz = np.array(new_mz + [mz.max(), mz.min()])
            new_mz = [new_mz.min(), new_mz.max()]

    new_mz = np.around(np.arange(
        np.around(new_mz[0], decimals=decimal),
        new_mz[1] + 1 / 10 ** decimal, 1 / 10 ** decimal),
        decimals=decimal
    )

    df = gather_intensities_values(intsy_files, mz_files, decimal, new_mz)

    return new_mz, df


def convert_to_good_numpy_format(dataset: DatasetName, decimal = 1):
    if not os.path.exists(f"dane/{dataset.value}/good_format"):
        os.makedirs(f"dane/{dataset.value}/good_format")

    # Load source data
    intsy_files = bad_format_intensities_files[dataset]['intsy_files']
    mz_files = bad_format_intensities_files[dataset]['mz_files']

    # Prepare dataframe with mz for numpy array in good format
    new_mz, df = get_dataframe_and_mz(intsy_files, mz_files, decimal)

    # Show example good format numpy array
    print(f'Shape intsy.npy : {df.shape}')
    print(f'Shape mz.npy : {len(new_mz)}')

    plt.plot(new_mz, df.values[0])
    plt.show()

    np.save(f"dane/{dataset.value}/good_format/intsy.npy", df.values)
    np.save(f"dane/{dataset.value}/good_format/mz.npy", new_mz)


if __name__ == '__main__':
    # convert_to_good_numpy_format(DatasetName.artificial)
    # convert_to_good_numpy_format(DatasetName.bladder)
    convert_to_good_numpy_format(DatasetName.liver)
    pass
