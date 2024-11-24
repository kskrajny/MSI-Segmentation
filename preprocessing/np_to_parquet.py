import os
import numpy as np
from scipy.stats import norm
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession
from tqdm import tqdm
from utils.utils import DatasetName


# Set working directory one level up
os.chdir(os.path.abspath(os.path.join(os.getcwd(), "..")))
print(f"Current working directory: {os.getcwd()}")


CONFIG = {
    DatasetName.artificial: {
        "path_to_data": "dane/sztuczne/",
        "file_name": "dane/sztuczne/good_format/intsy.npy",
        "save_parquet": True,
        "save_numpy": True,
        "conv_len": 5,
        "convolve": True,
    },
    DatasetName.bladder: {
        "path_to_data": "dane/pecherz/",
        "file_name": "dane/pecherz/good_format/intsy.npy",
        "save_parquet": False,
        "save_numpy": True,
        "conv_len": 10,
        "convolve": True,
    },
    DatasetName.liver: {
        "path_to_data": "dane/watroba/",
        "file_name": "dane/watroba/good_format/intsy.npy",
        "save_parquet": True,
        "save_numpy": True,
        "conv_len": 10,
        "convolve": False,
    },
    DatasetName.new: {
        "postfix": "3_T_1",
        "path_to_data": "dane/nowe/",
        "file_name": "dane/nowe/root/filtered_intsy_3_P_1.npy",
        # "file_name": "dane/nowe/root/filtered_intsy_3_T_2.npy",
        # "file_name": "dane/nowe/root/filtered_intsy_3_T_3.npy",
        # "file_name": "dane/nowe/root/filtered_intsy_T_1.npy",
        # "file_name": "dane/nowe/root/filtered_intsy_T_2.npy",
        # ...
        # "file_name": "dane/nowe/root/filtered_intsy_3_H_1.npy",
        # "file_name": "dane/nowe/root/filtered_intsy_3_H_2.npy",
        # ...
        # "file_name": "dane/nowe/root/filtered_intsy_O_1.npy",
        # "file_name": "dane/nowe/root/filtered_intsy_O_2.npy",
        # ....
        # options: { , _3}_{T, P, O, L, H, J}_{1, 2, 3} like for example T_2, 3_O_2
        "save_parquet": False,
        "save_numpy": True,
        "conv_len": 10,
        "convolve": True,
    }
}


def initialize_spark(app_name="DataProcessing", memory="15g"):
    """Initialize a Spark session."""
    return (
        SparkSession.builder
        .master("local")
        .appName(app_name)
        .config("spark.driver.memory", memory)
        .getOrCreate()
    )


def apply_convolution(data, conv_len):
    """Apply convolution to the data if enabled."""
    kernel = norm(0, 1).pdf(np.linspace(-1.5, 1.5, conv_len))
    kernel /= kernel.sum()
    return np.apply_along_axis(lambda x: np.convolve(x, kernel, mode="same"), axis=1, arr=data)


def save_to_parquet(spark, data, path, batch_size=6):
    """Save numpy data to a Parquet file in batches."""

    for start_idx in tqdm(range(0, data.shape[0], batch_size), desc="Saving to Parquet"):
        batch_data = data[start_idx : start_idx + batch_size]
        records = [
            (int(start_idx + idx), Vectors.dense(row + np.random.normal(0, 1e-8, row.shape)))
            for idx, row in enumerate(batch_data)
        ]
        df = spark.createDataFrame(records, ["pos", "features"])
        df.write.parquet(path, mode="append")


def process_and_save_data(
        path_to_data: str, file_name: str, save_parquet: bool, save_numpy: bool,
        convolve: bool, conv_len: int, postfix: str = ''
    ):
    path_to_data += postfix
    if not os.path.exists(f"{path_to_data}"):
        os.makedirs(f"{path_to_data}")

    """Process numpy files and save them as Parquet."""
    data = np.load(file_name, allow_pickle=True).astype(np.float32)
    if convolve:
        data = apply_convolution(data, conv_len)
    if save_parquet:
        spark = initialize_spark("Numpy to Parquet")
        try:
            parquet_path = os.path.join(path_to_data, f"parquet_convolve_{convolve}")
            save_to_parquet(spark, data, parquet_path)
            print(f"Data successfully saved to {parquet_path}")
        finally:
            spark.stop()
    if save_numpy:
        numpy_path = os.path.join(path_to_data, f"numpy_convolve_{convolve}")
        np.save(numpy_path, data)
        print(f"Data successfully saved to {numpy_path}")


if __name__ == "__main__":
    # config = CONFIG[DatasetName.artificial]
    # config = CONFIG[DatasetName.bladder]
    # config = CONFIG[DatasetName.liver]
    config = CONFIG[DatasetName.new]
    process_and_save_data(**config)
