import numpy as np
from matplotlib import pyplot as plt
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession
import scipy.stats

# ------------------------ VARIABLES TO SET  ------------------------ #
# path_to_data = "dane/sztuczne_dane/"
# conv_len = 5

# path_to_data = "dane/pecherz/"
# conv_len = 10

path_to_data = "dane/watroba/"
conv_len = 10

convolve = False
# ------------------------------------------------------------------- #


name = f'parquet_convolve_{convolve}'
file = path_to_data + f'root/intsy_dobre_magisterka.npy'

spark = SparkSession.builder \
    .master("local") \
    .appName("Magisterka") \
    .config("spark.driver.memory", "15g") \
    .getOrCreate()


data = np.load(file, allow_pickle=True).astype(np.float32)

if convolve:
    arr = scipy.stats.norm(0, 1).pdf(np.arange(-1.5, 1.5, 3 / conv_len))
    arr = arr - min(arr)
    data = np.apply_along_axis(
        lambda x: np.convolve(
            x,
            arr / sum(arr),
            mode='same'
        ),
        1,
        data
    )
i = 0
step = 40

while i < data.shape[0]:
    # plt.plot(data[i])
    # plt.show()
    arr = map(lambda x:
              (int(i + x[0]),
               Vectors.dense(x[1] + np.random.normal(0, 1e-8, x[1].shape))),
              enumerate(data[i: i + step])
              )
    spark.createDataFrame(arr, ['pos', 'features'])\
        .write.parquet(path_to_data + name, mode="append")
    i += step
