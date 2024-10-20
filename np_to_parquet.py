import os

import numpy as np
from matplotlib import pyplot as plt
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession
import scipy.stats

if __name__ ==  '__main__':
    for t in 'TPOLJH':
        # ------------------------ VARIABLES TO SET  ------------------------ #
        # path_to_data = "dane/sztuczne_dane/"
        # conv_len = 5

        # path_to_data = "dane/pecherz/"
        # conv_len = 10

        # path_to_data = "dane/watroba/"
        # conv_len = 10

        path_to_data = "dane/nowe/"
        conv_len = 10

        convolve = True
        # ------------------------------------------------------------------- #

        name = f'parquet_convolve_{convolve}'

        # Lista plików numpy do wczytania
        file_list = []

        for n in [1, 2, 3]:
            v = f'{t}_{n}'
            file_list.append(
                f'dane/nowe/root/filtered_intsy_{v}.npy'
            )

        '''
        # Inicjalizacja sesji Spark
        spark = SparkSession.builder \
            .master("local") \
            .appName("Magisterka") \
            .config("spark.driver.memory", "15g") \
            .getOrCreate()
        '''

        # Ładowanie i łączenie danych z plików numpy
        data_list = [np.load(file, allow_pickle=True).astype(np.float32) for file in file_list]
        data = np.concatenate(data_list, axis=0)

        # Opcjonalna konwolucja
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

        # Zapisanie połączonych danych do pliku numpy
        output_file = os.path.join(path_to_data, f'numpy_{convolve}_{t}.npy')
        print(data.shape)
        np.save(output_file, data)

        print(f'Dane zapisane do pliku: {output_file}')
        '''
        i = 0
        step = 100
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
        '''