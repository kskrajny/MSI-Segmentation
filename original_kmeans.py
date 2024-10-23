import copy
import os
import time
import tracemalloc

from pyspark import StorageLevel
from pyspark.sql import functions as F
from argparse import Namespace
from pathlib import Path
import numpy as np
from petastorm.spark import SparkDatasetConverter, make_spark_converter
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import PCA, VectorAssembler
from pyspark.ml.linalg import DenseVector, VectorUDT, SparseVector
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, mean
from pyspark.sql.types import ArrayType, DoubleType
from tqdm import tqdm

from evaluate import evaluate
from spectral_representation import calculate_spectral_representation, visualize_spectral_selection, \
    visualize_selected_points_on_mean_spectrum

def original_kmeans(v, c, path_to_data="dane/nowe/"):
    start = time.time()
    parquet_name = f'parquet_convolve_False_{v}'

    # output_folder = 'results/pecherz_original_conv/'
    # output_folder = 'results/pecherz_original/'
    # max_clusters = 12
    # n_components = 6

    # output_folder = 'results/sztuczne_dane_original_conv/'
    # output_folder = 'results/sztuczne_dane_original/'
    # max_clusters = 6
    # n_components = 3

    # output_folder = 'results/watroba_original_conv/'
    # output_folder = 'results/watroba_original/'
    # max_clusters = 3
    # n_components = 3

    output_folder = f"results/nowe_dane{c}_{v}_original/"
    max_clusters = 5
    n_components = 10

    num_points = 128
    # ------------------------------------------------------------------- #


    # RESULT DIRECTORY
    try:
        os.mkdir(output_folder)
    except:
        pass

    # DATA
    spark = SparkSession.builder \
        .master("local") \
        .appName("Magisterka") \
        .config("spark.sql.shuffle.partitions", "50") \
        .config("spark.storage.memoryFraction", "0.8") \
        .config("spark.memory.fraction", "0.8") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .config("spark.driver.maxResultSize", "6g") \
        .getOrCreate()
    spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, Path('.').absolute().as_uri())
    df = spark.read.parquet(path_to_data + parquet_name)
    #df = df.filter(df['pos'] < 5)

    # Check memory usage at this point
    df = df.sort('pos')

    # Use the new function to modify the DataFrame
    df = calculate_spectral_representation(df, num_points=num_points, cache_dir=output_folder, output_folder=output_folder)

    # Call the visualization function
    visualize_spectral_selection(df, num_points=num_points, output_folder=output_folder)

    # PCA
    pca = PCA(k=n_components, inputCol="features")
    pca.setOutputCol("pca_features")
    model = pca.fit(df)
    df = model.transform(df)
    df = df.drop('features')

    kmeans_start = time.time()
    # NORMAL CLUSTERING
    kmeans = KMeans(k=max_clusters, featuresCol='pca_features')
    model = kmeans.fit(df)
    df = model.transform(df)
    predictions = np.array(df.select('prediction').collect())
    np.save(output_folder + f'Normal_K-Means', predictions)
    df = df.drop('prediction')

    break1 = time.time()
    print(f'KMEANS TIME: {break1 - start}')
    # ITERATIVE CLUSTERING
    # max_clusters = 8
    '''
    clusters = 1
    clusters_arr = [1]
    i = 0
    for i in range(n_components):
        f_udf = udf(lambda x: DenseVector([x[i]]), VectorUDT())
        df = df.withColumn('features', f_udf(col('pca_features')))
        print(clusters)
        scores = []
        r = range(2, 5)
        for k in r:
            kmeans = KMeans(k=k)
            model = kmeans.fit(df)
            df = model.transform(df)
            evaluator = ClusteringEvaluator()
            evaluator.setPredictionCol("prediction")
            scores.append(evaluator.evaluate(df))
            df = df.drop('prediction')
        print(max(scores))
        if max(scores) > 0.55:
            k = list(r)[int(np.argmax(scores))]
            kmeans = KMeans(k=k)
            model = kmeans.fit(df)
            df = model.transform(df)
            df = df.withColumnRenamed('prediction', f'prediction{i}')
            i += 1
            clusters *= k
            clusters_arr.append(k)
        df = df.drop('features')
        if max_clusters <= clusters:
            break
    
    vecAssembler = VectorAssembler(outputCol='prediction')
    vecAssembler.setInputCols([f'prediction{j}' for j in range(len(clusters_arr) - 1)])
    df = vecAssembler.transform(df)
    
    print(f'Iterative TIME: {time.time() - break1 + (kmeans_start - start)}')
    # SAVE LABELS
    predictions = np.squeeze(np.array(df.select('prediction').collect()))
    np.save(output_folder + 'Iterative_K-Means', predictions)
    '''
    spark.stop()

    for prefix in ['Normal_K-Means']:
        evaluate(prefix, output_folder)


if __name__ == "__main__":
    # ------------------------ VARIABLES TO SET  ------------------------ #
    # - path_to_data
    # - parquet_name
    # - output_folder
    # - max_cluster
    # - n_components
    for c in ['', '_3']:
        path_to_data = "dane/nowe/"
        #v = ''
        #v = '_P'
        for v in 'H':# 'TPLHOJ':
            original_kmeans(v, c, path_to_data)