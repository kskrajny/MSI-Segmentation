import os
import time
from argparse import Namespace
from pathlib import Path
import numpy as np
from petastorm.spark import SparkDatasetConverter, make_spark_converter
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import PCA, VectorAssembler
from pyspark.ml.linalg import DenseVector, VectorUDT
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col

start = time.time()
# ------------------------ VARIABLES TO SET  ------------------------ #
# - path_to_data
# - parquet_name
# - output_folder
# - max_cluster
# - n_components

path_to_data = "dane/nowe/"
parquet_name = 'parquet_convolve_True'

# output_folder = 'results/pecherz_original_conv/'
# output_folder = 'results/pecherz_original/'
# max_clusters = 32
# n_components = 8

# output_folder = 'results/sztuczne_dane_original_conv/'
# output_folder = 'results/sztuczne_dane_original/'
# max_clusters = 6
# n_components = 3

# output_folder = 'results/watroba_original_conv/'
# output_folder = 'results/watroba_original/'
# max_clusters = 3
# n_components = 3

# output_folder = "results/nowe_dane_original_conv/"
output_folder = "results/nowe_dane_original/"
max_clusters = 3
n_components = 3

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
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.storage.memoryFraction", "0.8") \
    .config("spark.memory.fraction", "0.8") \
    .config("spark.driver.memory", "30g") \
    .config("spark.executor.memory", "30g") \
    .config("spark.driver.maxResultSize", "2g") \
    .getOrCreate()
spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, Path('.').absolute().as_uri())
df = spark.read.parquet(path_to_data + parquet_name).repartition(50).persist().sort('pos')
converted_data = make_spark_converter(df)


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
np.save(output_folder + 'Normal_K-Means', predictions)
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