from argparse import Namespace
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from petastorm.spark import SparkDatasetConverter, make_spark_converter
from pyspark.sql import SparkSession, Window
from torch.utils.data import DataLoader

from code.lime_model import fit_lime, evaluate_model
import pyspark.sql.functions as F


# output_folder = 'results/sztuczne_Self-supervised-clustering__05-06-2023-23-47-33/'
output_folder = 'results/pecherz_07-10-2023-07-43/'

# path_to_data = "dane/sztuczne_dane/"
path_to_data = "dane/pecherz/"


spark = SparkSession.builder \
    .master("local") \
    .appName("Magisterka") \
    .config("spark.driver.memory", "30g") \
    .config("spark.driver.maxResultSize", "10g") \
    .getOrCreate()
spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, Path('.').absolute().as_uri())

predictions = np.load(output_folder + 'Normal_K-Means.npy')
clusters_arr = np.max(predictions, axis=0) + 1


def f(x):
    v = 0
    for j in range(len(x)):
        v *= clusters_arr[j]
        v += x[j]
    return int(v)


predictions = np.apply_along_axis(f, 1, predictions)

mz = np.load(path_to_data + 'root/mz.npy')
df = spark.read.parquet(path_to_data + 'parquet_norm_False_convolve_True')
df = df.withColumn('row_index', F.row_number().over(Window.orderBy(F.monotonically_increasing_id())))

converted_data = make_spark_converter(df)

batch_size = 64
hparams_lime = Namespace(
    lr=2e-4,
    batch_size=batch_size,
    epochs=500,
    patience=5,
    batch_step=df.count() // batch_size,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    in_dim=len(df.first().features),
    out_dim=len(np.unique(predictions)),
    weight=[1 / len(predictions[predictions == p]) for p in np.unique(predictions)],
)

# LIME LEARNING
'''
with converted_data.make_torch_dataloader(
    batch_size=hparams_lime.batch_size,
    shuffling_queue_capacity=1024
) as dataloader:
    model, _ = fit_lime(hparams_lime, output_folder, iter(dataloader), predictions)

with converted_data.make_torch_dataloader(
    batch_size=hparams_lime.batch_size,
    num_epochs=1
) as dataloader:
    acc = evaluate_model(model, iter(dataloader), hparams_lime.device, predictions)
    with open(output_folder + 'lime_norm_accs', 'w') as convert_file:
        convert_file.write('Accuracy: {}'.format(acc))
    print(f'Accuracy: {acc}')
'''


class Dataset:
    def __init__(self, data, epochs):
        self.data = data
        self.epochs = epochs

    def __len__(self):
        return len(self.data) * self.epochs

    def __getitem__(self, idx):
        return {'features': torch.Tensor(self.data[idx % len(self.data)]['features']),
                'row_index': self.data[idx % len(self.data)]['row_index']}


dataset = Dataset(df.rdd.collect(), 100000)
dataloader = DataLoader(dataset, batch_size=batch_size)
model, _ = fit_lime(hparams_lime, output_folder, iter(dataloader), predictions)

dataset = Dataset(df.rdd.collect(), 1)
dataloader = DataLoader(dataset, batch_size=batch_size)
acc = evaluate_model(model, iter(dataloader), hparams_lime.device, predictions)
with open(output_folder + 'lime_norm_accs', 'w') as convert_file:
    convert_file.write('Accuracy: {}'.format(acc))
print(f'Accuracy: {acc}')
# TRAINING END

weight = model.weight.detach().numpy()
for i in range(weight.shape[0]):
    df = pd.DataFrame(
        data={
            "mz": np.array(list(map(lambda x: f"%.2f" % x, mz))),
            "lime": weight[i, 5:-5]
        }
    )

    df = df.sort_values('lime', ascending=False)[:30]
    ax = df.plot.barh(x='mz', y='lime')
    ax.invert_yaxis()
    plt.savefig(output_folder + f'lime_class_{i}')
