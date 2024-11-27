# MSI-self-supervised-clustering


## Workflow for Developers

---

### Install Dependencies
Python 3.10
```bash
pip install -r requirements.txt
```

---

### Prepare Dataset

#### Datasets Used in Papers
Here is a link to
[data](https://drive.google.com/drive/folders/14cli_aVFAocVRCBk0GRllIJwUyj4OTOu?usp=sharing)
user in research.

Choose from:
- nowe (biggest)
- pecherz
- sztuczne (smallest)
- watroba

To run algorithm on them download into `./data/{name}/root/`
and run [combine_preprocess](preprocessing/combine_preprocess.py) with relevant parameters.

NOTICE: Before running this check if you need parquet or just numpy. \
For small datasets use numpy, for big ones use parquet. \
Numpy is preferable, but some notebooks are using parquet.

#### Custom Dataset
In case you want to use custom data, save your data in numpy format like in the example below. \
Place them inside `dane/[custom_name]/root/` directory. \
For real examples please see some code in [`preprocessing`](preprocessing/) 

```
import numpy as np

# Coordinates of pixels in the 2d image
cords = np.array([
    [0, 0],
    [0, 1]
])

# Example m/z values
# IMPORTANT: increasing order and same difference between next elements should be preserved
mz = np.array([
    100, 100.1, 100.2, 100.3, 100.4, 100.5,
])

# Example intensities at specific m/z values
intensities = np.array([
    [0, 75.3, 0.0, 0, 55.2, 72.43],
    [0, 72.3, 0.0, 0, 10.4, 13.01]
])

np.save('data/custom/root/cords.npy', cords)
np.save('data/custom/root/mz.npy', mz)
np.save('data/custom/root/intsy.npy', intensities)
```

After that run [convolve_and_save](preprocessing/convolve_and_save.py) on new data.

---

### Encoding Algorithm

To train encoder use one of notebooks provided in [notebooks](notebooks).
They are supposed to be run on Google Colab, so if you want to train locally,
change paths to data, result directory and comment the lines that try to connect to Google Drive.

Notice that dataloader may use either numpy array or parquet as data source.

To train on new data set properly dimensions of a network.
To make it easier, here is the [script](notebooks/suggest_network_parameters.py)
that suggest possible network parameters.

---

### Clustering and Evaluation

To create image clusters run functions from [custer_and_evaluate](cluster_and_evaluate) directory. \
For raw data without applying encoding algorithm use
[original_kmeans.py](cluster_and_evaluate/original_kmeans.py). \
For encoding algorithm results use [contrastive_kmeans.py](cluster_and_evaluate/contrastive_kmeans.py).

Clustering valuation is by default performed after clustering within these scripts. \
In case one writes custom script use [evaluate.py](cluster_and_evaluate/evaluate.py) after clustering.

To analyze clusters visually you can see not only clustered images but also visualize encodings
after applying TSNE or PCA on them. For that reason see
[plot_features_2d.py.py](cluster_and_evaluate/plot_features_2d.py).

---