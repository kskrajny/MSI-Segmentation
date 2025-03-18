# Mass Spectrometry Imaging self-supervised clustering

This repository contains computer codes for a self-supervised segmentation of Mass Spectrometry Imaging data, as described in our publication titled *Efficient compression of mass spectrometry images via contrastive learning-based encoding*. 

## Dependencies 
To run the codes, you need a working installation of the Python 3.10 programming language. 
To install the necessary libraries, clone this repository and run 
```bash
pip install -r requirements.txt
```
This will automatically install all the necessary libraries listed in the file `requirements.txt` located in the main folder of this repository. 

Furthermore, if you need to run the analysis on larger data sets, you may need to install Parquet. In particular, Parquet is used in some of the notebooks provided in this repository which process large data sets. For smaller data sets, the `numpy` library (installed with the command above) is sufficient.   
## Reproducing the results from the publication
In this section, we describe how to reproduce the results published in *Efficient compression of mass spectrometry images via contrastive learning-based encoding*. We segment four data sets of MS images, referred to as:   
- *sztuczne* (the smallest data set, appropriate for a first run),
- *pecherz*,
- *nowe* (the biggest data set, appropriate for a thorough testing),
- *watroba*.
All the data sets can be found on our Google drive under [this link](https://drive.google.com/drive/folders/14cli_aVFAocVRCBk0GRllIJwUyj4OTOu?usp=sharing). 

To analyze any of these data sets, download the respective data set into `./data/{name}/root/`, where `.` refers to the directory of this repository and `{name}` refers to the name of the dataset.  
### Data pre-processing 
To preprocess a data set, run the `combine_preprocess.py` script found in the `preprocessing` subdirectory (you can follow [this link](https://github.com/kskrajny/MSI-Segmentation/blob/master/preprocessing/combine_preprocess.py) to access the script directly). 

### The encoding algorithm (data compression)
To train the encoder on any of the data sets listed above, use a respectively named notebook from the `notebooks` subdirectory (e.g. use the `notebooks/pecherz.ipynb` to analyze the *pecherz* data set).  
The notebooks are designed to be run on the Google Colab platform.  
If you want to train the encoder locally on your computer, modify the notebooks to change the paths to the data and the result directory appropriately. Furthermore, comment out the lines that connect to Google Drive.

Notice that, depending on the notebook, `dataloader` may use either `numpy` array or Parquet as data source (see the remark at the end of the Dependencies section).

### Clustering and evaluation
The `cluster_and_evaluate` directory contains two implementations of clustering algorithms to be used either prior to, or after encoding.  
The `original_kmeans.py` script is designed for raw data without applying the encoding algorithm. This clustering can be used as a baseline to evaluate the impact of encoding on clustering.  
To cluster encoded data, use the `contrastive_kmeans.py` script. 

By default, clustering evaluation is performed automatically after clustering within these scripts. 
In order to evaluate a custom clustering script (e.g. if you want to compare the performance of other clustering algorithms), use `evaluate.py` after clustering.

To analyze the results of the clustering, you can visualize the segmented MS images and visualize the encodings using tSNE or PCA. The appropriate functions are implemented in the `plot_features_2d.py` script in the `cluster_and_evaluate` directory.

## Segmenting a custom data set

### Preparing a custom data set
In case you want to use custom data, you will need to save your data in a `numpy` format as follows. First, create an array of coordinates of pixels. The code below shows an example for a 2x3 pixel image:
```
coords = np.array([
    [0, 0],
    [0, 1], 
    [0, 2],
    [1, 0],
    [1, 1],
    [1, 2]
])
```

Next, create a common array of m/z values for all pixels. It is important that the m/z values are specified in an increasing order with a constant difference between consecutive m/z values, like in this example for a m/z array with 6 values:
```
mz = np.array([
    100, 100.1, 100.2, 100.3, 100.4, 100.5,
])
```

Next, create an array of signal intensities, with rows corresponding to pixels and columns corresponding to m/z values from the array created in the previous step. The rows need to correspond to coordinates in the `coord` variable.   
```
intensities = np.array([
    [0.0, 75.3, 0.0, 0.0, 55.2, 72.43],
    [0.0, 72.3, 0.0, 0.0, 10.4, 13.01],
    [0.1, 72.4, 0.0, 0.0, 20.1, 11.20]
])
```

Next, save the variables `coords`, `mz` and `intsy` as three respectively named `numpy` arrays in a subdirectory of the `data` folder, named according to the name of the new data set. The code below will save the data as a new data set called `custom`. 
```
np.save('data/custom/root/cords.npy', cords)
np.save('data/custom/root/mz.npy', mz)
np.save('data/custom/root/intsy.npy', intensities)
```

Finally, combine and pre-process the `numpy` arrays by running the `convolve_and_save.py` script from the `preprocessing` subdirectory. Modify the `CONFIG` variable in the script to include the new data set and optionally specify additional parameters. The variable already contains example sets of parameters which you can use as a template. To enable the processing of the new data set, add a new line at the end of the script after `if __name__=="__main__"` according to the examples in the script. 

These steps will prepare the new data set, which you can process according to the steps described in Section *Reproducing the results from the publication*. 

### Tuning the Encoding Algorithm
To train the encoding algorithm on a new data set, it is important to specify the dimensions of the neural network. The script `notebooks/suggest_network_parameters.py` contains functions that can suggest possible sets of network parameters.  
