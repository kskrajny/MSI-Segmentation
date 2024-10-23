import time

from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import numpy as np
from sklearn.neighbors import NearestNeighbors

from evaluate import evaluate

import os

# Define the base directory to search
search_directory = 'results/'
pattern = '128'

# Check if the directory exists
if os.path.exists(search_directory):
    # Get a list of all directories within the search directory
    subdirectories = [subdir for subdir in os.listdir(search_directory)
                      if os.path.isdir(os.path.join(search_directory, subdir))]

    # Filter directories that contain '128' in their name
    filtered_subdirectories = [str(os.path.join(search_directory, subdir)) + '/'
                               for subdir in subdirectories if pattern in subdir]

    # Print or return the list of filtered subdirectories
    print(filtered_subdirectories)
else:
    print(f"Directory {search_directory} does not exist.")

start = time.time()
# ------------------------ VARIABLES TO SET  ------------------------ #
# - output_folder
# - max_cluster
# - n_components

# output_folders = [
    # 'results/pecherz/pecherz_28-07-2024-21-43_conv_True/'
    # 'results/pecherz/pecherz_28-07-2024-23-26_conv_False/'
# ]
# ax_clusters = 12
# n_components = 6

# output_folders = [
    # 'results/sztuczne_dane_26-07-2024-23-11_conv_True/',
    # 'results/sztuczne_dane_28-07-2024-22-06_conv_False/'
# ]
# max_clusters = 6
# n_components = 3

# output_folders = [
    # 'results/watroba_28-07-2024-21-45_conv_False/',
    # 'results/watroba_28-07-2024-23-19_conv_True/'
    # 'results/watroba_29-07-2024-20-45_conv_False/'
    # 'results/watroba_29-07-2024-17-39_conv_True/'
# ]
# max_clusters = 3
# n_components = 3

# output_folders = ["results/sztuczne_dane_conv_True/"]
# max_clusters = 6
# n_components = 3

output_folders = filtered_subdirectories
output_folders = ['results/nowe_dane_3_J_23-10-2024-01-47_128_conv_True/']
max_clusters = 5
n_components = 10
# ------------------------------------------------------------------- #
for output_folder in output_folders:
    # DATA
    CLR_feat = np.squeeze(np.load(output_folder + 'CLR_feat.npy'))

    # TSNE
    # tsne_feat = TSNE(n_components=3, perplexity=100).fit_transform(CLR_feat)

    # PCA
    pca_feat = PCA(n_components=n_components).fit_transform(CLR_feat)

    # feat = tsne_feat
    feat = pca_feat
    #feat = CLR_feat

    # NORMAL K-MEANS
    kmeans = KMeans(n_clusters=max_clusters)
    labels = kmeans.fit_predict(feat)


    np.save(output_folder + 'Normal_K-Means', np.expand_dims(labels, -1))
    with open(output_folder + 'Normal_time', 'w') as file:
        file.write(str(time.time() - start))
    '''
    # ITERATIVE K-MEANS
    start = time.time()
    max_clusters = 3 # - nwm na co to
    
    clusters = 1
    labels_arr = []
    clusters_arr = []
    i = 0
    
    for i in range(n_components):
        print(clusters)
        scores = []
        r = range(2, 5)
        for k in r:
            kmeans = KMeans(n_clusters=k)
            labels = kmeans.fit_predict(pca_feat[:, [i]]) # i:
            scores.append(silhouette_score(pca_feat[:, [i]], labels)) #i:
        print(max(scores))
        if max(scores) > 0.54:
            k = list(r)[int(np.argmax(scores))]
            kmeans = KMeans(n_clusters=k)
            labels_arr.append(kmeans.fit_predict(pca_feat[:, [i]])) #i:
            clusters *= k
            clusters_arr.append(k)
        if max_clusters <= clusters:
            break
    
    labels = np.transpose(np.stack(labels_arr), (1, 0))
    
    for output_folder in output_folders:
        np.save(output_folder + 'Iterative_K-Means', labels)
        with open(output_folder + 'Iterative_time', 'w') as file:
            file.write(str(time.time() - start))
    '''
    for prefix in ['Normal_K-Means']:
        evaluate(prefix, output_folder)