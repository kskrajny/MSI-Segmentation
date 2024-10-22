import time

from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import numpy as np
from sklearn.neighbors import NearestNeighbors

from evaluate import evaluate

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


output_folders = [
    # 'results/nowe_dane_all_09-09-2024-15-20_conv_True/',
    # 'results/nowe_dane_all_09-09-2024-19-08_96_conv_True/',
    # 'results/nowe_dane_all_09-09-2024-21-02_72_conv_True/',
    # 'results/nowe_dane_all_09-09-2024-22-13_54_conv_True/',
    # 'results/nowe_dane_all_09-09-2024-23-14_36_conv_True/',
    # 'results/nowe_dane_all_10-09-2024-00-22_24_conv_True/'
    # 'results/nowe_dane_all_10-09-2024-01-23_16_conv_True/',
    # 'results/nowe_dane_all_10-09-2024-09-29_12_conv_True/',
    # 'results/nowe_dane_all_10-09-2024-10-40_2_conv_True/',
    # 'results/nowe_dane_all_10-09-2024-12-00_5_conv_True/',
    # 'results/nowe_dane_all_10-09-2024-14-44_8_conv_True/',
    # 'results/nowe_dane_all_10-09-2024-15-57_8_conv_False/',

    #'results/nowe_dane_P_01-10-2024-12-01_128_conv_False/',
    # 'results/nowe_dane_T_01-10-2024-19-44_128_conv_True/',
]
max_clusters = 8
n_components = 1
# ------------------------------------------------------------------- #

# DATA
CLR_feat = np.squeeze(np.load(output_folders[0] + 'CLR_feat.npy'))

# TSNE
# tsne_feat = TSNE(n_components=3, perplexity=100).fit_transform(CLR_feat)

# PCA
pca_feat = PCA(n_components=n_components).fit_transform(CLR_feat)

# feat = tsne_feat
feat = pca_feat
# feat = CLR_feat

# NORMAL K-MEANS
kmeans = KMeans(n_clusters=max_clusters)
labels = kmeans.fit_predict(feat)

for output_folder in output_folders:
    np.save(output_folder + 'Normal_K-Means', np.expand_dims(labels, -1))
    with open(output_folder + 'Normal_time', 'w') as file:
        file.write(str(time.time() - start))

# ITERATIVE K-MEANS
start = time.time()
max_clusters = 8 # - nwm na co to

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

for prefix in ['Iterative_K-Means', 'Normal_K-Means']:
    evaluate(prefix, output_folders[0])