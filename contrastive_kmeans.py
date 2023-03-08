from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import numpy as np


# ------------------------ VARIABLES TO SET  ------------------------ #
# - output_folder
# - max_cluster
# - n_components

# output_folder = 'results/pecherz_08-11-2023-18-29_conv_False/'
output_folder = 'results/pecherz_08-11-2023-17-22_conv_True/'
max_clusters = 32
n_components = 8

# output_folder = 'results/sztuczne_dane_08-11-2023-11-16_conv_False/'
# output_folder = 'results/sztuczne_dane_08-11-2023-10-51_conv_True/'
# max_clusters = 6
# n_components = 3

# output_folder = 'results/watroba_08-13-2023-11-02_conv_True/'
# output_folder = 'results/watroba_08-13-2023-19-14_conv_False/'
# max_clusters = 3
# n_components = 3
# ------------------------------------------------------------------- #


# DATA
CLR_feat = np.squeeze(np.load(output_folder + 'CLR_feat.npy'))
'''
for i in range(1600):
    plt.plot(CLR_feat[i])
plt.show()
'''
print(CLR_feat.shape)


# PCA
pca = PCA(n_components=n_components)
pca_feat = pca.fit_transform(CLR_feat)


# NORMAL K-MEANS
kmeans = KMeans(n_clusters=max_clusters)
labels = kmeans.fit_predict(pca_feat)
np.save(output_folder + 'Normal_K-Means', np.expand_dims(labels, -1))


# ITERATIVE K-MEANS
# max_clusters = 8

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
    if max(scores) > 0.55:
        k = list(r)[int(np.argmax(scores))]
        kmeans = KMeans(n_clusters=k)
        labels_arr.append(kmeans.fit_predict(pca_feat[:, [i]])) #i:
        clusters *= k
        clusters_arr.append(k)
    if max_clusters <= clusters:
        break

labels = np.transpose(np.stack(labels_arr), (1, 0))
np.save(output_folder + 'Iterative_K-Means', labels)
