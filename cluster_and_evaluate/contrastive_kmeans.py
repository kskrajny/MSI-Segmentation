import time
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import os
from sklearn.metrics import silhouette_score
from evaluate import evaluate
from utils.utils import DatasetName


current_dir = os.getcwd()
if not current_dir.endswith('MSI-Segmentation'):
    os.chdir(os.path.abspath(os.path.join(current_dir, "..")))
    print(f"Working directory changed to: {os.getcwd()}")
else:
    print(f"Working directory remains: {current_dir}")


def contrastive_kmeans(output_folders, max_clusters, n_components, do_iter=False):
    for output_folder in output_folders:
        start = time.time()

        # DATA
        CLR_feat = np.squeeze(np.load(output_folder + 'CLR_feat.npy'))
        print(CLR_feat.shape)

        # TSNE
        tsne_feat = TSNE(n_components=3, perplexity=1000).fit_transform(CLR_feat)
        np.save(output_folder + 'TSNE_encoding', tsne_feat)

        # PCA
        pca_feat = PCA(n_components=n_components).fit_transform(CLR_feat)
        np.save(output_folder + 'PCA_encoding', pca_feat)

        # feat = tsne_feat
        feat = pca_feat
        # feat = CLR_feat

        # NORMAL K-MEANS
        kmeans = KMeans(n_clusters=max_clusters)
        labels = kmeans.fit_predict(feat)


        np.save(output_folder + 'Normal_K-Means', np.expand_dims(labels, -1))
        with open(output_folder + 'Normal_time', 'w') as file:
            file.write(str(time.time() - start))

        if do_iter:
            # This part of code last much longer time than simple kmeans
            # ITERATIVE K-MEANS
            start = time.time()
            max_clusters = 3 # - nwm na co to
            
            clusters = 1
            labels_arr = []
            clusters_arr = []
            
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

            np.save(output_folder + 'Iterative_K-Means', labels)
            with open(output_folder + 'Iterative_time', 'w') as file:
                file.write(str(time.time() - start))


        for prefix in ['Normal_K-Means']:
            evaluate(prefix, output_folder)


def get_subdirectories(search_directory = 'results/', pattern = 'conv'):
    # Check if the directory exists
    if os.path.exists(search_directory):
        # Get a list of all directories within the search directory
        subdirectories = [subdir for subdir in os.listdir(search_directory)
                          if os.path.isdir(os.path.join(search_directory, subdir))]

        # Filter directories that contain '128' in their name
        filtered_subdirectories = [str(os.path.join(search_directory, subdir)) + '/'
                                   for subdir in subdirectories
                                   if pattern in subdir and 'original' not in subdir
                                   and 'True' in subdir]
        return filtered_subdirectories
    else:
        print(f"Directory {search_directory} does not exist.")
        return []


CONFIG = {
    DatasetName.bladder: {
        "output_folders": [
            'results/pecherz/pecherz_28-07-2024-21-43_conv_True/'
            # 'results/pecherz/pecherz_28-07-2024-23-26_conv_False/'
        ],
        "max_clusters": 12,
        "n_components": 6
    },
    DatasetName.artificial: {
        "output_folders": [
            'results/sztuczne_dane_26-07-2024-23-11_conv_True/',
            # 'results/sztuczne_dane_28-07-2024-22-06_conv_False/'
        ],
        "max_clusters": 6,
        "n_components": 3
    },
    DatasetName.liver: {
        "output_folders": [
            'results/watroba_28-07-2024-21-45_conv_False/',
            # 'results/watroba_28-07-2024-23-19_conv_True/',
            # 'results/watroba_29-07-2024-20-45_conv_False/',
            # 'results/watroba_29-07-2024-17-39_conv_True/'
        ],
        "max_clusters": 3,
        "n_components": 3
    },
    DatasetName.new: {
        # "output_folders": ["results/nowe_dane_3_J_23-10-2024-01-47_128_conv_True/"],
        "output_folders": get_subdirectories(),
        "max_clusters": 5,
        "n_components": 10
    }
}

if __name__ == '__main__':
    # config = CONFIG[DatasetName.artificial]
    # config = CONFIG[DatasetName.bladder]
    # config = CONFIG[DatasetName.liver]
    config = CONFIG[DatasetName.new]
    contrastive_kmeans(**config)
