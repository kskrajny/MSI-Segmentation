import os
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from utils.utils import get_Acc, get_img, save_acc, get_conv_img, delete_zero_label_rows
from prepare_targets.prepare_liver_ground_truth import process_image
from PIL import Image
from sklearn.cluster import KMeans

plt.set_cmap('jet')

current_dir = os.getcwd()
if not current_dir.endswith('MSI-Segmentation'):
    os.chdir(os.path.abspath(os.path.join(current_dir, "..")))
    print(f"Working directory changed to: {os.getcwd()}")
else:
    print(f"Working directory remains: {current_dir}")

# data_folder = 'results/pecherz_08-11-2023-18-29_conv_False/'
# data_folder = 'results/pecherz_28-07-2024-21-43_conv_True/'
# data_folder = 'results/pecherz_original/'

# data_folder = 'results/sztuczne_dane_original_conv/'
# data_folder = 'results/sztuczne_dane_original/'
# data_folder = 'results/sztuczne_dane_26-07-2024-23-11_conv_True/'
# data_folder = 'results/sztuczne_dane_28-07-2024-22-06_conv_False/'

# data_folder = 'results/watroba_original_conv/'
# data_folder = 'results/watroba_original/'
# data_folder = 'results/watroba_28-07-2024-23-19_conv_True/'
# data_folder = 'results/watroba_28-07-2024-21-45_conv_False/'

data_folder = 'results/pecherz_original/'


def evaluate(prefix, data_folder, c=""):
    # ------------------------------------------------------------------- #
    # PREPARE PARAMETERS AND RESULT DIRECTORY

    t = data_folder[18] if not (c == '_3') else data_folder[20]
    n = data_folder[20] if not (c == '_3') else data_folder[22]

    dataset_name = data_folder.split('/')[1].split('_')[0]
    output_folder = data_folder + prefix + '/'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print(f'NOW EVALUATE {output_folder}')

    # ------------------------------------------------------------------- #
    # READ PREDICTIONS

    predictions = np.load(data_folder + f'{prefix}.npy') + 1
    clusters_arr = np.max(predictions, axis=0)

    def f(x):
        v = 0
        for j in range(len(x)):
            v *= clusters_arr[j]
            v += x[j]
        return int(v)

    gathered_predictions = np.apply_along_axis(f, 1, predictions)


    # ------------------------------------------------------------------- #
    # READ COORDINATES

    if dataset_name == 'nowe':
        v = f'{t}_{n}'
        cords = np.load(f'dane/{dataset_name}/root/filtered_coords{c}_{v}.npy')
        unique_cords = np.unique(cords, axis=0)
        min_x = np.min(cords[:, 0])
        min_y = np.min(cords[:, 1])
        cords[:, 0] = cords[:, 0] - min_x
        cords[:, 1] = cords[:, 1] - min_y
    else:
        cords = np.load(f'dane/{dataset_name}/root/cords.npy')


    # ------------------------------------------------------------------- #
    # GET LABELS

    gathered_label = None

    if dataset_name == 'sztuczne':
        label = pd.read_csv('dane/sztuczne_dane/root/labels.csv')
        # label = pd.read_csv('dane/sztuczne_dane/root/labels_origin.csv')
        gathered_label = label['L']
        # gathered_label = label['PA(44:0)'] #* 2 + label['']

    elif dataset_name == 'pecherz':
        image_path = 'dane/pecherz/root/pecherz_gt.bmp'
        image = Image.open(image_path)
        # Convert the image to a NumPy array (ignore the alpha channel)
        image_array = np.array(image)
        rgb_image_array = image_array[:, :, :3]  # Exclude the alpha channel
        # Reshape the array to have pixels as rows and RGB values as columns
        pixels = rgb_image_array.reshape(-1, 3)
        # Set the number of clusters (7 in this case)
        n_clusters = 7
        # Perform KMeans clustering on the RGB colors
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(pixels)
        # Map each pixel to the closest cluster center (this returns a 1D array of labels)
        gathered_label = kmeans.predict(pixels)


    elif dataset_name == 'watroba':
        gathered_label = process_image('dane/watroba/root/ground_truth_08.png', (92, 161))
        cords[:, 0] = 161 - cords[:, 0] - 1

    elif dataset_name == 'nowe':
        v = f'{t}_{n}'
        gathered_label_origin = np.load(f'dane/{dataset_name}/root/filtered_targets{c}_{v}.npy')
        gathered_label, gathered_predictions = delete_zero_label_rows(gathered_label_origin, gathered_predictions)


    # ------------------------------------------------------------------- #
    # MATCH PREDICTIONS TO CLUSTERS

    acc, info = get_Acc(gathered_predictions, gathered_label)
    gathered_predictions_matched = np.array(list(map(lambda x: info[info.label == x].gt_group, gathered_predictions)))


    # ------------------------------------------------------------------- #
    # PLOT GROUND TRUTH

    img = get_img(cords, gathered_label)
    np.save(output_folder + f"target{c}", img)
    plt.imshow(img)
    plt.savefig(output_folder + f'true_seg{c}')


    # ------------------------------------------------------------------- #
    # PLOT RESULTS

    alg_img = get_img(cords, gathered_predictions)
    plt.imshow(alg_img)
    plt.savefig(output_folder + f'alg_seg{c}')

    img_matched = get_img(cords, gathered_predictions_matched)
    np.save(output_folder + f"alg{c}", img_matched)
    plt.imshow(img_matched)
    plt.savefig(output_folder + f'alg_seg_match{c}')


    # ------------------------------------------------------------------- #
    # PLOT CONVOLUTION RESULTS

    conv_alg_img = get_conv_img(alg_img)
    plt.imshow(conv_alg_img)
    plt.savefig(output_folder + f'conv_alg_seg{c}')

    conv_alg_predictions = []
    for c_ in cords:
        conv_alg_predictions.append(conv_alg_img[c_[0], c_[1]])

    if dataset_name == 'nowe':
        conv_gathered_label, conv_alg_predictions = delete_zero_label_rows(gathered_label_origin, np.array(conv_alg_predictions))

    acc_conv, info_conv = get_Acc(conv_alg_predictions, gathered_label)
    conv_predictions_matched = np.array(list(map(lambda x: info_conv[info_conv.label == x].gt_group, conv_alg_predictions)))

    conv_img_matched = get_img(cords, conv_predictions_matched)
    np.save(output_folder + f"alg_conv{c}", conv_img_matched)
    plt.imshow(conv_img_matched)
    plt.savefig(output_folder + f'conv_seg_matched{c}')


    # ------------------------------------------------------------------- #
    # SAVE NON-IMAGE RESULTS

    save_acc('Acc', acc, output_folder, 'w', c)
    save_acc('Conv Acc', acc_conv, output_folder, 'a', c)

    np.savetxt(output_folder + f"ground_truth{c}.csv", gathered_label, delimiter=",", fmt="%d")
    np.savetxt(output_folder + f"prediction{c}.csv", gathered_predictions, delimiter=",", fmt="%d")
    np.savetxt(output_folder + f"matched{c}.csv", gathered_predictions_matched, delimiter=",", fmt="%d")




if __name__ == '__main__':
    for prefix in ['Normal_K-Means']:
        evaluate(prefix, data_folder)
