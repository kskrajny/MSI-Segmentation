import os
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from code.utils import get_Acc, get_img, save_acc, get_conv_img
from code.prepare_liver_ground_truth import process_image

plt.set_cmap('jet')


def evaluate(prefix, data_folder=None):
    # ------------------------ VARIABLES TO SET  ------------------------ #
    # RESULT DIRECTORY

    # data_folder = 'results/pecherz_08-11-2023-18-29_conv_False_V2/'
    # data_folder = 'results/pecherz_08-11-2023-17-22_conv_True_V2/'

    # data_folder = 'results/pecherz_08-11-2023-18-29_conv_False_V5/'
    # data_folder = 'results/pecherz_08-11-2023-17-22_conv_True_V5/'

    # data_folder = 'results/pecherz_08-11-2023-18-29_conv_False_V4/'
    # data_folder = 'results/pecherz_08-11-2023-17-22_conv_True_V4/'

    # data_folder = 'results/pecherz_08-11-2023-18-29_conv_False_V1/'
    # data_folder = 'results/pecherz_08-11-2023-17-22_conv_True_V1/'

    # data_folder = 'results/pecherz_08-11-2023-18-29_conv_False_V3/'
    # data_folder = 'results/pecherz_08-11-2023-17-22_conv_True_V3/'

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

    # data_folder = 'results/sztuczne_dane_conv_True/'

    # data_folder = f'results/{dataset_name}_conv_True/'

    # data_folder = 'results/pecherz/pecherz_28-07-2024-23-26_conv_False_V5/'

    data_folder = 'results/pecherz/pecherz_28-07-2024-21-43_conv_True/' if not data_folder else data_folder
    t = data_folder[18]

    dataset_name = data_folder.split('/')[1].split('_')[0]
    output_folder = data_folder + prefix + '/'
    # ------------------------------------------------------------------- #

    try:
        os.mkdir(output_folder)
        print(output_folder)
    except Exception as e:
        print(e)

    # PREDICTIONS
    predictions = np.load(data_folder + f'{prefix}.npy') + 1
    clusters_arr = np.max(predictions, axis=0)

    def f(x):
        v = 0
        for j in range(len(x)):
            v *= clusters_arr[j]
            v += x[j]
        return int(v)

    gathered_predictions = np.apply_along_axis(f, 1, predictions)

    # COORDINATES
    if dataset_name == 'nowe':
        file_list = []
        for n in [1, 2, 3]:
            v = f'{t}_{n}'
            array = np.load(f'dane/{dataset_name}/root/filtered_coords_{v}.npy')
            file_list.append(array)
            cords = np.concatenate(file_list)
            unique_cords = np.unique(cords, axis=0)
        min_x = np.min(cords[:, 0])
        min_y = np.min(cords[:, 1])
        cords[:, 0] = cords[:, 0] - min_x
        cords[:, 1] = cords[:, 1] - min_y
    else:
        cords = np.load(f'dane/{dataset_name}/root/cords.npy')

    # GT LABELS
    gathered_label = None
    if dataset_name == 'sztuczne':
        label = pd.read_csv('dane/sztuczne_dane/root/labels.csv')
        # label = pd.read_csv('dane/sztuczne_dane/root/labels_origin.csv')
        gathered_label = label['L']
        # gathered_label = label['PA(44:0)'] #* 2 + label['']
    elif dataset_name == 'pecherz':
        #label = pd.read_csv('dane/pecherz/root/segment_2r_bladder.csv')
        #label.drop(columns=['Unnamed: 0'], inplace=True)
        #gathered_label = label['V1'] #V1, V4, V3
        #print(gathered_label.shape)
        from PIL import Image
        from sklearn.cluster import KMeans
        #import scipy
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

        '''
        label = pd.read_csv('dane/pecherz/root/segment_2r_bladder.csv')
        label.drop(columns=['Unnamed: 0'], inplace=True)
        # Mapping column names to numbers
        column_mapping = {'V1': 1, 'V2': 2, 'V3': 3, 'V4': 4, 'V5': 5}

        # Function to map column name to number
        def map_column_to_last_max(row):
            max_value = row.max()
            if max_value == 1:
                return 0
            else:
                # Get the last column where the max value occurs
                max_column = row[row == max_value].idxmax()
                return column_mapping[max_column]

        # Applying the function to each row and getting a list of mapped values
        # gathered_label = label[['V1', 'V2', 'V3', 'V4', 'V5']].apply(map_column_to_last_max, axis=1)
        gathered_label = (label['V1'] > 1) + 2 * (label['V4'] > 1) + 4 * (label['V2'] > 1)\
                         + 8 * (label['V3'] > 2) + 16 * (label['V5'] > 2)
        '''
    elif dataset_name == 'watroba':
        gathered_label = process_image('dane/watroba/root/ground_truth_08.png', (92, 161))
        cords[:, 0] = 161 - cords[:, 0] - 1
    elif dataset_name == 'nowe':
        file_list = []
        for n in [1, 2, 3]:
            v = f'{t}_{n}'
            file_list.append(
                np.load(f'dane/{dataset_name}/root/filtered_targets_3_{v}.npy')
            )
        gathered_label_origin = np.concatenate(file_list).flatten()
        print(
            np.unique(gathered_label_origin, return_counts=True)
        )
        gathered_label, gathered_predictions = delete_zero_label_rows(gathered_label_origin, gathered_predictions)

    np.savetxt(output_folder + "ground_truth.csv", gathered_label, delimiter=",", fmt="%d")

    # EVALUATE GATHERED PREDICTIONS
    acc, info = get_Acc(gathered_predictions, gathered_label)
    np.savetxt(output_folder + "prediction.csv", gathered_predictions, delimiter=",", fmt="%d")

    gathered_predictions_matched = np.array(list(map(lambda x: info[info.label == x].gt_group, gathered_predictions)))
    np.savetxt(output_folder + "matched.csv", gathered_predictions_matched, delimiter=",", fmt="%d")

    save_acc('Acc', acc, output_folder, 'w')

    img = get_img(cords, gathered_label)
    plt.imshow(img)
    '''
    im = plt.imshow(img)
    values = np.unique(img)
    colors = [im.cmap(im.norm(value)) for value in values]
    patches = [mpatches.Patch(color=colors[i], label="Level {l}".format(l=values[i])) for i in range(len(values))]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    '''
    plt.savefig(output_folder + 'true_seg')

    alg_img = get_img(cords, gathered_predictions)
    plt.imshow(alg_img)
    plt.savefig(output_folder + 'alg_seg')

    img_matched = get_img(cords, gathered_predictions_matched)
    plt.imshow(img_matched)
    plt.savefig(output_folder + 'alg_seg_match')

    # CONVOLUTION
    conv_alg_img = get_conv_img(alg_img)
    plt.imshow(conv_alg_img)
    plt.savefig(output_folder + 'conv_alg_seg')

    conv_alg_predictions = []
    for c in cords:
        conv_alg_predictions.append(conv_alg_img[c[0], c[1]])

    #gathered_label, conv_alg_predictions = delete_zero_label_rows(gathered_label_origin, np.array(conv_alg_predictions))

    acc, info = get_Acc(conv_alg_predictions, gathered_label)
    conv_predictions_matched = np.array(list(map(lambda x: info[info.label == x].gt_group, conv_alg_predictions)))
    save_acc('Conv Acc', acc, output_folder, 'a')

    conv_img_matched = get_img(cords, conv_predictions_matched)
    plt.imshow(conv_img_matched)
    plt.savefig(output_folder + 'conv_seg_matched')

    '''
    # SINGLE LABELS
    for i in range(predictions.shape[1]):
        img = get_img(cords, predictions[:, i])
        fig, axs = plt.subplots(1, len(label.columns) + 1)
        axs[0].imshow(img)
        for j, col in enumerate(label.columns):
            img = get_img(cords, label[col])
            axs[j + 1].imshow(img)
        fig.savefig(output_folder + f'single_{i}')
    
    v = 1
    gathered_label = np.zeros((label.shape[0]))
    for i in range(1, 6):
        gathered_label += (label[f'V{i}'] - 1) * v
        v *= max(label[f'V{i}'])
    
    for i in range(1, 6):
        lab = label[f'V{i}']
        img = get_img(cords, lab)
        plt.title(str(i))
        plt.imshow(img)
        plt.show()
    
    labels = []
    for cord in cords:
        label = 0 if cord[1] >= 20 else 2
        label += 1 if 30 > cord[0] >= 10 and 30 > cord[1] >= 10 else 0
        labels.append(label)
        pd.DataFrame({'L': labels}).to_csv('dane/sztuczne_dane/root/labels.csv')
    '''

def delete_zero_label_rows(gather_labels, gather_prediction):
    # Find indices where labels are not equal to 0
    non_zero_indices = np.where(gather_labels != 0)[0]
    print(gather_prediction.shape, non_zero_indices.shape)
    # Select only the rows from both arrays where label is non-zero
    gather_labels_filtered = gather_labels[non_zero_indices]
    gather_prediction_filtered = gather_prediction[non_zero_indices]
    return gather_labels_filtered, gather_prediction_filtered

if __name__ == '__main__':
    for prefix in ['Iterative_K-Means', 'Normal_K-Means']:
        evaluate(prefix)
