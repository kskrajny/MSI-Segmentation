import os
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from code.utils import get_Acc, get_img, save_acc, get_conv_img
from code.prepare_liver_ground_truth import process_image

plt.set_cmap('jet')

# ------------------------ VARIABLES TO SET  ------------------------ #
# dataset_name = 'watroba'
dataset_name = 'pecherz'
# dataset_name = 'sztuczne_dane'

# prefix = 'Iterative'
prefix = 'Normal'


# RESULT DIRECTORY
# data_folder = 'results/pecherz_08-11-2023-18-29_conv_False/'
# data_folder = 'results/pecherz_08-11-2023-17-22_conv_True/'
# data_folder = 'results/pecherz_original_conv/'
data_folder = 'results/pecherz_original/'

# data_folder = 'results/sztuczne_dane_original_conv/'
# data_folder = 'results/sztuczne_dane_original/'
# data_folder = 'results/sztuczne_dane_08-11-2023-10-51_conv_True/'
# data_folder = 'results/sztuczne_dane_08-11-2023-11-16_conv_False/'

# data_folder = 'results/watroba_original_conv/'
# data_folder = 'results/watroba_original/'
# data_folder = 'results/watroba_08-13-2023-11-02_conv_True/'
# data_folder = 'results/watroba_08-13-2023-19-14_conv_False/'
output_folder = data_folder + prefix + '/'

# ------------------------------------------------------------------- #

try:
    os.mkdir(output_folder)
    print(output_folder)
except Exception as e:
    print(e)

# PREDICTIONS
predictions = np.load(data_folder + f'{prefix}_K-Means.npy')
clusters_arr = np.max(predictions, axis=0) + 1


def f(x):
    v = 0
    for j in range(len(x)):
        v *= clusters_arr[j]
        v += x[j]
    return int(v)


gathered_predictions = np.apply_along_axis(f, 1, predictions)

# COORDINATES
cords = np.load(f'dane/{dataset_name}/root/cords.npy')


# GT LABELS
gathered_label = None
if dataset_name == 'sztuczne_dane':
    label = pd.read_csv('dane/sztuczne_dane/root/labels.csv')
    label = pd.read_csv('dane/sztuczne_dane/root/labels_origin.csv')
    print(label)
    # gathered_label = label['L']
    gathered_label = label['PA(44:0)']
elif dataset_name == 'pecherz':
    label = pd.read_csv('dane/pecherz/root/segment_2r_bladder.csv')
    label.drop(columns=['Unnamed: 0'], inplace=True)
    gathered_label = label['V4'] #V1, V4, V3
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
    gathered_label = process_image('dane/watroba/root/liver_acc_prep.png', (92, 161))
    cords[:, 0] = 161 - cords[:, 0] - 1


# EVALUATE GATHERED PREDICTIONS
acc, info = get_Acc(gathered_predictions, gathered_label)
gathered_predictions_matched = np.array(list(map(lambda x: info[info.label == x].gt_group, gathered_predictions)))
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
