import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_losses(n_iterations, loss, xlabel, ylabel, title, savedir=None):
    iterations = np.linspace(1, n_iterations, n_iterations).astype(int)
    plt.figure(figsize=(10, 10))
    plt.plot(iterations, loss)
    plt.title(title, fontsize=30)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    if savedir is not None:
        plt.savefig(savedir, dpi=300)
    else:
        plt.show()
    plt.close()


def get_Acc(test_labels, gt_labels):
    labels_uni, labels_count = np.unique(test_labels, return_counts=True)
    group_counts = []
    accs = []
    gt_group = []

    for i in range(labels_uni.shape[0]):
        
        # 1. for each cluster, get all spectra
        label = labels_uni[i]
        count = labels_count[i]
        idx = test_labels == label
        gt_label = gt_labels[idx]  # ground truth labels for each spectrum in a group

        # 3. compute acc in this group
        values, counts = np.unique(gt_label, return_counts=True)
        pr = np.divide(counts, [len(gt_labels[gt_labels == v]) for v in values])
        maj_vote = values[np.argmax(pr)]
        n_vote = np.sum(gt_label == maj_vote)
        acc = n_vote/gt_label.shape[0]
        gt_group.append(values[np.argmax(pr)])

        # 4. store data
        group_counts.append(count)
        accs.append(acc)

    # organize results
    group_counts = np.array(group_counts).reshape(-1, 1)
    accs = np.array(accs).reshape(-1, 1)
    group_info = np.hstack((group_counts, accs))
    # 1st data, may have nan scores when 1 img in 1 group
    df_group_info = pd.DataFrame(group_info, columns=['group_counts', 'acc'])
    group_weights = (df_group_info['group_counts']/np.sum(df_group_info['group_counts'])).values.copy()
    df_group_info['group_weights'] = group_weights
    df_group_info['gt_group'] = gt_group
    df_group_info['label'] = labels_uni
    final_score = np.round(np.sum(group_weights*df_group_info.values[:, 1], axis=0), 4)
    return final_score, df_group_info


def get_img(cords, label):
    img = np.zeros((cords[:, 0].max() + 1, cords[:, 1].max() + 1))
    for c, l in zip(cords, label):
        img[c[0], c[1]] = int(l)
    return img


def save_acc(name, acc, output_folder, mode):
    print(f'{name}: {acc}')
    with open(output_folder + 'accs', mode) as convert_file:
        convert_file.write(f'{name}: {acc}\n')


def get_conv_img(img):
    new_img = np.copy(img)
    kernel_size = 3
    for i in range(kernel_size // 2, img.shape[0] - kernel_size // 2):
        for j in range(kernel_size // 2, img.shape[1] - kernel_size // 2):
            vals = img[
                   i - kernel_size // 2:i + 1 + kernel_size // 2,
                   j - kernel_size // 2:j + 1 + kernel_size // 2
                   ].flatten().astype(int)
            counts = np.bincount(vals)
            new_img[i, j] = np.argmax(counts)
    return new_img


def max_euclidean_distance(samples):
    n_samples = samples.shape[0]
    max_dist = 0

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            dist = np.linalg.norm(samples[i] - samples[j])
            max_dist = max(max_dist, dist)

    return max_dist
