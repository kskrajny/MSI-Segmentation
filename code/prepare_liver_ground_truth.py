import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def process_image(png_path, new_shape):
    # 1. Read the image
    img = Image.open(png_path)
    cords = np.load(f'dane/watroba/root/cords.npy')

    # 2. Resize the image
    img_resized = img.resize(new_shape)

    # Convert to numpy array
    data = np.array(img_resized)[:, :, :3]

    # Create an empty image for clustering
    clustered_img = np.zeros_like(data[:, :, 0])

    # 3. Cluster the image based on colors
    '''
    t = 200
    condition1 = (data[:, :, 0] > t) & (data[:, :, 1] > t) & (data[:, :, 2] > t)
    clustered_img[condition1] = 0
    t = 150
    condition2 = data[:, :, 2] > t
    clustered_img[condition2] = 1
    '''

    # ground truth old, red / green
    t = 100
    condition = ((data[:, :, 0] > t) | (t > data[:, :, 2]))
    clustered_img[condition] = 1

    '''
    # 4. Plot the images
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(img_resized)
    plt.title('Source Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(clustered_img)
    plt.title('Generated ground truth labels')
    plt.axis('off')
    '''

    plt.tight_layout()
    plt.show()
    labels = []
    for c in cords:
        labels.append(clustered_img[161 - c[0] - 1, c[1]])
    return np.array(labels)


if __name__ == '__main__':
    process_image('../dane/watroba/root/ground_truth_08.png', (92, 161))