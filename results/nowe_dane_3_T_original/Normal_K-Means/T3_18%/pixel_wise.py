import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the concatenated numpy array
if __name__ == '__main__':
    arr_list = []
    for n in [1, 2, 3]:
        array = np.load(f'filtered_coords_3_T_{n}.npy')
        arr_list.append(array)
    cords = np.concatenate(arr_list)
    x_max, y_max = cords.max(axis=0) + 1  # assuming coordinates start from 0

    # Ensure cords is a 2D array with shape (len, 2)
    if cords.shape[1] != 2:
        raise ValueError("Expected cords array with shape (len, 2)")

    # Load CSV files for labels
    prediction_df = np.loadtxt('prediction.csv', delimiter=",", dtype=int)
    matched_df = np.loadtxt('matched.csv', delimiter=",", dtype=int)
    ground_truth_df = np.loadtxt('ground_truth.csv', delimiter=",", dtype=int)

    for labels, name in zip([prediction_df, matched_df, ground_truth_df], ["pred", "matched", "gt"]):
        # Check if labels have the same length as cords
        if len(labels) != len(cords):
            raise ValueError("Labels and cords must have the same length")

        # Initialize an empty image and fill it with label values at the specified coordinates
        image = np.zeros((x_max, y_max), dtype=int)
        for idx, (x, y) in enumerate(cords):
            image[x, y] = labels[idx]

        # Plot the black and white image without colorbar
        plt.figure(figsize=(10, 8))
        plt.imshow(image.T, origin="lower")  # Transpose to align axes correctly
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.title(name)
        plt.savefig(f"{name}_pixel_plot.png")  # Save pixel-wise image

    # Create comparison plot between ground truth and matched
    comparison_labels = (ground_truth_df == matched_df).astype(int)

    # Define the resolution of the output image (based on cords' min and max values)
    x_max, y_max = cords.max(axis=0) + 1  # assuming coordinates start from 0
    image = np.zeros((x_max, y_max), dtype=int)

    # Populate the image based on the comparison results
    for idx, (x, y) in enumerate(cords):
        image[x, y] = comparison_labels[idx]

    # Plot the black and white image
    plt.figure(figsize=(10, 8))
    plt.imshow(image.T, cmap="gray", origin="lower")  # Transpose to align axes correctly

    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Ground Truth vs Matched Comparison (Pixel-wise)")
    plt.savefig("comparison_pixel_plot.png")