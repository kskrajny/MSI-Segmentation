import numpy as np
import matplotlib.pyplot as plt

# Load the concatenated numpy array
if __name__ == '__main__':
    arr_list = []
    for n in [1, 2, 3]:
        array = np.load(f'filtered_coords_3_T_{n}.npy')
        arr_list.append(array)
    cords = np.concatenate(arr_list)

    # Ensure cords is a 2D array with shape (len, 2)
    if cords.shape[1] != 2:
        raise ValueError("Expected cords array with shape (len, 2)")

    # Load CSV files for labels
    prediction_df = np.loadtxt('prediction.csv', delimiter=",", dtype=int)
    matched_df = np.loadtxt('matched.csv', delimiter=",", dtype=int)
    ground_truth_df = np.loadtxt('ground_truth.csv', delimiter=",", dtype=int)

    for labels, name in zip([prediction_df, matched_df, ground_truth_df], ["pred", "matched", "gt"]):
        # Check if labels have the same length as cords
        print(labels.shape, cords.shape)
        if len(labels) != len(cords):
            raise ValueError("Labels and cords must have the same length")

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(cords[:, 0], cords[:, 1], c=labels, cmap="viridis", alpha=0.7, marker='o')

        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.title(name)
        plt.savefig(f"{name}_plot.png", dpi=300)  # Save plot as image

    # Create comparison plot between ground truth and matched
    comparison_labels = (ground_truth_df == matched_df).astype(int)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(cords[:, 0], cords[:, 1], c=comparison_labels, cmap="coolwarm", alpha=0.7, marker='o')
    plt.colorbar(scatter, label="Match (1) or Mismatch (0)")

    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Ground Truth vs Matched Comparison")
    plt.savefig("comparison_plot.png", dpi=300)  # Save comparison plot