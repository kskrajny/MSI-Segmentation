import os
import numpy as np
import matplotlib.pyplot as plt


current_dir = os.getcwd()
if not current_dir.endswith('MSI-Segmentation'):
    os.chdir(os.path.abspath(os.path.join(current_dir, "..")))
    print(f"Working directory changed to: {os.getcwd()}")
else:
    print(f"Working directory remains: {current_dir}")

if __name__ == "__main__":
    output_folder = 'results/pecherz_original_max128'
    for a in range(3):
        for b in range(a):
            for t in ["PCA", "TSNE", "TSNE1000"]:
                # Load numpy array from file
                data = np.load(output_folder + f"{t}_encoding.npy")

                # Scatter plot with density (hexbin)
                plt.figure(figsize=(8, 6))
                plt.hexbin(data[:, a], data[:, b], gridsize=50, cmap="viridis", alpha=0.7)
                plt.colorbar(label="Density")
                plt.xlabel("X-axis")
                plt.ylabel("Y-axis")
                plt.savefig(output_folder + f"{t}-density-dimensions-{a}-{b}.pdf", format="pdf")
                plt.close()
