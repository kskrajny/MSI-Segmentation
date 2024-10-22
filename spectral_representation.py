import os
import hashlib
from pyspark.ml.linalg import DenseVector, VectorUDT
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import gc
from pyspark.sql.functions import col, udf
from pyspark.sql.types import ArrayType, DoubleType
import pyspark.sql.functions as F


def calculate_spectral_representation(df, num_points=100, output_folder=None, cache_dir='cache/'):
    # Ensure cache directory exists
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Compute a hash based on the schema and parameters to cache the result
    cache_key = hashlib.md5(f'{df.schema}{num_points}'.encode()).hexdigest()
    cache_file = os.path.join(cache_dir, f'spectral_representation_{cache_key}.npy')

    # Check if the cached file exists and load it
    if os.path.exists(cache_file):
        print("Loading cached spectral representation...")
        top_indices = np.load(cache_file)
    else:
        print("Computing spectral representation...")

        # Collect the 'features' column (assumes DenseVector type)
        spectra = np.array(df.select("features").rdd.map(lambda row: row.features.toArray()).collect())

        # Compute the mean of each point across the entire dataset
        spectra_mean = np.mean(spectra, axis=0)

        # Select the indices of the top `num_points` based on the highest means
        top_indices = np.argsort(spectra_mean)[-num_points:]

        # Cache the result for future use
        np.save(cache_file, top_indices)

        visualize_selected_points_on_mean_spectrum(spectra_mean, top_indices, num_points, output_folder)

    # Define a UDF to filter the spectral vectors and keep only the selected points
    def select_top_points(spectrum):
        return DenseVector([spectrum[i] for i in top_indices])

    select_udf = udf(select_top_points, VectorUDT())
    # Apply the UDF to modify the 'features' column in the DataFrame
    df = df.withColumn("features", select_udf(col("features")))

    return df


def visualize_spectral_selection(df, num_points=100, output_folder=None):
    # Convert DenseVector to list using UDF and collect it as a NumPy array
    df = df.withColumn("features_array", F.udf(lambda v: v.toArray().tolist(), F.ArrayType(DoubleType()))(df["features"]))

    # Collect the 'features_array' column from the DataFrame as NumPy array
    selected_spectra = np.array(df.select("features_array").rdd.map(lambda row: row['features_array']).collect())

    # Compute the mean of the original spectra across the dataset (assuming df is already modified)
    mean_spectra = np.mean(selected_spectra, axis=0)

    # Plot the selected spectral subset
    plt.figure(figsize=(10, 6))
    for i, spectrum in enumerate(selected_spectra[:5]):  # Plot the first 5 spectra for reference
        plt.plot(spectrum, label=f'Spectrum {i+1}', alpha=0.7)

    # Plot the mean of the selected points
    plt.plot(mean_spectra, label="Mean of Selected Spectra", color='black', linewidth=2)

    plt.title(f'Selected Subset of Spectral Points (Top {num_points})')
    plt.xlabel('Spectral Point Index')
    plt.legend()

    # Save the plot in the output folder
    plt.savefig(f"{output_folder}/spectral_selection_{num_points}.png")
    print('PLOTTED')

def visualize_selected_points_on_mean_spectrum(mean_original_spectrum, top_indices, num_points=100, output_folder=None):
    # Plot the overall mean spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(mean_original_spectrum, label="Overall Mean Spectrum", color='blue')

    # Highlight the selected points with red dots for visibility
    for idx in top_indices:
        plt.plot(idx, mean_original_spectrum[idx], 'ro', label='Selected Points' if idx == top_indices[0] else "")

    plt.title(f'Overall Mean Spectrum with Top {num_points} Selected Points')
    plt.xlabel('Spectral Point Index')
    plt.ylabel('Mean Value')
    plt.legend()

    # Save the plot in the output folder if provided
    if output_folder:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        plt.savefig(f"{output_folder}/mean_spectrum_selected_points_{num_points}.png")
    else:
        plt.show()
    plt.close()
    print('PLOTTED')



'''
def incremental_mean(df, col_name="features"):
    # Define a UDF to convert DenseVector to list (if needed)
    extract_values = F.udf(lambda v: v.toArray().tolist(), ArrayType(DoubleType()))
    df = df.withColumn("features_array", extract_values(F.col(col_name)))

    first_row = df.select("features_array").first()
    num_components = len(first_row["features_array"])
    mean_vector = np.zeros(num_components)

    for i, row in tqdm(enumerate(df.select("features_array").toLocalIterator(), start=1)):
        new_vector = np.array(row["features_array"])

        # Update the mean vector using incremental mean formula
        mean_vector = mean_vector * (i - 1) / i + new_vector * (1 / i)

        # Explicitly delete the reference to the row and new_vector after use
        del new_vector, row
        gc.collect()  # Force garbage collection

    print('MEANNN')
    return mean_vector
'''