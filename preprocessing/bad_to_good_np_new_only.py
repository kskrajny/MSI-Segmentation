import os
import numpy as np

current_dir = os.getcwd()
if not current_dir.endswith('MSI-Segmentation'):
    os.chdir(os.path.abspath(os.path.join(current_dir, "..")))
    print(f"Working directory changed to: {os.getcwd()}")
else:
    print(f"Working directory remains: {current_dir}")


labels_dictionary = {
    'healthy_g': 1,
    'highgrade_g': 2,
    'lowgrade_g': 3
}


file_versions_list = [
    # (target_type, data_variant_type)
    (a, f'{b}_{c}') for a in ['', '_3'] for b in 'TPOHLJ' for c in [1, 2, 3]
]

# Notice version should be a value from file_version_list
def convert_to_good_numpy_format_new_only(ver):
    # Load the numpy files
    coords = np.load(f'dane/nowe/root/coords_{ver[1]}_magisterka.npy')
    target = np.load(f'dane/nowe/root/targety{ver[0]}_{ver[1]}_magisterka.npy')
    target_coords = np.load(f'dane/nowe/root/coordsy_targetow{ver[0]}_{ver[1]}_magisterka.npy')
    intsy = np.load(f'dane/nowe/root/intsy_{ver[1]}_magisterka.npy')

    # print(coords.shape, target.shape, target_coords.shape, intsy.shape)
    # Create a dictionary for quick lookup of target values

    coords_to_target = {
        tuple(coord[:2]): labels_dictionary[value] if isinstance(value, np.str_) else value + 1
        for coord, value in zip(target_coords, target)
    }

    # Initialize lists to store filtered results
    filtered_intsy = []
    filtered_coords = []
    filtered_targets = []

    # Iterate through all coordinates and collect corresponding data
    for i, coord in enumerate(coords):
        coord_tuple = tuple(coord[:2])
        if coord_tuple in coords_to_target:
            filtered_intsy.append(intsy[i])  # Append matching intsy row
            filtered_coords.append(coords[i])  # Append corresponding coords
            filtered_targets.append(coords_to_target[coord_tuple])  # Append corresponding target

    # Convert lists back to numpy arrays
    filtered_intsy = np.array(filtered_intsy)
    filtered_coords = np.array(filtered_coords)
    filtered_targets = np.array(filtered_targets)

    files_to_delete = [
        # f'dane/nowe/root/coords_{v}_magisterka.npy',
        # f'dane/nowe/root/targety{c}_{v}_magisterka.npy',
        # f'dane/nowe/root/coordsy_targetow{c}_{v}_magisterka.npy',
        # f'dane/nowe/root/intsy_{v}_magisterka.npy'
    ]

    # Deleting files
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

    # Save the filtered results to new numpy files
    np.save(f'dane/nowe/root/filtered_intsy{ver[0]}_{ver[1]}.npy', filtered_intsy)
    np.save(f'dane/nowe/root/filtered_coords{ver[0]}_{ver[1]}.npy', filtered_coords)
    np.save(f'dane/nowe/root/filtered_targets{ver[0]}_{ver[1]}.npy', filtered_targets)

    # Print shapes for verification
    print("Filtered intsy shape:", filtered_intsy.shape)
    print("Filtered coords shape:", filtered_coords.shape)
    print("Filtered targets shape:", filtered_targets.shape)


if __name__ == '__main__':
    for letter in 'PTHLJO':
        for version in [
            ('_3', f'{letter}_1'), ('_3', f'{letter}_2'), ('_3', f'{letter}_3'),
        ]:
            convert_to_good_numpy_format_new_only(version)
