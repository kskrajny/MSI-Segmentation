import os

import numpy as np

d = {
    'healthy_g': 1,
    'highgrade_g': 2,
    'lowgrade_g': 3
}


if __name__ == '__main__':

    #c = '_3'
    c = ''

    for t in 'P': #'TPOLJH'
        for n in [1, 2, 3]:
            v = f'{t}_{n}'

            # Load the numpy files
            coords = np.load(f'dane/nowe/root/coords_{v}_magisterka.npy')
            target = np.load(f'dane/nowe/root/targety{c}_{v}_magisterka.npy')
            target_coords = np.load(f'dane/nowe/root/coordsy_targetow{c}_{v}_magisterka.npy')
            intsy = np.load(f'dane/nowe/root/intsy_{v}_magisterka.npy')

            print(coords.shape, target.shape, target_coords.shape, intsy.shape)
            # Create a dictionary for quick lookup of target values

            coords_to_target = {tuple(coord[:2]): d[value] if isinstance(value, np.str_) else value + 1
                                for coord, value in zip(target_coords, target)}

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
                #f'dane/nowe/root/coords_{v}_magisterka.npy',
                #f'dane/nowe/root/targety{c}_{v}_magisterka.npy',
                #f'dane/nowe/root/coordsy_targetow{c}_{v}_magisterka.npy',
                #f'dane/nowe/root/intsy_{v}_magisterka.npy'
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
            np.save(f'dane/nowe/root/filtered_intsy{c}_{v}.npy', filtered_intsy)
            np.save(f'dane/nowe/root/filtered_coords{c}_{v}.npy', filtered_coords)
            np.save(f'dane/nowe/root/filtered_targets{c}_{v}.npy', filtered_targets)

            # Print shapes for verification
            print("Filtered intsy shape:", filtered_intsy.shape)
            print("Filtered coords shape:", filtered_coords.shape)
            print("Filtered targets shape:", filtered_targets.shape)
