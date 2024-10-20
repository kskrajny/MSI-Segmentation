import os

import numpy as np

d = {
    'healthy_g': 1,
    'highgrade_g': 2,
    'lowgrade_g': 3
}

if __name__ == '__main__':
    for t in 'TPOLJH':
        for n in [1, 2, 3]:
            v = f'{t}_{n}'

            # Load the numpy files
            target = np.load(f'dane/nowe/root/targety_3_{v}_magisterka.npy')
            target_coords = np.load(f'dane/nowe/root/coordsy_targetow_3_{v}_magisterka.npy')
            already_filtered_coords = np.load(f'dane/nowe/root/filtered_coords_{v}.npy')

            print(
                np.unique(target, return_counts=True)
            )

            # Create a dictionary for quick lookup of target values
            coords_to_target = {tuple(coord[:2]): d[t] for coord, t in zip(target_coords, target)}

            # Initialize lists to store filtered results
            filtered_targets = []

            # Iterate through all coordinates and collect corresponding data
            for i, coord in enumerate(already_filtered_coords):
                coord_tuple = tuple(coord[:2])
                filtered_targets.append(coords_to_target.get(coord_tuple, 0))  # Append corresponding target

            # Convert lists back to numpy arrays
            filtered_targets = np.array(filtered_targets)
            print(np.unique(filtered_targets))
            np.save(f'dane/nowe/root/filtered_targets_3_{v}.npy', filtered_targets)

            print("Filtered targets shape:", filtered_targets.shape)
