from imzml_to_numpy import cords

# MSI-self-supervised-clustering

TODO introduction

___
# Workflow for Developers

## Prepare Dataset in Good Format

### Datasets Used in Papers
[Data in bad format used in resarch](https://drive.google.com/drive/u/2/folders/14cli_aVFAocVRCBk0GRllIJwUyj4OTOu) \
[Good format](https://drive.google.com/drive/u/1/folders/1C04FcG6QzxF6dE5n4qTk0ArI1-b8Cn3H)
(plik example_results_and_dane.zip, rozpakować i katalogi /dane oraz /resuts dodać bezpośrednio do źródłowego katalogu projektu)

### Custom Dataset
In case you want to use custom data, save your data in numpy format like in the example below. \
Place them inside `dane/[custom_name]/root/` directory. \
For real examples please see some code in [`preprocessing`](preprocessing/) 
```python
import numpy as np

# Coordinates of pixels in the 2d image
cords = np.array([
    [0, 0],
    [0, 1]
])

# Example m/z values
# IMPORTANT: increasing order and same difference between next elements should be preserved
mz = np.array([
    100, 100.1, 100.2, 100.3, 100.4, 100.5,
])

# Example intensities at specific m/z values
intensities = np.array([
    [0, 75.3, 0.0, 0, 55.2, 72.43],
    [0, 72.3, 0.0, 0, 10.4, 13.01]
])

np.save('data/custom/root/cords.npy', cords)
np.save('data/custom/root/mz.npy', mz)
np.save('data/custom/root/intsy.npy', intensities)
```

___
__Workflow:__  
1. Dodać dane do katalogu dane.
2. Doprowadzić dane do takiej formy jaką można otrzymać w wyniki uruchomienia **bad_np_to_good_(ver).py**,
jeśli dane są w formacie imzml, to być może **imzml_to_numpy.py** pomoże.
3. Uruchomić **np_to_parquet.py** (w przyszłości ten krok można ominąć i przpisać kod na niekorzystanie z parquet), zapisuje on dane w innej strukturze.
4. Obliczenia:  
    a. uruchomić **original_kmeans.py**, aby policzyć klastry na spektrach.  
    b. uruchomić trening sieci**.  
    bb. uruchomić **contrastive_kmeans.py**, aby policzyć klastry na reprezentacjach spektr.  
5. uruchomić **evaluate.py** aby policzyć wyniki, narysować obrazki.  

** kod do treningu sieci jest na google colab: https://drive.google.com/drive/u/1/folders/1C04FcG6QzxF6dE5n4qTk0ArI1-b8Cn3H
___

### Future Work:
Zalążki XAI w **evaluate_lime.py**.
