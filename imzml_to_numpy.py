import numpy as np
from pyimzml.ImzMLParser import ImzMLParser
from tqdm import tqdm

my_image = ImzMLParser('dane/mózg/root/lipid_MSI_profile_mode.imzML')

intsy = []
mzy = []
cords = []

for idx, (xcoord, ycoord, zcoord) in tqdm(enumerate(my_image.coordinates)):
    mz, ints = my_image.getspectrum(idx)
    intsy.append(ints)
    mzy.append(mz)
    cords.append([xcoord, ycoord])

intsy = np.array(intsy)
mzy = np.array(mzy)
cords = np.array(cords)

np.save('dane/mózg/root/intsy_mózg_magisterka', intsy)
np.save('dane/mózg/root/mz_mózg_magisterka', mzy)
np.save('dane/mózg/root/cords_mózg_magisterka', cords)
