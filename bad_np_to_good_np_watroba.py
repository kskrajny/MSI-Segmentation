import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path_to_data = "dane/watroba/"
files_intsy = [
    path_to_data + 'root/intsy_from_0_to_2999_magisterka.npy',
    path_to_data + 'root/intsy_from_3000_to_6000_magisterka.npy',
    path_to_data + 'root/intsy_from_6001_to_9000_magisterka.npy',
    path_to_data + 'root/intsy_from_9001_to_10011_magisterka.npy'
]
decimal = 1

mz = np.load(path_to_data + 'root/mz_big.npy', allow_pickle=True)

new_mz = []
for i in range(len(mz)):
    new_mz = np.concatenate([new_mz, mz[i]])
    new_mz = [min(new_mz), max(new_mz)]

new_mz = np.around(np.arange(
    np.around(new_mz[0], decimals=decimal), new_mz[1] + 1 / 10 ** decimal, 1 / 10 ** decimal), decimals=decimal)
df = pd.DataFrame(columns=new_mz)
print(len(new_mz))

for ints_file in files_intsy:
    ints = np.load(ints_file, allow_pickle=True)
    for i in range(len(ints)):
        mz_val = np.around(mz[i], decimals=decimal)
        vals = pd.DataFrame({"mz": mz_val, "int": ints[i]}).groupby("mz").sum()
        df = pd.concat([df, pd.DataFrame([vals.int], columns=vals.index)])
    df.fillna(0, inplace=True)
    print(df.shape)
    plt.plot(new_mz, df.values[0])
    plt.show()

np.save("dane/watroba/root/intsy_dobre_magisterka.npy", df.values)
np.save("dane/watroba/root/mz.npy", np.array(new_mz))
