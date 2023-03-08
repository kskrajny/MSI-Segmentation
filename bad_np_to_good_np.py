import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
ints_file = "dane/sztuczne_dane/root/intsy_sztuczne_magisterka.npy"
mz_file = "dane/sztuczne_dane/root/mz_sztuczne_magisterka.npy"
decimal = 1
'''
ints_file = "dane/pecherz/root/intsy_pecherz_magisterka.npy"
mz_file = "dane/pecherz/root/mz_pecherz_magisterka.npy"
decimal = 1

ints = np.load(ints_file, allow_pickle=True)
mz = np.load(mz_file, allow_pickle=True)

new_mz = []
for i in range(len(ints)):
    new_mz = np.concatenate([new_mz, mz[i]])
    new_mz = [min(new_mz), max(new_mz)]

new_mz = np.around(np.arange(
    np.around(new_mz[0], decimals=decimal), new_mz[1] + 1 / 10 ** decimal, 1 / 10 ** decimal), decimals=decimal)
print(len(new_mz))

df = pd.DataFrame(columns=new_mz)

for i in range(len(ints)):
    mz_val = np.around(mz[i], decimals=decimal)
    vals = pd.DataFrame({"mz": mz_val, "int": ints[i]}).groupby("mz").sum()
    df = pd.concat([df, pd.DataFrame([vals.int], columns=vals.index)])

df.fillna(0, inplace=True)
print(df.shape)
plt.plot(new_mz, df.values[0])
plt.show()

'''
np.save("dane/sztuczne_dane/root/intsy_dobre_magisterka.npy", df.values)
np.save("dane/sztuczne_dane/root/mz.npy", np.array(new_mz))
'''
np.save("dane/pecherz/root/intsy_dobre_magisterka.npy", df.values)
np.save("dane/pecherz/root/mz.npy", np.array(new_mz))
