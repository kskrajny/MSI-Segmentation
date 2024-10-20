import torch

if __name__ == '__main__':
    state_dict = torch.load('results/pecherz_08-11-2023-17-22_conv_True/model_CLR.ckpt', map_location=torch.device('cpu'))
    for key, value in state_dict.items():
        print(f"{key}: {value.shape}")

'''
PECHERZ: L4
Trening: 1h

WATROBA: L4
Trening: 30-40 min

Nowe: T4
Trening: 40 min

Sztuczne: ?
Trening: ?
'''



