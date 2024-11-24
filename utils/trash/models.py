''' UNUSED
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR


def noise(vec):
    vec_nonzero_mean = vec[vec > 0].mean()
    sigma = np.random.uniform(vec_nonzero_mean / 100, vec_nonzero_mean / 10)
    noise = np.random.normal(0, sigma, vec.shape)
    vec = vec + torch.Tensor(noise) * (vec != 0)
    return vec.float()


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class ANN(nn.Module):
    def __init__(self, dims, device='cpu'):
        super().__init__()
        self.device = device
        self.loss_function = nn.CrossEntropyLoss()
        self.activation = nn.ReLU()
        self.kernel_size = 64
        self.net = nn.ModuleList()
        for i in range(len(dims) - 2):
            self.net.append(LambdaLayer(lambda x: x.unsqueeze(1)))
            self.net.append(nn.Conv1d(1, 1, self.kernel_size, stride=self.kernel_size // 4))
            self.net.append(nn.Flatten())
            self.net.append(nn.BatchNorm1d(num_features=(dims[i]-self.kernel_size) // (self.kernel_size // 4) + 1))
            self.net.append(self.activation)
            self.net.append(nn.Linear((dims[i]-self.kernel_size) // (self.kernel_size // 4) + 1, dims[i + 1]))
            self.net.append(nn.BatchNorm1d(num_features=dims[i + 1]))
        self.net.append(nn.Linear(dims[-2], dims[-1]))
        self.net.append(nn.BatchNorm1d(num_features=dims[-1]))
        self.net.to(device)

    def forward(self, x):
        for module in self.net:
            x = module(x)
        return x


class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, device, temperature=0.25):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())

    def forward(self, emb_i, emb_j, encoder_inputs, decoder_outputs):
        # Contrastive Loss
        representations = torch.cat([emb_i, emb_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        contrastive_loss = torch.sum(loss_partial) / (2 * self.batch_size)

        # Mean & Std Loss
        std_loss = torch.mean(torch.std(representations, dim=0))
        mean_loss = torch.mean(torch.mean(representations, dim=1) ** 2)

        # MSE Loss
        mse_loss = torch.mean((encoder_inputs - decoder_outputs) ** 2)

        # Combine the losses
        total_loss = contrastive_loss + std_loss + mean_loss + mse_loss

        return total_loss


# 2.3) SimCLR model


class CLR(nn.Module):
    def __init__(self, dims, device):
        super().__init__()
        self.embedding = ANN(dims, device)
        self.device = device

    def forward(self, X):
        embedding = self.embedding(X)
        return embedding


def get_clr_training_components(hparams_CLR):
    model_CLR = CLR(hparams_CLR.dims, hparams_CLR.device)
    criterion_CLR = ContrastiveLoss(hparams_CLR.batch_size, hparams_CLR.device)
    optimizer_CLR = Adam(model_CLR.parameters(), lr=hparams_CLR.lr)
    scheduler_CLR = CosineAnnealingLR(optimizer_CLR, hparams_CLR.epochs)
    return model_CLR, criterion_CLR, optimizer_CLR, scheduler_CLR


def train_clr(model, criterion, optimizer, dataloader, batch_step):
    losses = []
    i = 0
    model.train()
    while i < batch_step:
        X = next(dataloader, None)
        optimizer.zero_grad()
        emb_1 = model(torch.Tensor(X['features']).to(model.device))
        emb_2 = model(torch.Tensor(noise(X['features'])).to(model.device))
        loss = criterion(emb_1, emb_2)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().numpy())
        i += 1
    return sum(losses)

def fit_clr(hparams_CLR, output_folder, dataloader):
    model_CLR, criterion_CLR, optimizer_CLR, scheduler_CLR = get_clr_training_components(hparams_CLR)
    total_losses = []
    best_loss = np.inf
    patience = 0
    for epoch in range(hparams_CLR.epochs):
        if patience > hparams_CLR.patience:
            break
        loss = train_clr(model_CLR, criterion_CLR, optimizer_CLR, dataloader, hparams_CLR.batch_step)
        scheduler_CLR.step()
        total_losses.append(loss)
        if best_loss > loss:
            torch.save(model_CLR.state_dict(), output_folder + '/model_CLR.ckpt')
            np.save(output_folder + '/losses_CLR.npy', total_losses)
            best_loss = loss
            patience = 0
        else:
            patience += 1
        print(f'Epoch {epoch}: loss: {loss}')
    return model_CLR, total_losses


def run_thru_NN(model, dataloader):
    output = []
    model.eval()
    X = next(dataloader, None)
    while X is not None:
        X = torch.Tensor(X['features']).to(model.device)
        with torch.no_grad():
            output_batch = model(X)
        output.append(output_batch.detach().cpu().numpy())
        X = next(dataloader, None)
    output = np.concatenate(output, axis=0)
    return output
'''