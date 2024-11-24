''' UNUSED
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR


def train_lime(model, criterion, optimizer, dataloader, batch_step, device, predictions):
    losses = []
    i = 0
    model.train()
    while i < batch_step:
        X = next(dataloader, None)
        optimizer.zero_grad()
        out = model(torch.Tensor(X['features']).to(device))
        loss = criterion(out, torch.Tensor(predictions[X['row_index'] - 1]).long().to(device))
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().numpy())
        i += 1
    return sum(losses)


def evaluate_model(model, dataloader, device, predictions):
    output = []
    model.eval()
    X = next(dataloader, None)
    while X is not None:
        with torch.no_grad():
            output_batch = model(torch.Tensor(X['features']).to(device))
        output.append(output_batch.detach().cpu().numpy().argmax(axis=1) == predictions[X['row_index'] - 1])
        X = next(dataloader, None)
    output = np.concatenate(output, axis=0)
    return output.mean()


def fit_lime(hparams, output_folder, dataloader, predictions):
    model = nn.Linear(hparams.in_dim, hparams.out_dim).to(hparams.device)
    criterion = nn.CrossEntropyLoss(weight=torch.Tensor(hparams.weight).to(hparams.device))
    optimizer = Adam(model.parameters(), lr=hparams.lr)
    scheduler = CosineAnnealingLR(optimizer, hparams.epochs)
    total_losses = []
    best_loss = np.inf
    patience = 0
    for epoch in range(hparams.epochs):
        print(f'Epoch {epoch}')
        if patience > hparams.patience:
            break
        loss = train_lime(model, criterion, optimizer, dataloader, hparams.batch_step, hparams.device, predictions)
        scheduler.step()
        total_losses.append(loss)
        if best_loss > loss:
            torch.save(model.state_dict(), output_folder + '/model_lime.ckpt')
            np.save(output_folder + '/losses_lime.npy', total_losses)
            best_loss = loss
            patience = 0
        else:
            patience += 1
        print(f'Loss: {loss}')
    return model, total_losses
'''
