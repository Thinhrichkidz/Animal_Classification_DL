import torch.nn as nn
import torch.optim
from torchvision.datasets import CIFAR10
from src.datasets import CIFARDataset
from src.models import SimpleCNN
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


import numpy as np
def train():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    num_epochs = 100
    batch_size = 64
    train_set = CIFAR10(root="data", train=True, download=True, transform=ToTensor())
    train_dataloader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    val_set = CIFAR10(root="data", train=False, download=True, transform=ToTensor())
    val_dataloader = DataLoader(
        dataset=val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )
    model = SimpleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    num_iters = len(train_dataloader)

    for epoch in range(num_epochs):
        model.train()
        for iter, (images,  labels) in enumerate(train_dataloader):
            images = images.to(device)
            labels = labels.to(device)

            # forward
            output = model(images)
            loss = criterion(output, labels)

            # backward + optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if iter % 10 == 0:
                print("Epoch {}/{}. Iter {}/{}. Loss {}".format(epoch+1, num_epochs, iter+1, num_iters, loss))

if __name__ == '__main__':
    train()