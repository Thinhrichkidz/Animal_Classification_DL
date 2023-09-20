import torch.nn as nn
import torch.optim
from torchvision.datasets import CIFAR10
from torch.utils.tensorboard import SummaryWriter
from src.datasets import CIFARDataset, AnimalDataset
from src.models import SimpleCNN, AdvancedCNN
from torchvision.transforms import ToTensor, Compose, Resize, ColorJitter, Normalize, RandomAffine
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score
import argparse
import os
import shutil
from tqdm.autonotebook import tqdm


def get_args():
    parser = argparse.ArgumentParser("""Train model for Animal dataset""")
    parser.add_argument("--batch-size", "-b", type=int, default=32, help="batch size of dataset")
    parser.add_argument("--epochs", "-e", type=int, default=100, help="number of epochs")
    parser.add_argument("--log_path", "-l", type=str, default="./tensorboard/animal", help="path to tensorboard")
    parser.add_argument("--save_path", "-s", type=str, default="./trained_models/animal", help="path to trained models")
    parser.add_argument("--load_checkpoint", "-m", type=str, default=None, help="path to checkpoint to be loaded")
    args = parser.parse_args()
    return args


def train(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    num_epochs = args.epochs
    batch_size = args.batch_size
    # train_set = CIFAR10(root="data", train=True, download=True, transform=ToTensor())
    train_transform = Compose([
        RandomAffine(degrees=(-5, 5), translate=(0.15, 0.15), scale=(0.9, 1.1)),
        Resize(size=(224, 224)),
        ColorJitter(brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05),
        ToTensor(),
    ])
    train_set = AnimalDataset(root="data", train=True, transform=train_transform)
    train_dataloader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    # val_set = CIFAR10(root="data", train=False, download=True, transform=ToTensor())
    val_transform = Compose([
        Resize(size=(224, 224)),
        ToTensor()
    ])
    val_set = AnimalDataset(root="data", train=False, transform=val_transform)
    val_dataloader = DataLoader(
        dataset=val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )
    model = AdvancedCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_iters = len(train_dataloader)
    if args.load_checkpoint:
        checkpoint = torch.load(args.load_checkpoint)
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        start_epoch = 0

    if os.path.isdir(args.log_path):
        shutil.rmtree(args.log_path)
    os.makedirs(args.log_path)
    writer = SummaryWriter(args.log_path)
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    best_acc = 0
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = []
        progress_bar = tqdm(train_dataloader, colour="cyan")
        for iter, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)

            # forward
            output = model(images)
            loss = criterion(output, labels)

            # backward + optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            writer.add_scalar("Train/Loss", np.mean(train_loss), num_iters*epoch+iter)
            progress_bar.set_description("Epoch {}/{}. Loss {:0.4f}".format(epoch + 1, num_epochs, np.mean(train_loss)))

        model.eval()
        all_labels = []
        all_predictions = []
        for iter, (images, labels) in enumerate(val_dataloader):
            images = images.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                # with torch.inference_model():
                output = model(images)
                _, predictions = torch.max(output, dim=1)
                all_predictions.extend(predictions.tolist())
                all_labels.extend(labels.tolist())
        acc = accuracy_score(all_labels, all_predictions)
        writer.add_scalar("Val/Accuracy", acc, epoch)
        checkpoint = {
            "epoch": epoch+1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        torch.save(checkpoint, os.path.join(args.save_path, "last.pt"))
        if acc > best_acc:
            torch.save(checkpoint, os.path.join(args.save_path, "best.pt"))
            best_acc = acc


if __name__ == '__main__':
    args = get_args()
    train(args)
