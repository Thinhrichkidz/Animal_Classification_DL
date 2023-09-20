import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class AnimalDataset(Dataset):
    def __init__(self, root, train, transform=None):
        data_path = os.path.join(root, "animals")
        self.image_paths = []
        self.labels = []
        if train:
            data_path = os.path.join(data_path, "train")
        else:
            data_path = os.path.join(data_path, "test")
        self.categories = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider",
                      "squirrel"]
        for i, category in enumerate(self.categories):
            category_path = os.path.join(data_path, category)
            for item in os.listdir(category_path):
                self.image_paths.append(os.path.join(category_path, item))
                self.labels.append(i)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image_path = self.image_paths[item]
        label = self.labels[item]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label


class CIFARDataset(Dataset):
    def __init__(self, root="data", train=True, transform=None):
        data_path = os.path.join(root, "cifar-10-batches-py")
        if train:
            data_files = [os.path.join(data_path, "data_batch_{}".format(i)) for i in range(1, 6)]
        else:
            data_files = [os.path.join(data_path, "test_batch")]
        self.images = []
        self.labels = []
        for data_file in data_files:
            with open(data_file, 'rb') as fo:
                data = pickle.load(fo, encoding='bytes')
                self.images.extend(data[b'data'])
                self.labels.extend(data[b'labels'])
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        image = self.images[item].reshape((3, 32, 32)).astype(np.float32)
        if self.transform:
            image = np.transpose(image, (1, 2, 0))
            image = self.transform(image)
        else:
            image = torch.from_numpy(image)
        label = self.labels[item]
        return image, label

if __name__ == '__main__':
    from torchvision.datasets import ImageFolder
    # dataset = AnimalDataset("../data", True)
    dataset = ImageFolder("../data/animals/train")
    index = 6456
    image, label = dataset.__getitem__(index)
    print(label)
    image.show()





