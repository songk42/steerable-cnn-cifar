import ml_collections
import numpy as np
import torch
from torch import nn
import torchvision
from torchvision.transforms import v2

from scnn.models.cnn import CNN
from scnn.models.autoencoder_mlp import AutoencoderMLP
from scnn.models.gcnn import GroupCNN
from scnn.models.scnn import C4SteerableCNN


def add_noise(img, mean=0, var=10):
    row, col = img.shape[-2:]
    sigma = var**0.5
    noise = torch.normal(mean, sigma, (row, col), device=img.device)
    img = img + img * noise
    img = torch.minimum(torch.maximum(img, torch.zeros_like(img)), torch.ones_like(img))
    return img


class NoiseTransform(nn.Module):
    def __init__(self, mean, var):
        super(NoiseTransform, self).__init__()
        self.mean = mean
        self.var = var

    def forward(self, input):
        return add_noise(input, self.mean, self.var)


def load_data(config, root="./data"):
    if config.dataset == "cifar10":
        transform_list = [
            v2.ToTensor(),
            v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    elif config.dataset == "caltech101":
        transform_list = [
            v2.ToTensor(),
            v2.Resize((320, 320)),
            v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]

    transform_test = v2.Compose(transform_list)
    if config.augment_data:
        transform_list.append(v2.RandomHorizontalFlip())
        transform_list.append(v2.RandomVerticalFlip())
    if config.add_noise:
        transform_list.append(NoiseTransform(config.noise_mean, config.noise_var))
    transform_train = v2.Compose(transform_list)

    if config.dataset == "cifar10":
        train_data = torchvision.datasets.CIFAR10(
            root=root, train=True, download=True, transform=transform_train
        )

        test_data = torchvision.datasets.CIFAR10(
            root=root, train=False, download=True, transform=transform_test
        )
    elif config.dataset == "caltech101":
        dataset = torchvision.datasets.Caltech101(
            root=root, download=True, transform=transform_train
        )
        num_train = int(config.data_split[0] * len(dataset))
        num_test = len(dataset) - num_train
        train_data, test_data = torch.utils.data.random_split(
            dataset, [num_train, num_test]
        )
    
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=config.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=config.batch_size, shuffle=False
    )

    return train_loader, test_loader


def get_num_classes(config):
    if config.dataset == "cifar10":
        return 10
    elif config.dataset == "caltech101":
        return 101


def create_model(config: ml_collections.ConfigDict):
    if config.dataset == "cifar10":
        img_size = 32
    elif config.dataset == "caltech101":
        img_size = 320
    num_classes = get_num_classes(config)
    if config.model == "cnn":
        return CNN(img_size=img_size, num_classes=num_classes)
    elif config.model == "gcnn":
        return GroupCNN(img_size=img_size, num_classes=num_classes)
    elif config.model == "autoencoder":
        return AutoencoderMLP(img_size=img_size, num_classes=num_classes)
    elif config.model == "scnn":
        return C4SteerableCNN(num_classes)
