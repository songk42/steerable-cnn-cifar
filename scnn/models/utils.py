import ml_collections
import torch
import torchvision
import torchvision.transforms as transforms

from scnn.models.data_aug import DataAugmentationCNN

def load_data(config, root='./data'):
    if config.dataset == "cifar10":
        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        if config.model == "data_aug":
            transform_list.append(transforms.RandomRotation(180))
        transform = transforms.Compose(transform_list)
        train_data = torchvision.datasets.CIFAR10(root=root, train=True,
                                                download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size,
                                                shuffle=True)

        test_data = torchvision.datasets.CIFAR10(root=root, train=False,
                                            download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=config.batch_size,
                                            shuffle=False)

    return train_loader, test_loader


def create_model(config: ml_collections.ConfigDict):
    if config.model == "data_aug":
        return DataAugmentationCNN(
            num_classes=config.num_classes
        )