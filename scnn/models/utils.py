import ml_collections
import torch
import torchvision
import torchvision.transforms as transforms

from scnn.models.data_aug import DataAugmentationCNN
from scnn.models.gcnn import GroupCNN

def load_data(config, root='./data'):
    if config.dataset == "cifar10":
        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        transform_test = transforms.Compose(transform_list)
        if config.augment_data:
            transform_list.append(transforms.RandomHorizontalFlip())
        transform = transforms.Compose(transform_list)
        train_data = torchvision.datasets.CIFAR10(root=root, train=True,
                                                download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size,
                                                shuffle=True)

        test_data = torchvision.datasets.CIFAR10(root=root, train=False,
                                            download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=config.batch_size,
                                            shuffle=False)
    elif config.dataset == "caltech101":
        transform_list = [
            transforms.ToTensor(),
            transforms.Resize((320, 320)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        transform_test = transforms.Compose(transform_list)
        if config.augment_data:
            transform_list.append(transforms.RandomHorizontalFlip())
        transform = transforms.Compose(transform_list)
        dataset = torchvision.datasets.Caltech101(root=root, download=True, transform=transform)
        num_train = int(config.data_split[0] * len(dataset))
        num_test = len(dataset) - num_train
        train_data, test_data = torch.utils.data.random_split(
            dataset, [num_train, num_test])
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size,
                                                shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=config.batch_size,
                                            shuffle=False)

    return train_loader, test_loader


def create_model(config: ml_collections.ConfigDict):
    if config.model == "data_aug":
        return DataAugmentationCNN(
            img_size=32,
            num_classes=config.num_classes
        )
    elif config.model == "gcnn":
        return GroupCNN(
            img_size=320,
            num_classes=config.num_classes
        )