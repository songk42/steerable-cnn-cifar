import ml_collections
import torch
import torchvision
import torchvision.transforms as transforms


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

    return train_loader, test_loader


