import torch
import torchvision
import torchvision.transforms as transforms


def load_data(batch_size=4, num_workers=2, root='./data', transform=None):
    train_data = torchvision.datasets.CIFAR10(root=root, train=True,
                                            download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers)

    test_data = torchvision.datasets.CIFAR10(root=root, train=False,
                                        download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                            shuffle=False, num_workers=num_workers)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    return train_loader, test_loader, classes