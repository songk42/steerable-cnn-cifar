import numpy as np
import haiku as hk
import jax

class CNN(hk.Module):
    def __init__(self, hidden_layers, hidden_channels, kernel_size, padding=1, stride=1):
        super(DataAugmentationCNN, self).__init__()
        self.conv_block = [
            hk.Conv2D(hidden_channels, kernel_size,
                               padding=padding, stride=stride),
            hk.ReLU(),
        ] + (hidden_layers - 2) * [
            hk.Conv2D(hidden_channels, kernel_size,
                                padding=padding, stride=stride),
            hk.ReLU(),
        ] + [
            hk.Conv2D(10, kernel_size,
                                padding=padding, stride=stride),
        ]

    def forward(self, x):
        for layer in self.conv_block:
            x = layer(x)
        return x
