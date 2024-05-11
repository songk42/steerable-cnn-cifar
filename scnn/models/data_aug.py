import numpy as np
import haiku as hk
import jax

class DataAugmentationCNN(hk.Module):
    def __init__(self, hidden_layers, in_channels, out_channels, kernel_size, padding=1, stride=1):
        super(DataAugmentationCNN, self).__init__()
        # self.in_channels = in_channels
        # self.out_channels = out_channels
        # self.kernel_size = kernel_size
        self.conv_block = [
            hk.Conv2D(out_channels, kernel_size,
                               padding=padding, stride=stride),
            hk.ReLU(),
        ] + (hidden_layers - 2) * [
            hk.Conv2D(out_channels, kernel_size,
                                padding=padding, stride=stride),
            hk.ReLU(),
        ] + [
            hk.Conv2D(10, kernel_size,
                                padding=padding, stride=stride),
            hk.ReLU(),
        ]
