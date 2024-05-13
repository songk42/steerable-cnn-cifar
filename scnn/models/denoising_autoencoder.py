import torch
import torch.nn as nn
from scnn.models.autoencoder import Autoencoder

class DenoisingAutoencoder(nn.Module):
    def __init__(
        self,
        img_size,
        num_classes,
    ):
        super(DenoisingAutoencoder, self).__init__()
        
        self.sigmoid1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.denoiser1 = Autoencoder(16)
        self.sigmoid2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.denoiser2 = Autoencoder(16)
        self.sigmoid3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.denoiser3 = Autoencoder(16)
        self.log_layer = nn.Sequential(
            nn.Linear(img_size**2 * 16, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.sigmoid1(x)
        x = self.denoiser1(x)
        x = self.sigmoid2(x)
        x = self.denoiser2(x)
        x = self.sigmoid3(x)
        x = self.denoiser3(x)
        x = self.log_layer(torch.flatten(x, 1))
        return x