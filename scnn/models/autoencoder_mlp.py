import torch
import torch.nn as nn
from scnn.models.autoencoder import Autoencoder

class AutoencoderMLP(nn.Module):
    def __init__(
        self,
        img_size,
        num_classes,
    ):
        super(AutoencoderMLP, self).__init__()
        
        self.autoencoder = Autoencoder(3)
        encoder_output_size = (img_size//8)**2 * 64
        self.mlp = nn.Sequential(
            nn.Linear(encoder_output_size, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ELU(inplace=True),
            torch.nn.Dropout(0.5),  # Dropout before the final layer
            nn.Linear(128, num_classes)
        )

    def pretrain(self, x):
        x = self.autoencoder(x)
        return x
    
    def forward(self, x):
        x = self.autoencoder.encoder(x)
        x = torch.flatten(x, 1)
        x = self.mlp(x)
        return x