import torch
import torch.nn as nn
from groupy.gconv.pytorch_gconv import P4MConvP4M, P4MConvZ2
from groupy.gconv.pytorch_gconv.pooling import plane_group_spatial_max_pooling

class GroupCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(GroupCNN, self).__init__()
        self.layer1 = nn.Sequential(
            P4MConvZ2(in_channels=3, out_channels=18, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(18),
            nn.ReLU(),
        )
        self.conv1 = nn.Sequential(
            P4MConvP4M(in_channels=18, out_channels=24, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm3d(24),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            P4MConvP4M(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(24),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            P4MConvP4M(in_channels=24, out_channels=48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(48),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            P4MConvP4M(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(48),
            nn.ReLU()
        )
        self.avgpool = nn.AvgPool3d(1)
        self.fc1 = nn.Linear(6144, 1536)
        self.fc2 = nn.Linear(1536, 384)
        self.fc3 = nn.Linear(384, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        # x = self.layer1_0(x)
        # print(x.shape)
        # x = self.layer1_1(x)
        # x = self.layer1_2(x)
        x = plane_group_spatial_max_pooling(x, ksize=2, stride=2)
        x = self.conv1(x)
        x = self.layer2(x)
        x = self.conv2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
