import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, img_size, num_classes=10):
        super(CNN, self).__init__()
        self.block1 = self._make_block(3, 32, 7, 1)
        self.block2 = self._make_block(32, 64, 5, 2)
        self.pool1 = nn.AvgPool2d((1, 1), 2)
        self.block3 = self._make_block(64, 64, 5, 2)
        self.block4 = self._make_block(64, 128, 5, 2)
        self.pool2 = nn.AvgPool2d((1, 1), 2)
        self.block5 = self._make_block(128, 128, 5, 2)
        self.block6 = self._make_block(128, 256, 5, 1)
        self.avgpool = nn.AvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(25 * img_size**2 // 4, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ELU(inplace=True),
            torch.nn.Dropout(0.5),  # Dropout before the final layer
            nn.Linear(128, num_classes)
        )

    def _make_block(self, in_channels, out_channels, kernel_size, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool1(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool2(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x



# import torch
# import torch.nn as nn

# class CNN(nn.Module):
#     def __init__(self, img_size, num_classes=10):
#         super(CNN, self).__init__()
#         self.layer11 = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=48, kernel_size=3, stride=1, padding="same"),
#             nn.BatchNorm2d(48),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#         self.layer12 = nn.Sequential(
#             nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding="same"),
#             nn.BatchNorm2d(48),
#             nn.ReLU(),
#         )
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channels=48, out_channels=64, kernel_size=7, stride=2, padding=3),
#             nn.BatchNorm2d(64),
#             nn.ReLU()
#         )
#         self.layer21 = nn.Sequential(
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding="same"),
#             nn.BatchNorm2d(64),
#             nn.ReLU()
#         )
#         self.layer22 = nn.Sequential(
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding="same"),
#             nn.BatchNorm2d(64),
#             nn.ReLU()
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU()
#         )
#         self.layer31 = nn.Sequential(
#             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding="same"),
#             nn.BatchNorm2d(128),
#             nn.ReLU()
#         )
#         self.layer32 = nn.Sequential(
#             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding="same"),
#             nn.BatchNorm2d(128),
#             nn.ReLU()
#         )
#         self.avgpool = nn.AvgPool2d((1, 1))
#         self.fc1 = nn.Linear(2 * img_size**2, img_size**2)
#         self.fc2 = nn.Linear(img_size**2, img_size**2 // 2)
#         self.fc3 = nn.Linear(img_size**2 // 2, img_size**2 // 4)
#         self.fc4 = nn.Linear(img_size**2 // 4, img_size**2 // 8)
#         self.fc5 = nn.Linear(img_size**2 // 8, num_classes)

#     def forward(self, x):
#         x = self.layer11(x)
#         x = self.layer12(x)
#         x = self.conv1(x)
#         x = self.layer21(x)
#         x = self.layer22(x)
#         x = self.conv2(x)
#         x = self.layer31(x)
#         x = self.layer32(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = self.fc2(x)
#         x = self.fc3(x)
#         x = self.fc4(x)
#         x = self.fc5(x)
#         return x
