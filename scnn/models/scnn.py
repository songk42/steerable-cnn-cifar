import torch
from escnn import gspaces, nn

# Check for GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class C4SteerableCNN(torch.nn.Module):
    def __init__(self, n_classes=10):
        super(C4SteerableCNN, self).__init__()

        # Model is equivariant under rotations by 90 degrees, modeled by C4
        self.r2_act = gspaces.rot2dOnR2(N=4)
        self.input_type = nn.FieldType(self.r2_act, 3 * [self.r2_act.trivial_repr])

        # Create convolutional blocks
        self.block1 = self._make_block(self.input_type, 32, kernel_size=7, padding=1)
        self.block2 = self._make_block(self.block1.out_type, 64)
        self.pool1 = nn.SequentialModule(nn.PointwiseAvgPoolAntialiased(self.block2.out_type, sigma=0.66, stride=2))
        self.block3 = self._make_block(self.block2.out_type, 64)
        self.block4 = self._make_block(self.block3.out_type, 128)
        self.pool2 = nn.SequentialModule(nn.PointwiseAvgPoolAntialiased(self.block4.out_type, sigma=0.66, stride=2))
        self.block5 = self._make_block(self.block4.out_type, 128)
        self.block6 = self._make_block(self.block5.out_type, 256, padding=1)  # Increased channels
        self.pool3 = nn.PointwiseAvgPoolAntialiased(self.block6.out_type, sigma=0.66, stride=1, padding=0)
        self.gpool = nn.GroupPooling(self.block6.out_type)
        c = self.gpool.out_type.size

        # Fully connected layers
        self.fully_net = torch.nn.Sequential(
            torch.nn.Linear(c, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ELU(inplace=True),
            torch.nn.Dropout(0.5),  # Dropout before the final layer
            torch.nn.Linear(128, n_classes)
        )

    def _make_block(self, in_type, out_channels, kernel_size=5, padding=2):
        out_type = nn.FieldType(self.r2_act, out_channels * [self.r2_act.regular_repr])
        return nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=kernel_size, padding=padding, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )

    def forward(self, input: torch.Tensor):
        x = nn.GeometricTensor(input, self.input_type)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool1(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool2(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.pool3(x)
        x = self.gpool(x)
        x = x.tensor
        x = self.fully_net(x.reshape(x.shape[0], -1))
        return x