import sys
sys.path.append('../')

import torch
from escnn import gspaces, nn
from utils import load_data
import ml_collections
import os
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW

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

# Load data and configure settings
config = ml_collections.ConfigDict()
config.dataset = "cifar10"
config.batch_size = 128  # Increased batch size for better generalization
config.augment_data = True

train_loader, test_loader = load_data(config)

# Instantiate the model and move it to the appropriate device
model = C4SteerableCNN().to(device)

# Loss function and optimizer
loss_function = torch.nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)  # AdamW optimizer for better convergence
scheduler = CosineAnnealingLR(optimizer, T_max=10)  # Cosine annealing scheduler

# Directory to save the models
save_dir = "saved_models"
os.makedirs(save_dir, exist_ok=True)

# Training loop
num_epochs = 50  # Increased number of epochs
best_accuracy = 0.0
early_stopping_patience = 10
epochs_no_improve = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (x, t) in enumerate(train_loader):
        optimizer.zero_grad()
        x, t = x.to(device), t.to(device)
        y = model(x)
        loss = loss_function(y, t)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    # Adjust the learning rate
    scheduler.step()

    total, correct = 0, 0
    with torch.no_grad():
        model.eval()
        for i, (x, t) in enumerate(test_loader):
            x, t = x.to(device), t.to(device)
            y = model(x)
            _, prediction = torch.max(y.data, 1)
            total += t.shape[0]
            correct += (prediction == t).sum().item()
    accuracy = correct / total * 100
    print(f"Epoch {epoch+1} | Loss: {running_loss/len(train_loader):.4f} | Test Accuracy: {accuracy:.2f}%")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        epochs_no_improve = 0
        torch.save(model.state_dict(), os.path.join(save_dir, f"best_model.pth"))
    else:
        epochs_no_improve += 1

    if epochs_no_improve == early_stopping_patience:
        print("Early stopping")
        break

# To load the model's state dictionary later
# model = C4SteerableCNN()
# model.load_state_dict(torch.load('saved_models/best_model.pth'))
# model.to(device)
