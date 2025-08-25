#!/usr/bin/env python3
"""
EvoAug2 Vanilla PyTorch Example

Demonstrates EvoAug2 usage with simple PyTorch training loop.
Shows RobustLoader creation, CNN design for genomic sequences, and training workflow.

Usage: python example_vanilla_pytorch.py
"""

import torch
import torch.nn as nn
from evoaug.evoaug import RobustLoader
from evoaug import augment
from evoaug_utils import utils

# Configuration
filepath = '/grid/koo/home/duran/ray_results/deepstarr-data.h5'
batch_size = 128
num_epochs = 10

# Dataset and augmentations
base_dataset = utils.H5Dataset(filepath, batch_size=batch_size, lower_case=False, transpose=False)

augment_list = [
    augment.RandomTranslocation(shift_min=0, shift_max=20),
    augment.RandomRC(rc_prob=0.0),
    augment.RandomMutation(mut_frac=0.05),
    augment.RandomNoise(noise_mean=0.0, noise_std=0.3),
    augment.RandomDeletion(delete_min=0, delete_max=30),
    augment.RandomInsertion(insert_min=0, insert_max=20),
]

train_loader = RobustLoader(
    base_dataset=base_dataset,
    augment_list=augment_list,
    max_augs_per_seq=2,
    hard_aug=True,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
)

class CNN(nn.Module):
    """1D CNN for genomic sequences with global average pooling for variable lengths."""
    def __init__(self, num_classes=1):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64, num_classes)
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        
    def forward(self, x):
        # x: (N, 4, L) -> (N, num_classes)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.mean(x, dim=2)  # Global average pooling
        x = self.dropout(x)
        return self.fc(x)

# Model setup
model = CNN()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        loss.backward()
        optimizer.step()

# Evaluation
model.eval()
with torch.no_grad():
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        y_hat = model(x)
        loss = criterion(y_hat, y)
        print(f"Test Loss: {loss.item():.4f}")
        break

print("Training completed!")