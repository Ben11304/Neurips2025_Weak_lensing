from vit_pytorch import ViT, SimpleViT
import torch
import torch.nn as nn
import torch.optim as optim
from vit_pytorch import ViT
from torch.utils.data import DataLoader, TensorDataset
import numpy as np








class CustomViT(nn.Module):
    """
    Custom ViT module that wraps vit_pytorch.ViT and splits output into pred_means and pred_sigmas.
    """
    def __init__(self):
        super().__init__()

        self.vit=SimpleViT(
            image_size = 256,
            patch_size = 32,
            num_classes = 32768,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            channels=1
        )


        # self.vit = ViT(
        #     image_size=256,
        #     patch_size=32,
        #     num_classes=30000,
        #     dim=1024,
        #     depth=6,
        #     heads=16,
        #     mlp_dim=2048,
        #     dropout=0.1,
        #     emb_dropout=0.1,
        #     channels=1
        # )

        self.fc_stack = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(32768, 512),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(512, 128),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(128, 4)
                )
         # Should be 4 (2 for means, 2 for sigmas)

    def forward(self, x):
        # Input x: [batch_size, channels, image_size, image_size]
        output = self.vit(x) 
        output=self.fc_stack(output)
        means = output[:, :2]
        log_sigmas = output[:, 2:]    # Predict log(σ) to ensure positivity
        sigmas = torch.exp(log_sigmas)
        return means, sigmas 
        
        #  # Shape: [batch_size, num_classes]
        # # print(f"CustomViT: ViT output shape: {output.shape}")  # Debug
        # if output.shape[1] != self.num_classes:
        #     raise ValueError(f"Expected output dim {self.num_classes}, got {output.shape[1]}")
        # pred_means = output[:, :2]  # First 2 values: [batch_size, 2]
        # pred_sigmas = output[:, 2:]  # Last 2 values: [batch_size, 2]
        # # print(f"CustomViT: pred_means shape: {pred_means.shape}, pred_sigmas shape: {pred_sigmas.shape}")  # Debug
        # return pred_means, pred_sigmas



# Simple CNN architecture for parameter estimation

class Simple_CNN(nn.Module):
    def __init__(self, height, width, num_targets):
        super(Simple_CNN, self).__init__()
        # Convolutional layers
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self._feature_size = self._get_conv_output_size(height, width)
        
        # Fully connected layers (regressor head)
        self.fc_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_targets)
        )

    def _get_conv_output_size(self, height, width):
        dummy_input = torch.zeros(1, 1, height, width)
        output = self.conv_stack(dummy_input)
        return int(np.prod(output.size()))

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.fc_stack(x)
        means = x[:, :2]
        log_sigmas = x[:, 2:]    # Predict log(σ) to ensure positivity
        sigmas = torch.exp(log_sigmas)
        return means, sigmas     # Note that means and sigmas here have to be rescaled properly due to standardization