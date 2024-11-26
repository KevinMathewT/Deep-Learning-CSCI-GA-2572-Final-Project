from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch


def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)


class MockModel(torch.nn.Module):
    """
    Does nothing. Just for testing.
    """

    def __init__(self, device="cuda", bs=64, n_steps=17, output_dim=256):
        super().__init__()
        self.device = device
        self.bs = bs
        self.n_steps = n_steps
        self.repr_dim = 256

    def forward(self, states, actions):
        """
        Args:
            states: [B, T, Ch, H, W]
            actions: [B, T-1, 2]

        Output:
            predictions: [B, T, D]
        """
        return torch.randn((self.bs, self.n_steps, self.repr_dim)).to(self.device)


class Prober(torch.nn.Module):
    def __init__(
        self,
        embedding: int,
        arch: str,
        output_shape: List[int],
    ):
        super().__init__()
        self.output_dim = np.prod(output_shape)
        self.output_shape = output_shape
        self.arch = arch

        arch_list = list(map(int, arch.split("-"))) if arch != "" else []
        f = [embedding] + arch_list + [self.output_dim]
        layers = []
        for i in range(len(f) - 2):
            layers.append(torch.nn.Linear(f[i], f[i + 1]))
            layers.append(torch.nn.ReLU(True))
        layers.append(torch.nn.Linear(f[-2], f[-1]))
        self.prober = torch.nn.Sequential(*layers)

    def forward(self, e):
        output = self.prober(e)
        return output

# --- JEPA Architecture ---

import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_channels=2, repr_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),  # (B, 2, 65, 65) -> (B, 32, 33, 33)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),             # (B, 32, 33, 33) -> (B, 64, 17, 17)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),            # (B, 64, 17, 17) -> (B, 128, 9, 9)
            nn.ReLU(),
        )
        self.fc = nn.Linear(128 * 9 * 9, repr_dim)  # (B, 128*9*9) -> (B, 256)

    def forward(self, x):
        x = self.conv(x)  # (B, 2, 65, 65) -> (B, 128, 9, 9)
        x = x.view(x.size(0), -1)  # Flatten: (B, 128, 9, 9) -> (B, 128*9*9)
        x = self.fc(x)  # (B, 128*9*9) -> (B, 256)
        return x


class Predictor(nn.Module):
    def __init__(self, repr_dim=256, action_dim=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(repr_dim + action_dim, repr_dim),  # (B, 256+2) -> (B, 256)
            nn.ReLU(),
            nn.Linear(repr_dim, repr_dim),              # (B, 256) -> (B, 256)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)  # Concatenate: (B, 256) + (B, 2) -> (B, 256+2)
        x = self.fc(x)  # (B, 256+2) -> (B, 256)
        return x


class JEPA(nn.Module):
    def __init__(self, input_channels=2, repr_dim=256, action_dim=2):
        super().__init__()
        self.encoder = Encoder(input_channels, repr_dim)
        self.predictor = Predictor(repr_dim, action_dim)

    def forward(self, states, actions):
        """
        Args:
            states: [B, T, 2, 65, 65] - Observations of the environment
            actions: [B, T-1, 2] - Action vectors (dx, dy)

        Returns:
            predictions: [B, T, 256] - Predicted representations for each timestep
        """
        batch_size, seq_len, _, _, _ = states.shape
        repr_dim = self.encoder.fc.out_features  # Dimension of the representation

        # Reshape states for the encoder
        states = states.view(-1, *states.shape[2:])  # (B, T, 2, 65, 65) -> (B*T, 2, 65, 65)

        # Encode all states
        encoded_states = self.encoder(states)  # (B*T, 2, 65, 65) -> (B*T, 256)

        # Reshape back to (B, T, D)
        encoded_states = encoded_states.view(batch_size, seq_len, repr_dim)  # (B*T, 256) -> (B, T, 256)

        # Initialize output tensor for predictions
        predictions = torch.zeros_like(encoded_states).to(states.device)  # (B, T, 256)

        # Use the first timestep's encoding as the initial state
        prev_state = encoded_states[:, 0]  # (B, 256)

        # Iterate through timesteps for prediction
        for t in range(seq_len - 1):
            predictions[:, t] = prev_state  # Store the current prediction
            prev_state = self.predictor(prev_state, actions[:, t])  # Predict the next state

        # Add the final prediction
        predictions[:, -1] = prev_state  # (B, 256)

        return predictions  # (B, T, 256)
