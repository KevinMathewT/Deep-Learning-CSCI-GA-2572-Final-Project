from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch

from configs import *


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


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(IN_C, 32, kernel_size=3, stride=2, padding=1),  # (B, 2, 65, 65) -> (B, 32, 33, 33)
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 33 * 33, EMBED_DIM)  # (B, 32 * 33 * 33) -> (B, EMBED_DIM)
    
    def forward(self, x):
        x = self.conv(x)  # (B, 2, 65, 65) -> (B, 32, 33, 33)
        x = self.flatten(x)  # (B, 32, 33, 33) -> (B, 32 * 33 * 33)
        x = self.fc(x)  # (B, 32 * 33 * 33) -> (B, EMBED_DIM)
        return x

class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(EMBED_DIM + ACTION_DIM, EMBED_DIM),  # (B, EMBED_DIM + ACTION_DIM) -> (B, EMBED_DIM)
            # nn.ReLU(),
            # nn.Linear(EMBED_DIM, EMBED_DIM),  # (B, EMBED_DIM) -> (B, EMBED_DIM)
        )

    def forward(self, sa):
        return self.fc(sa)  # (B * (T - 1), EMBED_DIM + ACTION_DIM) -> (B * (T - 1), EMBED_DIM)

class JEPA(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder()
        self.pred = Predictor()

    def forward(self, s, a):
        B, T, C, H, W = s.shape
        s = s.view(B * T, C, H, W)  # (B * T, 2, 65, 65)
        enc_s = self.enc(s)  # (B * T, EMBED_DIM)
        enc_s = enc_s.view(B, T, -1)  # (B, T, EMBED_DIM)
        preds = torch.zeros_like(enc_s)  # (B, T, EMBED_DIM)
        preds[:, 0, :] = enc_s[:, 0, :]  # (B, EMBED_DIM)
        sa_pairs = torch.cat([enc_s[:, :-1, :], a], dim=-1)  # (B, T-1, EMBED_DIM + ACTION_DIM)
        sa_pairs = sa_pairs.view(-1, EMBED_DIM + ACTION_DIM)  # (B * (T - 1), EMBED_DIM + ACTION_DIM)
        pred_states = self.pred(sa_pairs)  # (B * (T - 1), EMBED_DIM)
        preds[:, 1:, :] = pred_states.view(B, T-1, EMBED_DIM)  # (B, T, EMBED_DIM)
        # preds has first timestep prediction same as input itself - after that every timestep pred is derived from the previous
        return preds

