import math
from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch
import timm

from optimizer import get_optimizer, get_scheduler
from configs import JEPAConfig
from utils import create_minimal_feature_model

from typing import Dict

from loss import VICRegLoss

vicreg_loss = VICRegLoss()


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
            During training:
                states: [B, T, Ch, H, W]
            During inference:
                states: [B, 1, Ch, H, W]
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


class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def training_step(self, batch):
        raise NotImplementedError

    def validation_step(self, batch):
        raise NotImplementedError


# Encoder class adjusted to accept config
class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.repr_dim = config.embed_dim
        self.conv = nn.Sequential(
            nn.Conv2d(
                config.in_c, 32, kernel_size=3, stride=2, padding=1
            ),  # Conv2d layer
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 33 * 33, self.repr_dim)  # Fully connected layer

    def forward(self, x):
        # x: (B*T, C, H, W) = (B*T, 2, 65, 65)
        x = self.conv(x)  # (B*T, 2, 65, 65) -> (B*T, 32, 33, 33)
        x = self.flatten(x)  # (B*T, 32, 33, 33) -> (B*T, 32*33*33)
        x = self.fc(x)  # (B*T, 32*33*33) -> (B*T, embed_dim)
        return x  # (B*T, embed_dim)


# Predictor class adjusted to accept config
class Predictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.repr_dim = config.embed_dim
        self.action_proj = nn.Linear(
            config.action_dim, self.repr_dim
        )  # Project action to repr_dim
        # Optionally, you can add a projection for the state embedding
        # self.state_proj = nn.Linear(self.repr_dim, self.repr_dim)
        # For simplicity, we'll assume identity for state embedding

        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.repr_dim, self.repr_dim),  # Further processing
        )

    def forward(self, s_embed, a):
        # s_embed: (B*(T-1), repr_dim)
        # a: (B*(T-1), action_dim)
        a_proj = self.action_proj(a)  # (B*(T-1), repr_dim)
        # s_proj = self.state_proj(s_embed)  # If projecting s_embed
        s_proj = s_embed  # If not projecting s_embed

        x = s_proj + a_proj  # Element-wise addition: (B*(T-1), repr_dim)
        x = self.fc(x)  # Further processing: (B*(T-1), repr_dim)
        return x  # (B*(T-1), repr_dim)


# JEPA model adjusted to accept config
class JEPA(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.enc = Encoder(config)
        self.pred = Predictor(config)
        self.optimizer = get_optimizer(config, self.parameters())
        self.scheduler = get_scheduler(self.optimizer, config)

        self.config = config
        self.repr_dim = config.embed_dim

    def forward(self, states, actions, teacher_forcing=True):
        B, _, C, H, W = states.shape  # states: (B, T, C, H, W)
        T = actions.shape[1] + 1  # Number of timesteps | actions: (B, T-1, action_dim)

        if teacher_forcing:
            states = states.view(B * T, C, H, W)  # Reshape to (B*T, C, H, W)
            enc_states = self.enc(states)  # (B*T, embed_dim)
            enc_states = enc_states.view(B, T, -1)  # (B, T, embed_dim)
            preds = torch.zeros_like(enc_states)  # preds: (B, T, embed_dim)
            preds[:, 0, :] = enc_states[:, 0, :]  # Initialize first timestep

            # Prepare inputs for the predictor
            states_embed = enc_states[:, :-1, :]  # (B, T-1, embed_dim)
            states_embed = states_embed.reshape(
                -1, self.config.embed_dim
            )  # (B*(T-1), embed_dim)
            actions = actions.reshape(
                -1, self.config.action_dim
            )  # (B*(T-1), action_dim)

            pred_states = self.pred(states_embed, actions)  # (B*(T-1), embed_dim)
            pred_states = pred_states.view(
                B, T - 1, self.config.embed_dim
            )  # (B, T-1, embed_dim)
            preds[:, 1:, :] = pred_states  # Assign predictions to preds

            return (
                preds,
                enc_states,
            )  # preds: (B, T, embed_dim), enc_states: (B, T, embed_dim)

        else:
            states_0 = states[:, 0, :, :, :]  # (B, C, H, W)
            enc_state = self.enc(states_0)  # (B, embed_dim)
            preds = [enc_state]  # List to store predictions

            for t in range(1, T):
                action_t_minus1 = actions[:, t - 1, :]  # (B, action_dim)
                state_embed_t_minus1 = preds[-1]  # Use the last predicted embedding
                pred_state = self.pred(
                    state_embed_t_minus1, action_t_minus1
                )  # (B, embed_dim)
                preds.append(pred_state)

            # Stack predictions and true encodings along the time dimension
            preds = torch.stack(preds, dim=1)  # (B, T, embed_dim)

            return preds

    def compute_mse_loss(self, preds, enc_s):
        # preds, enc_s: (B, T, embed_dim)
        loss = F.mse_loss(
            preds[:, 1:], enc_s[:, 1:]
        )  # Compute MSE loss for timesteps 1 to T-1
        return loss

    def compute_vicreg_loss(self, preds, enc_s, gamma=1.0, epsilon=1e-4):
        """
        Compute VICReg loss with invariance, variance, and covariance terms.

        Args:
            preds: Predicted embeddings from the predictor. Shape (B, T, embed_dim).
            enc_s: Target embeddings from the encoder. Shape (B, T, embed_dim).
            gamma: Target standard deviation for variance term.
            epsilon: Small value to avoid numerical instability.

        Returns:
            vicreg_loss: The combined VICReg loss.
        """

        # taking from config
        lambda_invariance = self.config.vicreg_loss.lambda_invariance
        mu_variance = self.config.vicreg_loss.mu_variance
        nu_covariance = self.config.vicreg_loss.nu_covariance

        preds, enc_s = preds[:, 1:], enc_s[:, 1:]

        # Flatten temporal dimensions for batch processing
        B, T, embed_dim = preds.shape
        Z = preds.reshape(B * T, embed_dim)  # Predicted embeddings
        Z_prime = enc_s.reshape(B * T, embed_dim)  # Target embeddings

        # --- Invariance Term ---
        invariance_loss = torch.mean(
            torch.sum((Z - Z_prime) ** 2, dim=1)
        )  # Mean squared Euclidean distance

        # --- Variance Term ---
        # Compute standard deviation along the batch dimension
        std_Z = torch.sqrt(Z.var(dim=0, unbiased=False) + epsilon)
        std_Z_prime = torch.sqrt(Z_prime.var(dim=0, unbiased=False) + epsilon)

        variance_loss = torch.mean(F.relu(gamma - std_Z)) + torch.mean(
            F.relu(gamma - std_Z_prime)
        )

        # --- Covariance Term ---
        # Center the embeddings
        Z_centered = Z - Z.mean(dim=0, keepdim=True)
        Z_prime_centered = Z_prime - Z_prime.mean(dim=0, keepdim=True)

        # Compute covariance matrices
        cov_Z = (Z_centered.T @ Z_centered) / (B * T - 1)
        cov_Z_prime = (Z_prime_centered.T @ Z_prime_centered) / (B * T - 1)

        # Sum of squared off-diagonal elements
        cov_loss_Z = torch.sum(cov_Z**2) - torch.sum(torch.diag(cov_Z) ** 2)
        cov_loss_Z_prime = torch.sum(cov_Z_prime**2) - torch.sum(
            torch.diag(cov_Z_prime) ** 2
        )

        covariance_loss = cov_loss_Z + cov_loss_Z_prime

        # --- Total VICReg Loss ---
        vicreg_loss = (
            lambda_invariance * invariance_loss
            + mu_variance * variance_loss
            + nu_covariance * covariance_loss
        )

        return vicreg_loss, invariance_loss, variance_loss, covariance_loss

    def training_step(self, batch, device):
        states, actions = batch.states.to(device, non_blocking=True), batch.actions.to(
            device, non_blocking=True
        )
        preds, enc_s = self.forward(states, actions)

        # loss = self.compute_mse_loss(preds, enc_s) # Normal Loss Calculation
        loss, invariance_loss, variance_loss, covariance_loss = (
            self.compute_vicreg_loss(preds, enc_s)
        )  # VICReg Loss Calculation

        self.optimizer.zero_grad()
        loss.backward()

        # Compute grad_norm without clipping
        grad_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                grad_norm += param_norm.item() ** 2
        grad_norm = grad_norm**0.5

        self.optimizer.step()
        self.scheduler.step()  # Step the scheduler

        learning_rate = self.optimizer.param_groups[0]["lr"]

        # Compute the absolute value of the action weights
        action_weight = (
            self.pred.action_proj.weight
        )  # Get weights from the action projection layer
        action_weight_abs = (
            action_weight.abs().mean().item()
        )  # Compute the mean absolute value

        # Compute deviation from identity for fc layer
        fc_weight = self.pred.fc[
            1
        ].weight  # Get the weights of the Linear layer inside fc
        identity_matrix = torch.eye(fc_weight.size(0), fc_weight.size(1)).to(
            fc_weight.device
        )
        deviation_from_identity = torch.norm(
            fc_weight - identity_matrix, p="fro"
        ) / torch.norm(identity_matrix, p="fro")

        # Prepare the output dictionary
        output = {
            "loss": loss.item(),
            "grad_norm": grad_norm,
            "learning_rate": learning_rate,
            "action_weight_abs": action_weight_abs,
            "deviation_from_identity_pred_final_proj": deviation_from_identity.item(),  # Log the deviation
            "invariance_loss": invariance_loss,
            "variance_loss": variance_loss,
            "covariance_loss": covariance_loss,
            # Add more loggable values here if needed
        }

        # Non-loggable data
        output["non_logs"] = {
            "states": states.detach(),
            "actions": actions.detach(),
            "enc_embeddings": enc_s.detach(),
            "pred_embeddings": preds.detach(),
        }

        return output

    def validation_step(self, batch):
        states, actions = batch.states, batch.actions
        preds, enc_s = self.forward(states, actions)
        loss = self.compute_mse_loss(preds, enc_s)
        learning_rate = self.optimizer.param_groups[0]["lr"]

        # Compute the absolute value of the action weights
        action_weight = (
            self.pred.action_proj.weight
        )  # Get weights from the action projection layer
        action_weight_abs = (
            action_weight.abs().mean().item()
        )  # Compute the mean absolute value

        # Compute deviation from identity for fc layer
        fc_weight = self.pred.fc[
            1
        ].weight  # Get the weights of the Linear layer inside fc
        identity_matrix = torch.eye(fc_weight.size(0), fc_weight.size(1)).to(
            fc_weight.device
        )
        deviation_from_identity = torch.norm(
            fc_weight - identity_matrix, p="fro"
        ) / torch.norm(identity_matrix, p="fro")

        # Prepare the output dictionary
        output = {
            "loss": loss.item(),
            "learning_rate": learning_rate,
            "action_weight_abs": action_weight_abs,
            "deviation_from_identity_pred_final_proj": deviation_from_identity.item(),  # Log the deviation
        }

        # Non-loggable data
        output["non_logs"] = {
            "states": states.detach(),
            "actions": actions.detach(),
            "enc_embeddings": enc_s.detach(),
            "pred_embeddings": preds.detach(),
        }

        return output


# AdversarialJEPA model adjusted to accept config
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (N, input_dim)
        return self.net(x)  # (N, input_dim) -> (N, 1)


class ActionRegularizer(nn.Module):
    def __init__(self, embed_dim, action_dim, action_reg_hidden_dim):
        super().__init__()
        self.action_reg_net = nn.Sequential(
            nn.Linear(embed_dim, action_reg_hidden_dim),
            nn.ReLU(),
            nn.Linear(action_reg_hidden_dim, action_dim),
        )

    def forward(self, states_embed, pred_states):
        # Calculate embedding differences
        embedding_diff = (
            pred_states - states_embed
        )  # Difference between input and output of predictor

        # Predict actions from embedding differences
        predicted_actions = self.action_reg_net(embedding_diff)  # (B*(T-1), action_dim)
        return predicted_actions


class AdversarialJEPAWithRegularization(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.enc = Encoder(config)
        self.pred = Predictor(config)
        self.disc = Discriminator(config.embed_dim)
        self.action_reg_net = ActionRegularizer(
            config.embed_dim, config.action_dim, config.action_reg_hidden_dim
        )

        self.gen_opt = get_optimizer(
            config,
            list(self.enc.parameters())
            + list(self.pred.parameters())
            + list(self.action_reg_net.parameters()),
        )
        self.disc_opt = get_optimizer(config, self.disc.parameters())

        self.gen_sched = get_scheduler(self.gen_opt, config)
        self.disc_sched = get_scheduler(self.disc_opt, config)

        self.config = config
        self.repr_dim = config.embed_dim

    def forward(self, states, actions, teacher_forcing=True):
        B, _, C, H, W = states.shape  # states: (B, T, C, H, W)
        T = actions.shape[1] + 1  # Number of timesteps | actions: (B, T-1, action_dim)

        if teacher_forcing:
            states = states.view(B * T, C, H, W)  # Reshape to (B*T, C, H, W)
            enc_states = self.enc(states)  # (B*T, embed_dim)
            enc_states = enc_states.view(B, T, -1)  # (B, T, embed_dim)
            preds = torch.zeros_like(enc_states)  # preds: (B, T, embed_dim)
            preds[:, 0, :] = enc_states[:, 0, :]  # Initialize first timestep

            # Prepare inputs for the predictor
            states_embed = enc_states[:, :-1, :]  # (B, T-1, embed_dim)
            states_embed = states_embed.reshape(
                -1, self.config.embed_dim
            )  # (B*(T-1), embed_dim)
            actions = actions.reshape(
                -1, self.config.action_dim
            )  # (B*(T-1), action_dim)

            pred_states = self.pred(states_embed, actions)  # (B*(T-1), embed_dim)
            pred_states = pred_states.view(
                B, T - 1, self.config.embed_dim
            )  # (B, T-1, embed_dim)
            preds[:, 1:, :] = pred_states  # Assign predictions to preds

            return (
                preds,
                enc_states,
            )  # preds: (B, T, embed_dim), enc_states: (B, T, embed_dim)

        else:
            states_0 = states[:, 0, :, :, :]  # (B, C, H, W)
            enc_state = self.enc(states_0)  # (B, embed_dim)
            preds = [enc_state]  # List to store predictions

            for t in range(1, T):
                action_t_minus1 = actions[:, t - 1, :]  # (B, action_dim)
                state_embed_t_minus1 = preds[-1]  # Use the last predicted embedding
                pred_state = self.pred(
                    state_embed_t_minus1, action_t_minus1
                )  # (B, embed_dim)
                preds.append(pred_state)

            # Stack predictions and true encodings along the time dimension
            preds = torch.stack(preds, dim=1)  # (B, T, embed_dim)

            return preds

    def compute_mse_loss(self, preds, enc_s):
        # preds, enc_s: (B, T, embed_dim)
        loss = F.mse_loss(
            preds[:, 1:], enc_s[:, 1:]
        )  # Compute MSE loss for timesteps 1 to T-1
        return loss

    def compute_regularization_loss(self, states_embed, pred_states, actions):
        """
        Computes the regularization loss based on the embedding difference and actions.
        """
        # Predict actions from embedding differences
        predicted_actions = self.action_reg_net(
            states_embed, pred_states
        )  # (B*(T-1), action_dim)
        actions = actions.view(
            -1, self.config.action_dim
        )  # Flatten actions to (B*(T-1), action_dim)

        # Compute MSE loss between predicted and actual actions
        reg_loss = F.mse_loss(predicted_actions, actions)
        return reg_loss

    def compute_vicreg_loss(self, preds, enc_s, gamma=1.0, epsilon=1e-4):
        """
        Compute VICReg loss with invariance, variance, and covariance terms.

        Args:
            preds: Predicted embeddings from the predictor. Shape (B, T, embed_dim).
            enc_s: Target embeddings from the encoder. Shape (B, T, embed_dim).
            gamma: Target standard deviation for variance term.
            epsilon: Small value to avoid numerical instability.

        Returns:
            vicreg_loss: The combined VICReg loss.
        """

        # taking from config
        lambda_invariance = self.config.vicreg_loss.lambda_invariance
        mu_variance = self.config.vicreg_loss.mu_variance
        nu_covariance = self.config.vicreg_loss.nu_covariance

        preds, enc_s = preds[:, 1:], enc_s[:, 1:]

        # Flatten temporal dimensions for batch processing
        B, T, embed_dim = preds.shape
        Z = preds.reshape(B * T, embed_dim)  # Predicted embeddings
        Z_prime = enc_s.reshape(B * T, embed_dim)  # Target embeddings

        # --- Invariance Term ---
        invariance_loss = torch.mean(
            torch.sum((Z - Z_prime) ** 2, dim=1)
        )  # Mean squared Euclidean distance

        # --- Variance Term ---
        # Compute standard deviation along the batch dimension
        std_Z = torch.sqrt(Z.var(dim=0, unbiased=False) + epsilon)
        std_Z_prime = torch.sqrt(Z_prime.var(dim=0, unbiased=False) + epsilon)

        variance_loss = torch.mean(F.relu(gamma - std_Z)) + torch.mean(
            F.relu(gamma - std_Z_prime)
        )

        # --- Covariance Term ---
        # Center the embeddings
        Z_centered = Z - Z.mean(dim=0, keepdim=True)
        Z_prime_centered = Z_prime - Z_prime.mean(dim=0, keepdim=True)

        # Compute covariance matrices
        cov_Z = (Z_centered.T @ Z_centered) / (B * T - 1)
        cov_Z_prime = (Z_prime_centered.T @ Z_prime_centered) / (B * T - 1)

        # Sum of squared off-diagonal elements
        cov_loss_Z = torch.sum(cov_Z**2) - torch.sum(torch.diag(cov_Z) ** 2)
        cov_loss_Z_prime = torch.sum(cov_Z_prime**2) - torch.sum(
            torch.diag(cov_Z_prime) ** 2
        )

        covariance_loss = cov_loss_Z + cov_loss_Z_prime

        # --- Total VICReg Loss ---
        vicreg_loss = (
            lambda_invariance * invariance_loss
            + mu_variance * variance_loss
            + nu_covariance * covariance_loss
        )

        return vicreg_loss, invariance_loss, variance_loss, covariance_loss

    def compute_discriminator_loss(self, preds, enc_s):
        """
        Computes the discriminator loss using real and fake embeddings.
        """
        # Extract embeddings
        real_embeddings = enc_s[:, 1:].reshape(-1, self.config.embed_dim)
        fake_embeddings = preds[:, 1:].detach().reshape(-1, self.config.embed_dim)

        # Create labels
        real_labels = torch.ones(
            real_embeddings.size(0), 1, device=real_embeddings.device
        )
        fake_labels = torch.zeros(
            fake_embeddings.size(0), 1, device=fake_embeddings.device
        )

        # Discriminator predictions
        real_predictions = self.disc(real_embeddings)
        fake_predictions = self.disc(fake_embeddings)

        # Compute binary cross-entropy losses
        disc_loss_real = F.binary_cross_entropy(real_predictions, real_labels)
        disc_loss_fake = F.binary_cross_entropy(fake_predictions, fake_labels)

        # Average the losses
        disc_loss = (disc_loss_real + disc_loss_fake) / 2
        return disc_loss

    def compute_generator_loss(self, preds):
        fake_embeddings = preds[:, 1:].detach().reshape(-1, self.config.embed_dim)
        real_labels = torch.ones(
            fake_embeddings.size(0), 1, device=fake_embeddings.device
        )
        gen_predictions = self.disc(preds[:, 1:].reshape(-1, self.config.embed_dim))
        gen_loss = F.binary_cross_entropy(gen_predictions, real_labels)

        return gen_loss

    def training_step(self, batch, device):
        states, actions = batch.states.to(device, non_blocking=True), batch.actions.to(
            device, non_blocking=True
        )

        # Step 1: Train the discriminator
        with torch.no_grad():  # Detach generator computations to avoid unnecessary graph retention
            preds, enc_s = self.forward(states, actions)
        disc_loss = self.compute_discriminator_loss(preds, enc_s)
        self.disc_opt.zero_grad()
        disc_loss.backward()
        self.disc_opt.step()

        # Step 2: Train the generator
        preds, enc_s = self.forward(states, actions)  # Perform forward pass again
        gen_loss = self.compute_generator_loss(preds)
        reg_loss = self.compute_regularization_loss(
            enc_s[:, :-1].reshape(-1, self.config.embed_dim),
            preds[:, 1:].reshape(-1, self.config.embed_dim),
            actions.reshape(-1, self.config.action_dim),
        )
        vicreg_loss, invariance_loss, variance_loss, covariance_loss = (
            self.compute_vicreg_loss(preds, enc_s)
        )

        total_loss = (
            self.config.delta_gen * gen_loss
            + self.config.lambda_reg * reg_loss
            + vicreg_loss
        )

        self.gen_opt.zero_grad()
        total_loss.backward()
        self.gen_opt.step()

        # Update learning rate schedulers
        self.gen_sched.step()
        self.disc_sched.step()

        learning_rate = self.gen_opt.param_groups[0]["lr"]
        disc_learning_rate = self.disc_opt.param_groups[0]["lr"]

        # Compute the absolute value of the action weights
        action_weight = (
            self.pred.action_proj.weight
        )  # Get weights from the action projection layer
        action_weight_abs = (
            action_weight.abs().mean().item()
        )  # Compute the mean absolute value

        # Compute deviation from identity for fc layer
        fc_weight = self.pred.fc[
            1
        ].weight  # Get the weights of the Linear layer inside fc
        identity_matrix = torch.eye(fc_weight.size(0), fc_weight.size(1)).to(
            fc_weight.device
        )
        deviation_from_identity = torch.norm(
            fc_weight - identity_matrix, p="fro"
        ) / torch.norm(identity_matrix, p="fro")

        # Prepare the output dictionary
        output = {
            "loss": total_loss.item(),
            "gen_loss": gen_loss.item(),
            "reg_loss": reg_loss.item(),
            "vicreg_loss": vicreg_loss.item(),
            "invariance_loss": invariance_loss.item(),
            "variance_loss": variance_loss.item(),
            "covariance_loss": covariance_loss.item(),
            "disc_loss": disc_loss.item(),
            "learning_rate": learning_rate,
            "disc_learning_rate": disc_learning_rate,
            "action_weight_abs": action_weight_abs,
            "deviation_from_identity_pred_final_proj": deviation_from_identity.item(),  # Log the deviation
        }

        # Non-loggable data
        output["non_logs"] = {
            "states": states.detach(),
            "actions": actions.detach(),
            "enc_embeddings": enc_s.detach(),
            "pred_embeddings": preds.detach(),
        }

        return output

    def validation_step(self, batch):
        states, actions = batch.states, batch.actions
        preds, enc_s = self.forward(states, actions)
        loss = self.compute_mse_loss(preds, enc_s)

        # Compute the absolute value of the action weights
        action_weight = (
            self.pred.action_proj.weight
        )  # Get weights from the action projection layer
        action_weight_abs = (
            action_weight.abs().mean().item()
        )  # Compute the mean absolute value

        # Compute deviation from identity for fc layer
        fc_weight = self.pred.fc[
            1
        ].weight  # Get the weights of the Linear layer inside fc
        identity_matrix = torch.eye(fc_weight.size(0), fc_weight.size(1)).to(
            fc_weight.device
        )
        deviation_from_identity = torch.norm(
            fc_weight - identity_matrix, p="fro"
        ) / torch.norm(identity_matrix, p="fro")

        # Prepare the output dictionary
        output = {
            "loss": loss.item(),
            "action_weight_abs": action_weight_abs,
            "deviation_from_identity_pred_final_proj": deviation_from_identity.item(),  # Log the deviation
        }

        # Non-loggable data
        output["non_logs"] = {
            "states": states.detach(),
            "actions": actions.detach(),
            "enc_embeddings": enc_s.detach(),
            "pred_embeddings": preds.detach(),
        }

        return output


class AdversarialJEPA(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.enc = Encoder(config)
        self.pred = Predictor(config)
        self.disc = Discriminator(config.embed_dim)
        self.opt = get_optimizer(config, self.parameters())
        self.disc_opt = get_optimizer(config, self.disc.parameters())

        self.scheduler = get_scheduler(self.opt, config)
        self.scheduler_disc = get_scheduler(self.disc_opt, config)

        self.config = config
        self.repr_dim = config.embed_dim

    def forward(self, states, actions, teacher_forcing=True):
        B, _, C, H, W = states.shape  # states: (B, T, C, H, W)
        T = actions.shape[1] + 1  # Number of timesteps | actions: (B, T-1, action_dim)

        if teacher_forcing:
            states = states.view(B * T, C, H, W)  # Reshape to (B*T, C, H, W)
            enc_states = self.enc(states)  # (B*T, embed_dim)
            enc_states = enc_states.view(B, T, -1)  # (B, T, embed_dim)
            preds = torch.zeros_like(enc_states)  # preds: (B, T, embed_dim)
            preds[:, 0, :] = enc_states[:, 0, :]  # Initialize first timestep

            # Prepare inputs for the predictor
            states_embed = enc_states[:, :-1, :]  # (B, T-1, embed_dim)
            states_embed = states_embed.reshape(
                -1, self.config.embed_dim
            )  # (B*(T-1), embed_dim)
            actions = actions.reshape(
                -1, self.config.action_dim
            )  # (B*(T-1), action_dim)

            pred_states = self.pred(states_embed, actions)  # (B*(T-1), embed_dim)
            pred_states = pred_states.view(
                B, T - 1, self.config.embed_dim
            )  # (B, T-1, embed_dim)
            preds[:, 1:, :] = pred_states  # Assign predictions to preds

            return (
                preds,
                enc_states,
            )  # preds: (B, T, embed_dim), enc_states: (B, T, embed_dim)

        else:
            states_0 = states[:, 0, :, :, :]  # (B, C, H, W)
            enc_state = self.enc(states_0)  # (B, embed_dim)
            preds = [enc_state]  # List to store predictions

            for t in range(1, T):
                action_t_minus1 = actions[:, t - 1, :]  # (B, action_dim)
                state_embed_t_minus1 = preds[-1]  # Use the last predicted embedding
                pred_state = self.pred(
                    state_embed_t_minus1, action_t_minus1
                )  # (B, embed_dim)
                preds.append(pred_state)

            # Stack predictions and true encodings along the time dimension
            preds = torch.stack(preds, dim=1)  # (B, T, embed_dim)

            return preds

    def compute_mse_losses(self, preds, enc_s):
        real_embeds = enc_s[:, 1:].reshape(
            -1, self.config.embed_dim
        )  # (B*(T-1), embed_dim)
        fake_embeds = (
            preds[:, 1:].detach().reshape(-1, self.config.embed_dim)
        )  # (B*(T-1), embed_dim)

        real_labels = torch.ones(real_embeds.size(0), 1).to(
            real_embeds.device
        )  # (B*(T-1), 1)
        fake_labels = torch.zeros(fake_embeds.size(0), 1).to(
            fake_embeds.device
        )  # (B*(T-1), 1)

        real_preds = self.disc(real_embeds)  # (B*(T-1), 1)
        fake_preds = self.disc(fake_embeds)  # (B*(T-1), 1)

        disc_loss_real = F.binary_cross_entropy(real_preds, real_labels)
        disc_loss_fake = F.binary_cross_entropy(fake_preds, fake_labels)
        disc_loss = (disc_loss_real + disc_loss_fake) / 2

        # Generator (Predictor) loss
        gen_preds = self.disc(
            preds[:, 1:].detach().reshape(-1, self.config.embed_dim)
        )  # (B*(T-1), 1)
        pred_labels = torch.ones(gen_preds.size(0), 1).to(
            gen_preds.device
        )  # (B*(T-1), 1)
        gen_loss = F.binary_cross_entropy(gen_preds, pred_labels)

        # Reconstruction loss
        rec_loss = F.mse_loss(preds[:, 1:], enc_s[:, 1:])  # (B, T-1, embed_dim)

        # Total loss
        total_loss = gen_loss + rec_loss

        return disc_loss, total_loss

    def training_step(self, batch, device):
        states, actions = batch.states.to(device, non_blocking=True), batch.actions.to(
            device, non_blocking=True
        )
        preds, enc_s = self.forward(states, actions)

        # Update Discriminator
        disc_loss, total_loss = self.compute_mse_losses(preds, enc_s)
        self.disc_opt.zero_grad()
        disc_loss.backward(retain_graph=True)
        self.disc_opt.step()

        # Update Encoder and Predictor
        self.opt.zero_grad()
        total_loss.backward()

        # Compute grad_norm without clipping
        grad_norm = 0.0
        for p in list(self.enc.parameters()) + list(self.pred.parameters()):
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                grad_norm += param_norm.item() ** 2
        grad_norm = grad_norm**0.5

        self.opt.step()

        self.scheduler.step()  # Step the scheduler
        self.scheduler_disc.step()  # Step the scheduler

        learning_rate = self.opt.param_groups[0][
            "lr"
        ]  # Assuming same LR for all optimizers
        disc_learning_rate = self.disc_opt.param_groups[0][
            "lr"
        ]  # Assuming same LR for all optimizers

        # Compute the absolute value of the action weights
        action_weight = (
            self.pred.action_proj.weight
        )  # Get weights from the action projection layer
        action_weight_abs = (
            action_weight.abs().mean().item()
        )  # Compute the mean absolute value

        # Compute deviation from identity for fc layer
        fc_weight = self.pred.fc[
            1
        ].weight  # Get the weights of the Linear layer inside fc
        identity_matrix = torch.eye(fc_weight.size(0), fc_weight.size(1)).to(
            fc_weight.device
        )
        deviation_from_identity = torch.norm(
            fc_weight - identity_matrix, p="fro"
        ) / torch.norm(identity_matrix, p="fro")

        # Prepare the output dictionary
        output = {
            "loss": total_loss.item(),
            "disc_loss": disc_loss.item(),
            "grad_norm": grad_norm,
            "learning_rate": learning_rate,
            "disc_learning_rate": disc_learning_rate,
            "action_weight_abs": action_weight_abs,
            "deviation_from_identity_pred_final_proj": deviation_from_identity.item(),  # Log the deviation
        }

        # Non-loggable data
        output["non_logs"] = {
            "states": states.detach(),
            "actions": actions.detach(),
            "enc_embeddings": enc_s.detach(),
            "pred_embeddings": preds.detach(),
        }

        return output

    def validation_step(self, batch):
        states, actions = batch.states, batch.actions
        preds, enc_s = self.forward(states, actions)
        loss = self.compute_mse_loss(preds, enc_s)
        learning_rate = self.optimizer.param_groups[0]["lr"]

        # Compute the absolute value of the action weights
        action_weight = (
            self.pred.action_proj.weight
        )  # Get weights from the action projection layer
        action_weight_abs = (
            action_weight.abs().mean().item()
        )  # Compute the mean absolute value

        # Compute deviation from identity for fc layer
        fc_weight = self.pred.fc[
            1
        ].weight  # Get the weights of the Linear layer inside fc
        identity_matrix = torch.eye(fc_weight.size(0), fc_weight.size(1)).to(
            fc_weight.device
        )
        deviation_from_identity = torch.norm(
            fc_weight - identity_matrix, p="fro"
        ) / torch.norm(identity_matrix, p="fro")

        # Prepare the output dictionary
        output = {
            "loss": loss.item(),
            "learning_rate": learning_rate,
            "action_weight_abs": action_weight_abs,
            "deviation_from_identity_pred_final_proj": deviation_from_identity.item(),  # Log the deviation
        }

        # Non-loggable data
        output["non_logs"] = {
            "states": states.detach(),
            "actions": actions.detach(),
            "enc_embeddings": enc_s.detach(),
            "pred_embeddings": preds.detach(),
        }

        return output


# InfoMaxJEPA model adjusted to accept config
class InfoMaxJEPA(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.enc = Encoder(config)
        self.pred = Predictor(config)
        self.optimizer = get_optimizer(config, self.parameters())
        self.scheduler = get_scheduler(self.optimizer, config)

        self.config = config
        self.repr_dim = config.embed_dim

    def forward(self, s, a):
        B, T, C, H, W = s.shape  # s: (B, T, C, H, W)
        s = s.view(B * T, C, H, W)  # (B*T, C, H, W)
        enc_s = self.enc(s)  # (B*T, embed_dim)
        enc_s = enc_s.view(B, T, -1)  # (B, T, embed_dim)
        sa_pairs = torch.cat(
            [enc_s[:, :-1, :], a], dim=-1
        )  # (B, T-1, embed_dim + action_dim)
        sa_pairs = sa_pairs.view(
            -1, self.config.embed_dim + self.config.action_dim
        )  # (B*(T-1), embed_dim + action_dim)
        pred_states = self.pred(sa_pairs)  # (B*(T-1), embed_dim)
        pred_states = pred_states.view(
            B, T - 1, self.config.embed_dim
        )  # (B, T-1, embed_dim)
        return pred_states, enc_s[:, 1:, :]  # Return predictions and targets

    def compute_mse_loss(self, preds, targets):
        B, T_minus_1, D = preds.shape  # preds: (B, T-1, embed_dim)
        preds = preds.reshape(B * T_minus_1, D)  # (B*(T-1), embed_dim)
        targets = targets.reshape(B * T_minus_1, D)  # (B*(T-1), embed_dim)

        preds = F.normalize(preds, dim=-1)  # Normalize embeddings
        targets = F.normalize(targets, dim=-1)

        logits = torch.matmul(preds, targets.t())  # (B*(T-1), B*(T-1))
        labels = torch.arange(B * T_minus_1).to(preds.device)  # (B*(T-1))

        temperature = 0.07
        logits = logits / temperature

        loss = F.cross_entropy(logits, labels)
        return loss

    def training_step(self, batch, device):
        states, actions = batch.states.to(device, non_blocking=True), batch.actions.to(
            device, non_blocking=True
        )
        preds, targets = self.forward(states, actions)
        loss, temperature = self.compute_mse_loss(preds, targets)
        self.optimizer.zero_grad()
        loss.backward()

        # Compute grad_norm without clipping
        grad_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                grad_norm += param_norm.item() ** 2
        grad_norm = grad_norm**0.5

        self.optimizer.step()
        self.scheduler.step()  # Step the scheduler

        learning_rate = self.optimizer.param_groups[0]["lr"]

        # Prepare the output dictionary
        output = {
            "loss": loss.item(),
            "grad_norm": grad_norm,
            "learning_rate": learning_rate,
            "temperature": temperature,
        }

        # Non-loggable data
        output["non_logs"] = {
            "states": states[:, 1:, :, :, :].detach(),
            "actions": actions.detach(),
            "enc_embeddings": targets.detach(),
            "pred_embeddings": preds.detach(),
        }

        return output

    def validation_step(self, batch):
        states, actions = batch.states, batch.actions
        preds, targets = self.forward(states, actions)
        loss, temperature = self.compute_mse_loss(preds, targets)
        learning_rate = self.optimizer.param_groups[0]["lr"]

        # Prepare the output dictionary
        output = {
            "loss": loss.item(),
            "learning_rate": learning_rate,
            "temperature": temperature,
        }

        # Non-loggable data
        output["non_logs"] = {
            "states": states[:, 1:, :, :, :].detach(),
            "actions": actions.detach(),
            "enc_embeddings": targets.detach(),
            "pred_embeddings": preds.detach(),
        }

        return output


# JEPA model adjusted to accept config
class ActionRegularizationJEPA(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.enc = Encoder(config)
        self.pred = Predictor(config)
        self.action_reg_net = nn.Sequential(
            nn.Linear(config.embed_dim, config.action_reg_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.action_reg_hidden_dim, config.action_dim),
        )  # Small network for action prediction from embedding differences

        self.optimizer = get_optimizer(config, self.parameters())
        self.scheduler = get_scheduler(self.optimizer, config)

        self.config = config
        self.repr_dim = config.embed_dim

    def forward(self, states, actions, teacher_forcing=True):
        B, _, C, H, W = states.shape  # states: (B, T, C, H, W)
        T = actions.shape[1] + 1  # Number of timesteps | actions: (B, T-1, action_dim)

        if teacher_forcing:
            states = states.view(B * T, C, H, W)  # Reshape to (B*T, C, H, W)
            enc_states = self.enc(states)  # (B*T, embed_dim)
            enc_states = enc_states.view(B, T, -1)  # (B, T, embed_dim)
            preds = torch.zeros_like(enc_states)  # preds: (B, T, embed_dim)
            preds[:, 0, :] = enc_states[:, 0, :]  # Initialize first timestep

            # Prepare inputs for the predictor
            states_embed = enc_states[:, :-1, :]  # (B, T-1, embed_dim)
            states_embed = states_embed.reshape(
                -1, self.config.embed_dim
            )  # (B*(T-1), embed_dim)
            actions = actions.reshape(
                -1, self.config.action_dim
            )  # (B*(T-1), action_dim)

            pred_states = self.pred(states_embed, actions)  # (B*(T-1), embed_dim)
            pred_states = pred_states.view(
                B, T - 1, self.config.embed_dim
            )  # (B, T-1, embed_dim)
            preds[:, 1:, :] = pred_states  # Assign predictions to preds

            return (
                preds,
                enc_states,
            )  # preds: (B, T, embed_dim), enc_states: (B, T, embed_dim)

        else:
            states_0 = states[:, 0, :, :, :]  # (B, C, H, W)
            enc_state = self.enc(states_0)  # (B, embed_dim)
            preds = [enc_state]  # List to store predictions

            for t in range(1, T):
                action_t_minus1 = actions[:, t - 1, :]  # (B, action_dim)
                state_embed_t_minus1 = preds[-1]  # Use the last predicted embedding
                pred_state = self.pred(
                    state_embed_t_minus1, action_t_minus1
                )  # (B, embed_dim)
                preds.append(pred_state)

            # Stack predictions and true encodings along the time dimension
            preds = torch.stack(preds, dim=1)  # (B, T, embed_dim)

            return preds

    def compute_mse_loss(self, preds, enc_s):
        # preds, enc_s: (B, T, embed_dim)
        loss = F.mse_loss(
            preds[:, 1:], enc_s[:, 1:]
        )  # Compute MSE loss for timesteps 1 to T-1
        return loss

    def compute_regularization_loss(self, states_embed, pred_states, actions):
        """
        Computes the regularization loss based on the embedding difference and actions.
        """
        # Calculate embedding differences
        embedding_diff = (
            pred_states - states_embed
        )  # Difference between input and output of predictor
        actions = actions.view(
            -1, self.config.action_dim
        )  # Flatten actions to (B*(T-1), action_dim)

        # Predict actions from embedding differences
        predicted_actions = self.action_reg_net(embedding_diff)  # (B*(T-1), action_dim)

        # Compute MSE loss between predicted and actual actions
        reg_loss = F.mse_loss(predicted_actions, actions)
        return reg_loss

    def compute_vicreg_loss(self, preds, enc_s, gamma=1.0, epsilon=1e-4):
        """
        Compute VICReg loss with invariance, variance, and covariance terms.

        Args:
            preds: Predicted embeddings from the predictor. Shape (B, T, embed_dim).
            enc_s: Target embeddings from the encoder. Shape (B, T, embed_dim).
            gamma: Target standard deviation for variance term.
            epsilon: Small value to avoid numerical instability.

        Returns:
            vicreg_loss: The combined VICReg loss.
        """

        # taking from config
        lambda_invariance = self.config.vicreg_loss.lambda_invariance
        mu_variance = self.config.vicreg_loss.mu_variance
        nu_covariance = self.config.vicreg_loss.nu_covariance

        preds, enc_s = preds[:, 1:], enc_s[:, 1:]

        # Flatten temporal dimensions for batch processing
        B, T, embed_dim = preds.shape
        Z = preds.reshape(B * T, embed_dim)  # Predicted embeddings
        Z_prime = enc_s.reshape(B * T, embed_dim)  # Target embeddings

        # --- Invariance Term ---
        invariance_loss = torch.mean(
            torch.sum((Z - Z_prime) ** 2, dim=1)
        )  # Mean squared Euclidean distance

        # --- Variance Term ---
        # Compute standard deviation along the batch dimension
        std_Z = torch.sqrt(Z.var(dim=0, unbiased=False) + epsilon)
        std_Z_prime = torch.sqrt(Z_prime.var(dim=0, unbiased=False) + epsilon)

        variance_loss = torch.mean(F.relu(gamma - std_Z)) + torch.mean(
            F.relu(gamma - std_Z_prime)
        )

        # --- Covariance Term ---
        # Center the embeddings
        Z_centered = Z - Z.mean(dim=0, keepdim=True)
        Z_prime_centered = Z_prime - Z_prime.mean(dim=0, keepdim=True)

        # Compute covariance matrices
        cov_Z = (Z_centered.T @ Z_centered) / (B * T - 1)
        cov_Z_prime = (Z_prime_centered.T @ Z_prime_centered) / (B * T - 1)

        # Sum of squared off-diagonal elements
        cov_loss_Z = torch.sum(cov_Z**2) - torch.sum(torch.diag(cov_Z) ** 2)
        cov_loss_Z_prime = torch.sum(cov_Z_prime**2) - torch.sum(
            torch.diag(cov_Z_prime) ** 2
        )

        covariance_loss = cov_loss_Z + cov_loss_Z_prime

        # --- Total VICReg Loss ---
        vicreg_loss = (
            lambda_invariance * invariance_loss
            + mu_variance * variance_loss
            + nu_covariance * covariance_loss
        )

        return vicreg_loss, invariance_loss, variance_loss, covariance_loss

    def training_step(self, batch, device):
        states, actions = batch.states.to(device, non_blocking=True), batch.actions.to(
            device, non_blocking=True
        )
        preds, enc_s = self.forward(states, actions)

        # loss = self.compute_mse_loss(preds, enc_s) # Normal Loss Calculation

        # Compute regularization loss
        states_embed = enc_s[:, :-1, :].reshape(
            -1, self.config.embed_dim
        )  # Input to predictor
        pred_states = preds[:, 1:, :].reshape(
            -1, self.config.embed_dim
        )  # Output of predictor
        action_reg_loss = self.compute_regularization_loss(
            states_embed, pred_states, actions
        )

        vic_reg_loss, invariance_loss, variance_loss, covariance_loss = (
            self.compute_vicreg_loss(preds, enc_s)
        )  # VICReg Loss Calculation

        # Combine losses
        total_loss = vic_reg_loss + self.config.lambda_reg * action_reg_loss

        self.optimizer.zero_grad()
        total_loss.backward()

        # Compute grad_norm without clipping
        grad_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                grad_norm += param_norm.item() ** 2
        grad_norm = grad_norm**0.5

        self.optimizer.step()
        self.scheduler.step()  # Step the scheduler

        learning_rate = self.optimizer.param_groups[0]["lr"]

        # Compute the absolute value of the action weights
        action_weight = (
            self.pred.action_proj.weight
        )  # Get weights from the action projection layer
        action_weight_abs = (
            action_weight.abs().mean().item()
        )  # Compute the mean absolute value

        # Compute deviation from identity for fc layer
        fc_weight = self.pred.fc[
            1
        ].weight  # Get the weights of the Linear layer inside fc
        identity_matrix = torch.eye(fc_weight.size(0), fc_weight.size(1)).to(
            fc_weight.device
        )
        deviation_from_identity = torch.norm(
            fc_weight - identity_matrix, p="fro"
        ) / torch.norm(identity_matrix, p="fro")

        # Prepare the output dictionary
        output = {
            "loss": total_loss.item(),
            "grad_norm": grad_norm,
            "learning_rate": learning_rate,
            "action_weight_abs": action_weight_abs,
            "deviation_from_identity_pred_final_proj": deviation_from_identity.item(),  # Log the deviation
            "action_reg_loss": action_reg_loss.item(),
            # 'mse_loss': loss.item(),
            "invariance_loss": invariance_loss,
            "variance_loss": variance_loss,
            "covariance_loss": covariance_loss,
            # Add more loggable values here if needed
        }

        # Non-loggable data
        output["non_logs"] = {
            "states": states.detach(),
            "actions": actions.detach(),
            "enc_embeddings": enc_s.detach(),
            "pred_embeddings": preds.detach(),
        }

        return output

    def validation_step(self, batch):
        states, actions = batch.states, batch.actions
        preds, enc_s = self.forward(states, actions)
        loss = self.compute_mse_loss(preds, enc_s)
        learning_rate = self.optimizer.param_groups[0]["lr"]

        # Compute the absolute value of the action weights
        action_weight = (
            self.pred.action_proj.weight
        )  # Get weights from the action projection layer
        action_weight_abs = (
            action_weight.abs().mean().item()
        )  # Compute the mean absolute value

        # Compute deviation from identity for fc layer
        fc_weight = self.pred.fc[
            1
        ].weight  # Get the weights of the Linear layer inside fc
        identity_matrix = torch.eye(fc_weight.size(0), fc_weight.size(1)).to(
            fc_weight.device
        )
        deviation_from_identity = torch.norm(
            fc_weight - identity_matrix, p="fro"
        ) / torch.norm(identity_matrix, p="fro")

        # Prepare the output dictionary
        output = {
            "loss": loss.item(),
            "learning_rate": learning_rate,
            "action_weight_abs": action_weight_abs,
            "deviation_from_identity_pred_final_proj": deviation_from_identity.item(),  # Log the deviation
        }

        # Non-loggable data
        output["non_logs"] = {
            "states": states.detach(),
            "actions": actions.detach(),
            "enc_embeddings": enc_s.detach(),
            "pred_embeddings": preds.detach(),
        }

        return output


class Encoder2D(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.repr_dim = config.embed_dim * config.out_c

        _input_size = 65
        self.output_side = int(
            math.sqrt(self.repr_dim / config.out_c)
        )  # Calculate the side of the 2D embedding
        # Determine the number of convolutional blocks required
        self.num_conv_blocks = int(math.log2(_input_size / self.output_side))
        if 2**self.num_conv_blocks * self.output_side != 2 ** int(
            math.log2(_input_size)
        ):
            raise ValueError(
                "Cannot evenly reduce input_size to output_side using stride-2 convolutions."
            )

        layers = []
        in_channels = 2  # Input has 2 channels (agent and wall)
        out_channels = 16  # Start with 32 output channels
        for i in range(self.num_conv_blocks):
            layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=0,
                )
                if i == 0
                else nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
            )  # Halve the spatial dimensions
            layers.append(nn.ReLU())
            in_channels = out_channels
            out_channels = min(out_channels * 2, 256)  # Cap channels at 256

        # Final convolution to reduce to single-channel output
        layers.append(
            nn.Conv2d(in_channels, 1, kernel_size=1)
        )  # Single-channel embedding

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        # Input: (B, 2, 65, 65)
        x = self.conv(x)  # Dynamically reduce to (B, 1, output_side, output_side)
        return x  # Output shape: (B, 1, output_side, output_side)


class Predictor2D(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.repr_dim = config.embed_dim * config.out_c
        self.output_side = int(
            math.sqrt(self.repr_dim / config.out_c)
        )  # Calculate 2D embedding dimensions

        self.action_proj = nn.Linear(
            self.config.action_dim, self.output_side**2
        )  # Project action to repr_dim
        # Optionally, you can add a projection for the state embedding
        # self.state_proj = nn.Linear(self.repr_dim, self.repr_dim)
        # For simplicity, we'll assume identity for state embedding

        self.conv = nn.Sequential(
            nn.Conv2d(
                self.config.out_c + 1,
                self.config.out_c,
                kernel_size=3,
                stride=1,
                padding=1,
            ),  # Combine state and action
            nn.ReLU(),
            nn.Conv2d(
                self.config.out_c,
                self.config.out_c,
                kernel_size=3,
                stride=1,
                padding=1,
            ),  # Reduce to single-channel
            # nn.ReLU(),
            # nn.Conv2d(
            #     self.config.out_c * 2,
            #     self.config.out_c,
            #     kernel_size=3,
            #     stride=1,
            #     padding=1,
            # ),  # Reduce to single-channel
        )

    def forward(self, s_embed, a):
        # s_embed: (B, 1, output_side, output_side)
        # a: (B, action_dim)
        B, _, H, W = s_embed.shape

        # Project actions to match the 2D embedding
        a_proj = self.action_proj(a).view(B, 1, H, W)  # Action embedding: (B, 1, H, W)
        x = torch.cat(
            [s_embed, a_proj], dim=1
        )  # Combine state and action: (B, 2, H, W)
        x = self.conv(x)  # Process combined input: (B, 1, H, W)
        return x  # Predicted embedding: (B, 1, H, W)


class ActionRegularizer2D(nn.Module):
    def __init__(self, config, embed_dim, action_dim):
        super().__init__()
        self.output_side = int(
            math.sqrt(embed_dim)
        )  # Calculate the side of the 2D embedding
        self.action_reg_net = nn.Sequential(
            nn.Conv2d(config.out_c, 16, kernel_size=3, padding=1),  # 2D conv layer
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),  # Output single channel
            nn.Flatten(),  # Flatten to prepare for linear mapping
            nn.Linear(
                self.output_side * self.output_side, action_dim
            ),  # Map to action_dim
        )

    def forward(self, states_embed, pred_states):
        """
        Args:
            states_embed: Tensor of shape (B*(T-1), 1, output_side, output_side) - previous state embeddings
            pred_states: Tensor of shape (B*(T-1), 1, output_side, output_side) - predicted state embeddings

        Returns:
            predicted_actions: Tensor of shape (B*(T-1), action_dim) - predicted actions
        """
        # Calculate embedding differences
        embedding_diff = (
            pred_states - states_embed
        )  # (B*(T-1), 1, output_side, output_side)

        # Predict actions from embedding differences
        predicted_actions = self.action_reg_net(embedding_diff)  # (B*(T-1), action_dim)
        return predicted_actions


class ActionRegularizationJEPA2D(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.enc = Encoder2D(config)
        self.pred = Predictor2D(config)
        self.action_reg_net = ActionRegularizer2D(
            config, config.embed_dim, config.action_dim
        )  # Small network for action prediction from embedding differences

        self.optimizer = get_optimizer(config, self.parameters())
        self.scheduler = get_scheduler(self.optimizer, config)

        self.config = config
        self.repr_dim = config.embed_dim * config.out_c

    def forward(self, states, actions, teacher_forcing=True):
        B, _, C, H, W = states.shape  # states: (B, T, C, H, W)
        T = actions.shape[1] + 1  # Number of timesteps | actions: (B, T-1, action_dim)

        if teacher_forcing:
            states = states.view(B * T, C, H, W)  # Reshape to (B*T, C, H, W)

            enc_states = self.enc(states)  # (B*T, 1, H', W')
            _, _, H_out, W_out = enc_states.shape
            enc_states = enc_states.view(B, T, 1, H_out, W_out)  # (B, T, 1, H', W')
            preds = torch.zeros_like(enc_states)  # preds: (B, T, 1, H', W')
            preds[:, 0, :, :, :] = enc_states[
                :, 0, :, :, :
            ]  # Initialize first timestep

            # Prepare inputs for the predictor
            states_embed = enc_states[:, :-1, :, :, :]  # (B, T-1, 1, H', W')
            states_embed = states_embed.contiguous().view(
                -1, 1, H_out, W_out
            )  # (B*(T-1), 1, H', W')
            actions = actions.view(-1, self.config.action_dim)  # (B*(T-1), action_dim)

            pred_states = self.pred(states_embed, actions)  # (B*(T-1), 1, H', W')
            pred_states = pred_states.view(
                B, T - 1, 1, H_out, W_out
            )  # (B, T-1, 1, H', W')
            preds[:, 1:, :, :, :] = pred_states  # Assign predictions to preds

            return (
                preds,
                enc_states,
            )  # preds: (B, T, 1, H', W'), enc_states: (B, T, 1, H', W')

        else:
            states_0 = states[:, 0, :, :, :]  # (B, C, H, W)
            enc_state = self.enc(states_0)  # (B, 1, H', W')
            preds = [enc_state]  # List to store predictions

            for t in range(1, T):
                action_t_minus1 = actions[:, t - 1, :]  # (B, action_dim)
                state_embed_t_minus1 = preds[
                    -1
                ]  # Use the last predicted embedding (B, 1, H', W')
                pred_state = self.pred(
                    state_embed_t_minus1, action_t_minus1
                )  # (B, 1, H', W')
                preds.append(pred_state)

            # Stack predictions and true encodings along the time dimension
            preds = torch.stack(preds, dim=1)  # (B, T, 1, H', W')
            preds = preds.view(B, T, -1)  # (B, T, H'*W')
            return preds

    def compute_mse_loss(self, preds, enc_s):
        # preds, enc_s: (B, T, 1, H', W')
        loss = F.mse_loss(
            preds[:, 1:], enc_s[:, 1:]
        )  # Compute MSE loss for timesteps 1 to T-1
        return loss

    def compute_regularization_loss(self, states_embed, pred_states, actions):
        """
        Computes the regularization loss based on the embedding difference and actions.
        """
        # states_embed and pred_states: (B*(T-1), 1, H', W')
        # actions: (B*(T-1), action_dim)
        # Predict actions from embedding differences
        predicted_actions = self.action_reg_net(
            states_embed, pred_states
        )  # (B*(T-1), action_dim)

        # Compute MSE loss between predicted and actual actions
        reg_loss = F.mse_loss(predicted_actions, actions)
        return reg_loss

    def compute_vicreg_loss(self, preds, enc_s, gamma=1.0, epsilon=1e-4):
        """
        Compute VICReg loss with invariance, variance, and covariance terms.

        Args:
            preds: Predicted embeddings from the predictor. Shape (B, T, 1, H', W').
            enc_s: Target embeddings from the encoder. Shape (B, T, 1, H', W').
            gamma: Target standard deviation for variance term.
            epsilon: Small value to avoid numerical instability.

        Returns:
            vicreg_loss: The combined VICReg loss.
        """

        # taking from config
        lambda_invariance = self.config.vicreg_loss.lambda_invariance
        mu_variance = self.config.vicreg_loss.mu_variance
        nu_covariance = self.config.vicreg_loss.nu_covariance

        preds, enc_s = preds[:, 1:], enc_s[:, 1:]  # Drop the first timestep

        # Flatten spatial and temporal dimensions for batch processing
        B, T, C, H, W = preds.shape
        embed_dim = C * H * W
        Z = preds.reshape(B * T, embed_dim)  # Predicted embeddings
        Z_prime = enc_s.reshape(B * T, embed_dim)  # Target embeddings

        # --- Invariance Term ---
        invariance_loss = torch.mean(
            torch.sum((Z - Z_prime) ** 2, dim=1)
        )  # Mean squared Euclidean distance

        # --- Variance Term ---
        # Compute standard deviation along the batch dimension
        std_Z = torch.sqrt(Z.var(dim=0, unbiased=False) + epsilon)
        std_Z_prime = torch.sqrt(Z_prime.var(dim=0, unbiased=False) + epsilon)

        variance_loss = torch.mean(F.relu(gamma - std_Z)) + torch.mean(
            F.relu(gamma - std_Z_prime)
        )

        # --- Covariance Term ---
        # Center the embeddings
        Z_centered = Z - Z.mean(dim=0, keepdim=True)
        Z_prime_centered = Z_prime - Z_prime.mean(dim=0, keepdim=True)

        # Compute covariance matrices
        cov_Z = (Z_centered.T @ Z_centered) / (B * T - 1)
        cov_Z_prime = (Z_prime_centered.T @ Z_prime_centered) / (B * T - 1)

        # Sum of squared off-diagonal elements
        cov_loss_Z = torch.sum(cov_Z**2) - torch.sum(torch.diag(cov_Z) ** 2)
        cov_loss_Z_prime = torch.sum(cov_Z_prime**2) - torch.sum(
            torch.diag(cov_Z_prime) ** 2
        )

        covariance_loss = cov_loss_Z + cov_loss_Z_prime

        # --- Total VICReg Loss ---
        vicreg_loss = (
            lambda_invariance * invariance_loss
            + mu_variance * variance_loss
            + nu_covariance * covariance_loss
        )

        return vicreg_loss, invariance_loss, variance_loss, covariance_loss

    def training_step(self, batch, device):
        states, actions = batch.states.to(device, non_blocking=True), batch.actions.to(
            device, non_blocking=True
        )
        preds, enc_s = self.forward(states, actions)  # preds, enc_s: (B, T, 1, H, W)

        # Compute regularization loss
        B, T, _, H, W = enc_s.shape  # (B, T, 1, H, W)
        states_embed = enc_s[:, :-1, :, :, :].reshape(
            -1, 1, H, W
        )  # Input to predictor: (B*(T-1), 1, H, W)
        pred_states = preds[:, 1:, :, :, :].reshape(
            -1, 1, H, W
        )  # Output of predictor: (B*(T-1), 1, H, W)
        actions = actions.reshape(
            -1, self.config.action_dim
        )  # Actions: (B*(T-1), action_dim)

        action_reg_loss = self.compute_regularization_loss(
            states_embed, pred_states, actions
        )  # Uses ActionRegularizer2D internally

        # Compute VICReg Loss
        vic_reg_loss, invariance_loss, variance_loss, covariance_loss = (
            self.compute_vicreg_loss(preds, enc_s)
        )  # VICReg Loss Calculation

        # Combine losses
        total_loss = vic_reg_loss + self.config.lambda_reg * action_reg_loss

        self.optimizer.zero_grad()
        total_loss.backward()

        # Compute grad_norm without clipping
        grad_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                grad_norm += param_norm.item() ** 2
        grad_norm = grad_norm**0.5

        self.optimizer.step()
        self.scheduler.step()  # Step the scheduler

        learning_rate = self.optimizer.param_groups[0]["lr"]

        # Compute the absolute value of the action weights
        action_weight = self.pred.action_proj.weight  # Project action weights
        action_weight_abs = action_weight.abs().mean().item()

        # Compute deviation from identity for fc layer
        # fc_weight = self.pred.fc[1].weight  # Get the weights of the Linear layer inside fc
        # identity_matrix = torch.eye(fc_weight.size(0), fc_weight.size(1)).to(
        #     fc_weight.device
        # )
        # deviation_from_identity = torch.norm(
        #     fc_weight - identity_matrix, p="fro"
        # ) / torch.norm(identity_matrix, p="fro")

        # Prepare the output dictionary
        output = {
            "loss": total_loss.item(),
            "grad_norm": grad_norm,
            "learning_rate": learning_rate,
            "action_weight_abs": action_weight_abs,
            # "deviation_from_identity_pred_final_proj": deviation_from_identity.item(),
            "action_reg_loss": action_reg_loss.item(),
            "invariance_loss": invariance_loss,
            "variance_loss": variance_loss,
            "covariance_loss": covariance_loss,
        }

        # Non-loggable data
        output["non_logs"] = {
            "states": states.detach(),
            "actions": actions.detach(),
            "enc_embeddings": enc_s.detach(),
            "pred_embeddings": preds.detach(),
        }

        return output

    def validation_step(self, batch):
        states, actions = batch.states, batch.actions
        preds, enc_s = self.forward(states, actions)  # preds, enc_s: (B, T, 1, H, W)

        # Compute MSE Loss
        loss = self.compute_mse_loss(preds, enc_s)  # preds, enc_s: (B, T, 1, H, W)

        # Learning rate
        learning_rate = self.optimizer.param_groups[0]["lr"]

        # Compute the absolute value of the action weights
        action_weight = self.pred.action_proj.weight  # Project action weights
        action_weight_abs = action_weight.abs().mean().item()

        # Compute deviation from identity for fc layer
        # fc_weight = self.pred.fc[1].weight  # Get the weights of the Linear layer inside fc
        # identity_matrix = torch.eye(fc_weight.size(0), fc_weight.size(1)).to(
        #     fc_weight.device
        # )
        # deviation_from_identity = torch.norm(
        #     fc_weight - identity_matrix, p="fro"
        # ) / torch.norm(identity_matrix, p="fro")

        # Prepare the output dictionary
        output = {
            "loss": loss.item(),
            "learning_rate": learning_rate,
            "action_weight_abs": action_weight_abs,
            # "deviation_from_identity_pred_final_proj": deviation_from_identity.item(),  # Log the deviation
        }

        # Non-loggable data
        output["non_logs"] = {
            "states": states.detach(),
            "actions": actions.detach(),
            "enc_embeddings": enc_s.detach(),
            "pred_embeddings": preds.detach(),
        }

        return output


class FlexibleEncoder2D(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        feature_index = config.feature_index  # e.g., 1 for self.backbone(x)[1]
        self.backbone, input_ch = create_minimal_feature_model(config, feature_index)
        self.channel_adjust = nn.Conv2d(input_ch, config.out_c, kernel_size=1)

    def forward(self, x):
        # Reshape input to merge batch and trajectory dimensions
        original_shape = x.shape
        x = x.view(-1, *original_shape[-3:])  # Reshape to [batch*trajectory, channels, height, width]
        features = self.backbone(x)  # Extract features
        features = self.channel_adjust(features)  # Adjust channels
        return features


class ActionRegularizationJEPA2DFlexibleEncoder(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.enc = FlexibleEncoder2D(config)
        self.pred = Predictor2D(config)
        self.action_reg_net = ActionRegularizer2D(
            config, config.embed_dim, config.action_dim
        )  # Small network for action prediction from embedding differences

        self.optimizer = get_optimizer(config, self.parameters())
        self.scheduler = get_scheduler(self.optimizer, config)

        self.config = config
        self.repr_dim = config.embed_dim * config.out_c

    def forward(self, states, actions, teacher_forcing=True, return_enc=False, pred_flattened=True):
        B, _, C, H, W = states.shape  # states: (B, T, C, H, W)
        T = actions.shape[1] + 1  # Number of timesteps | actions: (B, T-1, action_dim)

        if teacher_forcing:
            states = states.view(B * T, C, H, W)  # Reshape to (B*T, C, H, W)

            enc_states = self.enc(states)  # (B*T, C', H', W')
            _, C_out, H_out, W_out = enc_states.shape
            enc_states = enc_states.view(
                B, T, C_out, H_out, W_out
            )  # (B, T, C', H', W')
            preds = torch.zeros_like(enc_states)  # preds: (B, T, C', H', W')
            preds[:, 0, :, :, :] = enc_states[
                :, 0, :, :, :
            ]  # Initialize first timestep

            # Prepare inputs for the predictor
            states_embed = enc_states[:, :-1, :, :, :]  # (B, T-1, C', H', W')
            states_embed = states_embed.contiguous().view(
                -1, C_out, H_out, W_out
            )  # (B*(T-1), C', H', W')
            actions = actions.view(-1, self.config.action_dim)  # (B*(T-1), action_dim)

            pred_states = self.pred(states_embed, actions)  # (B*(T-1), C', H', W')
            pred_states = pred_states.view(
                B, T - 1, C_out, H_out, W_out
            )  # (B, T-1, C', H', W')
            preds[:, 1:, :, :, :] = pred_states  # Assign predictions to preds

            if return_enc:
                return preds, enc_states # preds: (B, T, C', H', W'), enc_states: (B, T, C', H', W')
            return preds

        else:
            states_0 = states[:, 0, :, :, :]  # (B, C, H, W)
            enc_state = self.enc(states_0)  # (B, C', H', W')
            preds = [enc_state]  # List to store predictions

            for t in range(1, T):
                action_t_minus1 = actions[:, t - 1, :]  # (B, action_dim)
                state_embed_t_minus1 = preds[
                    -1
                ]  # Use the last predicted embedding (B, C', H', W')
                pred_state = self.pred(
                    state_embed_t_minus1, action_t_minus1
                )  # (B, C', H', W')
                preds.append(pred_state)

            # Stack predictions and true encodings along the time dimension
            preds = torch.stack(preds, dim=1)  # (B, T, C', H', W')
            if pred_flattened:
                preds = preds.view(B, T, -1)  # (B, T, C'*H'*W')

            if return_enc:
                states = states.view(B * T, C, H, W)
                enc_states = self.enc(states)  # (B*T, C', H', W')
                _, C_out, H_out, W_out = enc_states.shape
                enc_states = enc_states.view(
                    B, T, C_out, H_out, W_out
                )  # (B, T, C', H', W')
                return preds, enc_states
            return preds

    def compute_mse_loss(self, preds, enc_s):
        # preds, enc_s: (B, T, 1, H', W')
        loss = F.mse_loss(
            preds[:, 1:], enc_s[:, 1:]
        )  # Compute MSE loss for timesteps 1 to T-1
        return loss

    def compute_regularization_loss(self, states_embed, pred_states, actions):
        """
        Computes the regularization loss based on the embedding difference and actions.
        """
        # states_embed and pred_states: (B*(T-1), 1, H', W')
        # actions: (B*(T-1), action_dim)
        # Predict actions from embedding differences
        predicted_actions = self.action_reg_net(
            states_embed, pred_states
        )  # (B*(T-1), action_dim)

        # Compute MSE loss between predicted and actual actions
        reg_loss = F.mse_loss(predicted_actions, actions)
        return reg_loss

    def training_step(self, batch, device):
        states, actions = batch.states.to(device, non_blocking=True), batch.actions.to(
            device, non_blocking=True
        )
        preds, enc_s = self.forward(
            states, 
            actions, 
            teacher_forcing=self.config.teacher_forcing, 
            return_enc=self.config.return_enc, 
            pred_flattened=self.config.pred_flattened
        )  # preds, enc_s: (B, T, 1, H, W)

        # Compute regularization loss
        B, T, _, H, W = enc_s.shape  # (B, T, 1, H, W)
        states_embed = enc_s[:, :-1, :, :, :]  # Input to predictor: (B, (T-1), 1, H, W)
        pred_states = preds[:, 1:, :, :, :]  # Output of predictor: (B, (T-1), 1, H, W)
        actions = actions.reshape(
            -1, self.config.action_dim
        )  # Actions: (B*(T-1), action_dim)

        action_reg_loss = self.compute_regularization_loss(
            states_embed.reshape(
            -1, self.config.out_c, H, W
        ), pred_states.reshape(
            -1, self.config.out_c, H, W
        ), actions
        )  # Uses ActionRegularizer2D internally

        # Compute VICReg Loss
        vic_reg_loss, invariance_loss, variance_loss, covariance_loss = vicreg_loss(
                pred_states.flatten(start_dim=0, end_dim=1).flatten(start_dim=1, end_dim=-1), # (B * T, C * H * W)
                states_embed.flatten(start_dim=0, end_dim=1).flatten(start_dim=1, end_dim=-1), # (B * T, C * H * W)
            )

        # Compute VICReg Loss
        vic_reg_loss_s, invariance_loss_s, variance_loss_s, covariance_loss_s = vicreg_loss(
            pred_states.flatten(start_dim=0, end_dim=1) # (B * T, H, W, C)
            .permute(0, 2, 3, 1) # (B * T, H, W, C)
            .flatten(end_dim=-2), # (B * T * H * W, C)
            states_embed.flatten(start_dim=0, end_dim=1)
            .permute(0, 2, 3, 1) # (B * T, H, W, C)
            .flatten(end_dim=-2), # (B * T * H * W, C)
        )

        # Combine losses
        total_loss = (
            (vic_reg_loss + vic_reg_loss_s)
        ) + self.config.lambda_reg * action_reg_loss

        self.optimizer.zero_grad()
        total_loss.backward()

        # Compute grad_norm without clipping
        grad_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                grad_norm += param_norm.item() ** 2
        grad_norm = grad_norm**0.5

        self.optimizer.step()
        self.scheduler.step()  # Step the scheduler

        learning_rate = self.optimizer.param_groups[0]["lr"]

        # Compute the absolute value of the action weights
        action_weight = self.pred.action_proj.weight  # Project action weights
        action_weight_abs = action_weight.abs().mean().item()

        # Compute deviation from identity for fc layer
        # fc_weight = self.pred.fc[1].weight  # Get the weights of the Linear layer inside fc
        # identity_matrix = torch.eye(fc_weight.size(0), fc_weight.size(1)).to(
        #     fc_weight.device
        # )
        # deviation_from_identity = torch.norm(
        #     fc_weight - identity_matrix, p="fro"
        # ) / torch.norm(identity_matrix, p="fro")

        # Prepare the output dictionary
        output = {
            "loss": total_loss.item(),
            "grad_norm": grad_norm,
            "learning_rate": learning_rate,
            "action_weight_abs": action_weight_abs,
            # "deviation_from_identity_pred_final_proj": deviation_from_identity.item(),
            "action_reg_loss": action_reg_loss.item(),
            "invariance_loss": invariance_loss,
            "variance_loss": variance_loss,
            "covariance_loss": covariance_loss,
            "invariance_loss_s": invariance_loss_s,
            "variance_loss_s": variance_loss_s,
            "covariance_loss_s": covariance_loss_s,
        }

        # Non-loggable data
        output["non_logs"] = {
            "states": states.detach(),
            "actions": actions.detach(),
            "enc_embeddings": enc_s.detach(),
            "pred_embeddings": preds.detach(),
        }

        return output

    def validation_step(self, batch):
        states, actions = batch.states, batch.actions
        preds, enc_s = self.forward(
            states, 
            actions, 
            teacher_forcing=self.config.teacher_forcing, 
            return_enc=self.config.return_enc, 
            pred_flattened=self.config.pred_flattened
        )  # preds, enc_s: (B, T, 1, H, W)

        # Compute MSE Loss
        loss = self.compute_mse_loss(preds, enc_s)  # preds, enc_s: (B, T, 1, H, W)

        # Learning rate
        learning_rate = self.optimizer.param_groups[0]["lr"]

        # Compute the absolute value of the action weights
        action_weight = self.pred.action_proj.weight  # Project action weights
        action_weight_abs = action_weight.abs().mean().item()

        # Compute deviation from identity for fc layer
        # fc_weight = self.pred.fc[1].weight  # Get the weights of the Linear layer inside fc
        # identity_matrix = torch.eye(fc_weight.size(0), fc_weight.size(1)).to(
        #     fc_weight.device
        # )
        # deviation_from_identity = torch.norm(
        #     fc_weight - identity_matrix, p="fro"
        # ) / torch.norm(identity_matrix, p="fro")

        # Prepare the output dictionary
        output = {
            "loss": loss.item(),
            "learning_rate": learning_rate,
            "action_weight_abs": action_weight_abs,
            # "deviation_from_identity_pred_final_proj": deviation_from_identity.item(),  # Log the deviation
        }

        # Non-loggable data
        output["non_logs"] = {
            "states": states.detach(),
            "actions": actions.detach(),
            "enc_embeddings": enc_s.detach(),
            "pred_embeddings": preds.detach(),
        }

        return output


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model


# Explanation:
# This class replaces the original `Encoder` class.
# Instead of the downsampling convolutional approach shown in the sample,
# we directly use the resnet18 with features_only=True as done originally.
# The shape handling remains the same: input is (B,T,2,H,W),
# we flatten to (B*T,2,H,W), run through ResNet, take the feature map at index [1],
# and then reshape back to (B,T,C,H',W').
class Encoder2Dv2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Use the same backbone as original code
        self.resnet = create_model(
            "resnet18.a1_in1k",
            pretrained=False,
            num_classes=0,
            in_chans=2,
            features_only=True,
        )

    def forward(self, x):
        # x: (B, T, 2, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)  # Flatten time into batch
        features = self.resnet(x)[1]  # As in original code, take index [1]
        # Suppose features: (B*T, C', H', W')
        C_out = features.shape[1]
        H_out = features.shape[2]
        W_out = features.shape[3]

        features = features.view(B, T, C_out, H_out, W_out)
        return features  # (B, T, C_out, H_out, W_out)


# Explanation:
# This class replaces the original `Predictor` class.
# We keep the exact predictor architecture and logic from above.
# The original code sets `Predictor(input_dim=66, output_dim=64)` after obtaining features from the encoder.
# Input: concatenation of state embedding (64 channels from resnet) + action (2 channels) = 66 channels.
# Output: reduce to 64 channels.
class Predictor2Dv2(nn.Module):
    def __init__(self, input_dim=66, output_dim=64):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Conv2d(input_dim, input_dim - 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(input_dim - 2, output_dim, kernel_size=3, padding=1),
        )

    def forward(self, encoded_o_t, action):
        # encoded_o_t: (B, T, C, H', W')
        # action: (B, T-1, 2)
        B, T, C, H, W = encoded_o_t.shape

        # Reshape action
        action = action.view(B, T - 1, 2, 1, 1)
        action = action.repeat(1, 1, 1, H, W)  # (B, T-1, 2, H, W)

        predictions = []
        for t in range(T - 1):
            # Concatenate current encoded state with action along channel dim
            x = torch.cat([encoded_o_t[:, t], action[:, t]], dim=1)  # (B, C+2, H, W)
            pred = self.predictor(x)  # (B, output_dim, H, W)
            predictions.append(pred)

        # Stack predictions along time
        return torch.stack(predictions, dim=1)  # (B, T-1, output_dim, H, W)


# Explanation:
# The original code does not do action regression from embedding differences.
# The provided template has an `ActionRegularizer2D` class and
# mentions predicting actions from state differences, but the original code does not have this.
# To maintain the requested format, we'll define this class but not use it.
class ActionRegularizer2Dv2(nn.Module):
    def __init__(self, config, embed_dim, action_dim):
        super().__init__()
        # Not used, as original code does not do action regression
        pass

    def forward(self, states_embed, pred_states):
        # Return zero since original code does not do this step
        return torch.zeros(states_embed.size(0), 2, device=states_embed.device)


# Explanation:
# We now combine everything into the `ActionRegularizationJEPA2D` class.
# Instead of the downsampling logic in the provided code snippet, we directly use
# our `Encoder2D` (which uses ResNet) and `Predictor2D` (our ConvNet) from above.
# We incorporate the VICReg computations exactly as in the original code:
#   inv-loss (MSE), var-loss (thresholded standard deviation), cov-loss (off-diagonal covariance).
# We'll run the forward pass, predict next states, flatten as in the original training loop,
# compute the VICReg losses at spatial and global scales, and return the metrics.
#
# The original code:
# - Encodes states
# - Predicts next states
# - Computes VICReg loss (inv, var, cov) both globally and spatially
# - Combines them, and backprops.
#
# We'll do the same here in `training_step`.
#
# Note: We assume `BaseModel`, `get_optimizer`, `get_scheduler`, and `config` are defined elsewhere.
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR


class ActionRegularizationJEPA2Dv2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.enc = Encoder2Dv2(config)
        self.pred = Predictor2Dv2(input_dim=66, output_dim=64)  # Same as original code
        # self.action_reg_net = ActionRegularizer2D(config, config.embed_dim, config.action_dim)  # Not used

        # Create optimizer and scheduler as per original code
        params = list(self.enc.parameters()) + list(self.pred.parameters())
        self.optimizer = AdamW(params, lr=1e-3)
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=1e-2,
            steps_per_epoch=config.steps_per_epoch,
            epochs=config.epochs,
        )

        # VICReg parameters from original code
        self.inv_coeff = 25.0
        self.var_coeff = 15.0
        self.cov_coeff = 1.0
        self.gamma = 1.0

    def forward(self, states, actions, teacher_forcing=True):
        # states: (B, T, 2, H, W)
        # actions: (B, T-1, 2)
        B, T, C, H, W = states.shape
        # C should be 2 as per encoder input

        if teacher_forcing:
            # Pass states directly to encoder; it handles flattening internally.
            enc_states = self.enc(states)  # (B, T, C_out, H_out, W_out)
            # Initialize predictions (same shape as enc_states)
            preds = torch.zeros_like(enc_states)  # (B, T, C_out, H_out, W_out)
            preds[:, 0] = enc_states[
                :, 0
            ]  # First predicted state is the first encoded state

            # Use predictor to predict next states
            # The predictor expects:
            #   enc_states: (B, T, C_out, H_out, W_out)
            #   actions: (B, T-1, 2)
            # returns: (B, T-1, output_dim, H_out, W_out)
            pred_states = self.pred(enc_states, actions)

            # Assign predicted states to preds at t=1 to T-1
            preds[:, 1:] = pred_states

            return preds, enc_states

        else:
            # Non-teacher forcing scenario:
            # Take the first state only
            states_0 = states[:, 0]  # (B, 2, H, W)
            # Encoder expects (B, T, 2, H, W), so add a time dimension for a single step
            enc_state = self.enc(states_0.unsqueeze(1))  # (B, 1, C_out, H_out, W_out)

            preds = [enc_state]

            # Predict step by step for remaining timesteps
            for t in range(1, T):
                action_t_minus1 = actions[:, t - 1]  # (B, 2)
                state_embed_t_minus1 = preds[-1]  # (B, 1, C_out, H_out, W_out)

                # The predictor expects a sequence along T, but we have only one timestep to predict.
                # We'll just pass a single-step sequence (T=1):
                # actions also need a time dimension: (B,1,2)
                pred_state = self.pred(
                    state_embed_t_minus1, action_t_minus1.unsqueeze(1)
                )
                # pred_state: (B, 1, C_out, H_out, W_out)

                preds.append(pred_state)

            # Stack predictions: (B, T, C_out, H_out, W_out)
            preds = torch.cat(preds, dim=1)
            preds = preds.view(B, T, -1)  # (B, T, C_out*H_out*W_out)
            return preds

    def representation_loss(self, x, y):
        # As original VICReg: MSE loss
        return F.mse_loss(x, y)

    def variance_loss(self, x, gamma):
        x = x - x.mean(dim=0)
        std = x.std(dim=0)
        var_loss = F.relu(gamma - std).mean()
        return var_loss

    def covariance_loss(self, x):
        x = x - x.mean(dim=0)
        cov = (x.T @ x) / (x.shape[0] - 1)
        cov.fill_diagonal_(0.0)
        cov_loss = cov.pow(2).sum() / x.shape[1]
        return cov_loss

    def compute_vicreg_losses(self, predicted_next, target_encoded):
        # Compute VICReg losses as in the original code
        # predicted_next, target_encoded: after flattening appropriately

        # invariance
        inv_loss = self.inv_coeff * self.representation_loss(
            predicted_next, target_encoded
        )

        # variance
        var_loss_x = self.var_coeff * self.variance_loss(predicted_next, self.gamma)
        var_loss_y = self.var_coeff * self.variance_loss(target_encoded, self.gamma)
        var_loss = (var_loss_x + var_loss_y) / 2

        # covariance
        cov_loss_x = self.cov_coeff * self.covariance_loss(predicted_next)
        cov_loss_y = self.cov_coeff * self.covariance_loss(target_encoded)
        cov_loss = (cov_loss_x + cov_loss_y) / 2

        loss = inv_loss + var_loss + cov_loss
        return {
            "loss": loss,
            "inv-loss": inv_loss,
            "var-loss": var_loss,
            "cov-loss": cov_loss,
        }

    def training_step(self, batch, device):
        states, _, actions = batch
        states = states.to(device)
        actions = actions.to(device)

        predicted_next, enc_states = self(states, actions)

        # target_encoded = enc_states[:, 1:] (B, T-1, C, H', W')
        target_encoded = enc_states[:, 1:]

        # predicted_next: (B, T-1, C, H', W')
        # For VICReg, we compute loss globally and also add a spatially-resolved term as in original code.

        # Flatten for global VICReg:
        B, Tm1, C, H, W = predicted_next.shape
        # Flatten channels + spatial:
        pred_global = predicted_next.flatten(
            start_dim=1, end_dim=-1
        )  # (B, (T-1)*C*H*W)
        targ_global = target_encoded.flatten(
            start_dim=1, end_dim=-1
        )  # (B, (T-1)*C*H*W)

        vicreg_global_metrics = self.compute_vicreg_losses(pred_global, targ_global)

        # Spatial-level VICReg:
        # Permute to treat each spatial location as a separate sample:
        # predicted_next_spatial: (B*(T-1)*H*W, C)
        predicted_next_spatial = predicted_next.permute(0, 1, 3, 4, 2).flatten(
            end_dim=-2
        )  # (B*(T-1)*H*W, C)
        target_encoded_spatial = target_encoded.permute(0, 1, 3, 4, 2).flatten(
            end_dim=-2
        )  # (B*(T-1)*H*W, C)

        vicreg_spatial_metrics = self.compute_vicreg_losses(
            predicted_next_spatial, target_encoded_spatial
        )

        combined_loss = vicreg_global_metrics["loss"] + vicreg_spatial_metrics["loss"]

        # Backprop
        self.optimizer.zero_grad()
        combined_loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        # Prepare metrics for logging
        output = {
            "loss": combined_loss.item(),
            "inv_loss": (
                vicreg_global_metrics["inv-loss"].item()
                + vicreg_spatial_metrics["inv-loss"].item()
            ),
            "var_loss": (
                vicreg_global_metrics["var-loss"].item()
                + vicreg_spatial_metrics["var-loss"].item()
            ),
            "cov_loss": (
                vicreg_global_metrics["cov-loss"].item()
                + vicreg_spatial_metrics["cov-loss"].item()
            ),
            "lr": self.scheduler.get_last_lr()[0],
        }

        return output

    def validation_step(self, batch, device):
        states, _, actions = batch
        states = states.to(device)
        actions = actions.to(device)

        enc_states, predicted_next = self(states, actions)
        target_encoded = enc_states[:, 1:]

        # Compute just the global VICReg loss for validation for simplicity
        B, Tm1, C, H, W = predicted_next.shape
        pred_global = predicted_next.flatten(start_dim=1, end_dim=-1)
        targ_global = target_encoded.flatten(start_dim=1, end_dim=-1)

        vicreg_metrics = self.compute_vicreg_losses(pred_global, targ_global)
        val_loss = vicreg_metrics["loss"].item()

        output = {
            "val_loss": val_loss,
            "inv_loss": vicreg_metrics["inv-loss"].item(),
            "var_loss": vicreg_metrics["var-loss"].item(),
            "cov_loss": vicreg_metrics["cov-loss"].item(),
            "lr": self.scheduler.get_last_lr()[0],
        }

        return output


# Modify Encoder class
class Encoder2Dv3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.convnext = create_model(
            "resnet18.a1_in1k",
            pretrained=False,
            num_classes=0,
            in_chans=2,
            features_only=True,
        )

    def forward(self, x):
        # Reshape input to merge batch and trajectory dimensions
        original_shape = x.shape
        x = x.view(
            -1, *original_shape[-3:]
        )  # Reshape to [batch*trajectory, channels, height, width]
        features = self.convnext(x)[1]

        # Reshape features back to original trajectory structure
        features = features.view(
            original_shape[0], original_shape[1], *features.shape[-3:]
        )
        return features


# Modify Predictor class
class Predictor2Dv3(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Conv2d(input_dim, input_dim - 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(input_dim - 2, output_dim, kernel_size=3, padding=1),
        )

    def forward(self, encoded_o_t, action):
        # Reshape inputs
        batch_size, trajectory_length = encoded_o_t.shape[:2]

        # Reshape action to match encoded_o_t dimensions
        action = action.view(batch_size, trajectory_length - 1, 2, 1, 1)
        action = action.repeat(1, 1, 1, encoded_o_t.size(3), encoded_o_t.size(4))

        # Prepare inputs for prediction
        predictions = []
        for t in range(trajectory_length - 1):
            # Concatenate current encoded state with action
            x = torch.cat([encoded_o_t[:, t], action[:, t]], dim=1)
            pred = self.predictor(x)
            predictions.append(pred)

        return torch.stack(predictions, dim=1)


class JEPA2Dv3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.enc = Encoder2Dv3(config)
        self.pred = Predictor2Dv3(input_dim=66, output_dim=64)  # Same as original code
        # self.action_reg_net = ActionRegularizer2D(config, config.embed_dim, config.action_dim)  # Not used

        # Create optimizer and scheduler as per original code
        params = list(self.enc.parameters()) + list(self.pred.parameters())
        self.optimizer = AdamW(params, lr=1e-3)
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=1e-2,
            steps_per_epoch=config.steps_per_epoch,
            epochs=config.epochs,
        )

        # VICReg parameters from original code
        self.inv_coeff = 25.0
        self.var_coeff = 15.0
        self.cov_coeff = 1.0
        self.gamma = 1.0

    def forward(self, states, actions, teacher_forcing=True):
        # states: (B, T, 2, H, W)
        # actions: (B, T-1, 2)
        B, T, C, H, W = states.shape
        # C should be 2 as per encoder input

        if teacher_forcing:
            # Pass states directly to encoder; it handles flattening internally.
            enc_states = self.enc(states)  # (B, T, C_out, H_out, W_out)
            # Initialize predictions (same shape as enc_states)
            preds = torch.zeros_like(enc_states)  # (B, T, C_out, H_out, W_out)
            preds[:, 0] = enc_states[
                :, 0
            ]  # First predicted state is the first encoded state

            # Use predictor to predict next states
            # The predictor expects:
            #   enc_states: (B, T, C_out, H_out, W_out)
            #   actions: (B, T-1, 2)
            # returns: (B, T-1, output_dim, H_out, W_out)
            pred_states = self.pred(enc_states, actions)

            # Assign predicted states to preds at t=1 to T-1
            preds[:, 1:] = pred_states

            return preds, enc_states

        else:
            # Non-teacher forcing scenario:
            # Take the first state only
            states_0 = states[:, 0]  # (B, 2, H, W)
            # Encoder expects (B, T, 2, H, W), so add a time dimension for a single step
            enc_state = self.enc(states_0.unsqueeze(1))  # (B, 1, C_out, H_out, W_out)

            preds = [enc_state]

            # Predict step by step for remaining timesteps
            for t in range(1, T):
                action_t_minus1 = actions[:, t - 1]  # (B, 2)
                state_embed_t_minus1 = preds[-1]  # (B, 1, C_out, H_out, W_out)

                # The predictor expects a sequence along T, but we have only one timestep to predict.
                # We'll just pass a single-step sequence (T=1):
                # actions also need a time dimension: (B,1,2)
                pred_state = self.pred(
                    state_embed_t_minus1, action_t_minus1.unsqueeze(1)
                )
                # pred_state: (B, 1, C_out, H_out, W_out)

                preds.append(pred_state)

            # Stack predictions: (B, T, C_out, H_out, W_out)
            preds = torch.cat(preds, dim=1)
            preds = preds.view(B, T, -1)  # (B, T, C_out*H_out*W_out)
            return preds

    def representation_loss(self, x, y):
        # As original VICReg: MSE loss
        return F.mse_loss(x, y)

    def variance_loss(self, x, gamma):
        x = x - x.mean(dim=0)
        std = x.std(dim=0)
        var_loss = F.relu(gamma - std).mean()
        return var_loss

    def covariance_loss(self, x):
        x = x - x.mean(dim=0)
        cov = (x.T @ x) / (x.shape[0] - 1)
        cov.fill_diagonal_(0.0)
        cov_loss = cov.pow(2).sum() / x.shape[1]
        return cov_loss

    def compute_vicreg_losses(self, predicted_next, target_encoded):
        # Compute VICReg losses as in the original code
        # predicted_next, target_encoded: after flattening appropriately

        # invariance
        inv_loss = self.inv_coeff * self.representation_loss(
            predicted_next, target_encoded
        )

        # variance
        var_loss_x = self.var_coeff * self.variance_loss(predicted_next, self.gamma)
        var_loss_y = self.var_coeff * self.variance_loss(target_encoded, self.gamma)
        var_loss = (var_loss_x + var_loss_y) / 2

        # covariance
        cov_loss_x = self.cov_coeff * self.covariance_loss(predicted_next)
        cov_loss_y = self.cov_coeff * self.covariance_loss(target_encoded)
        cov_loss = (cov_loss_x + cov_loss_y) / 2

        loss = inv_loss + var_loss + cov_loss
        return {
            "loss": loss,
            "inv-loss": inv_loss,
            "var-loss": var_loss,
            "cov-loss": cov_loss,
        }

    def training_step(self, batch, device):
        states, _, actions = batch
        states = states.to(device)
        actions = actions.to(device)

        predicted_next, enc_states = self(states, actions)

        # target_encoded = enc_states[:, 1:] (B, T-1, C, H', W')
        target_encoded = enc_states[:, 1:]

        # predicted_next: (B, T-1, C, H', W')
        # For VICReg, we compute loss globally and also add a spatially-resolved term as in original code.

        # Flatten for global VICReg:
        B, Tm1, C, H, W = predicted_next.shape
        # Flatten channels + spatial:
        pred_global = predicted_next.flatten(
            start_dim=1, end_dim=-1
        )  # (B, (T-1)*C*H*W)
        targ_global = target_encoded.flatten(
            start_dim=1, end_dim=-1
        )  # (B, (T-1)*C*H*W)

        vicreg_global_metrics = self.compute_vicreg_losses(pred_global, targ_global)

        # Spatial-level VICReg:
        # Permute to treat each spatial location as a separate sample:
        # predicted_next_spatial: (B*(T-1)*H*W, C)
        predicted_next_spatial = predicted_next.permute(0, 1, 3, 4, 2).flatten(
            end_dim=-2
        )  # (B*(T-1)*H*W, C)
        target_encoded_spatial = target_encoded.permute(0, 1, 3, 4, 2).flatten(
            end_dim=-2
        )  # (B*(T-1)*H*W, C)

        vicreg_spatial_metrics = self.compute_vicreg_losses(
            predicted_next_spatial, target_encoded_spatial
        )

        combined_loss = vicreg_global_metrics["loss"] + vicreg_spatial_metrics["loss"]

        # Backprop
        self.optimizer.zero_grad()
        combined_loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        # Prepare metrics for logging
        output = {
            "loss": combined_loss.item(),
            "inv_loss": (
                vicreg_global_metrics["inv-loss"].item()
                + vicreg_spatial_metrics["inv-loss"].item()
            ),
            "var_loss": (
                vicreg_global_metrics["var-loss"].item()
                + vicreg_spatial_metrics["var-loss"].item()
            ),
            "cov_loss": (
                vicreg_global_metrics["cov-loss"].item()
                + vicreg_spatial_metrics["cov-loss"].item()
            ),
            "lr": self.scheduler.get_last_lr()[0],
        }

        return output

    def validation_step(self, batch, device):
        states, _, actions = batch
        states = states.to(device)
        actions = actions.to(device)

        enc_states, predicted_next = self(states, actions)
        target_encoded = enc_states[:, 1:]

        # Compute just the global VICReg loss for validation for simplicity
        B, Tm1, C, H, W = predicted_next.shape
        pred_global = predicted_next.flatten(start_dim=1, end_dim=-1)
        targ_global = target_encoded.flatten(start_dim=1, end_dim=-1)

        vicreg_metrics = self.compute_vicreg_losses(pred_global, targ_global)
        val_loss = vicreg_metrics["loss"].item()

        output = {
            "val_loss": val_loss,
            "inv_loss": vicreg_metrics["inv-loss"].item(),
            "var_loss": vicreg_metrics["var-loss"].item(),
            "cov_loss": vicreg_metrics["cov-loss"].item(),
            "lr": self.scheduler.get_last_lr()[0],
        }

        return output


# Model mapping and get_model function
MODEL_MAP: Dict[str, BaseModel] = {
    "JEPA": JEPA,
    "AdversarialJEPA": AdversarialJEPA,
    "InfoMaxJEPA": InfoMaxJEPA,
    "ActionRegularizationJEPA": ActionRegularizationJEPA,
    "AdversarialJEPAWithRegularization": AdversarialJEPAWithRegularization,
    "ActionRegularizationJEPA2D": ActionRegularizationJEPA2D,
    "ActionRegularizationJEPA2DFlexibleEncoder": ActionRegularizationJEPA2DFlexibleEncoder,
    # Add more models here as needed
}


def get_model(model_name: str):
    model_class = MODEL_MAP.get(model_name)
    if model_class is None:
        raise ValueError(
            f"Unknown model type: {model_name}. Available models are: {list(MODEL_MAP.keys())}"
        )
    return model_class
