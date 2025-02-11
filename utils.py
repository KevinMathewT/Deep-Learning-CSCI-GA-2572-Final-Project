import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import seaborn as sns
import numpy as np
import wandb
import torch
import glob
import os
from pathspec import PathSpec
from collections import OrderedDict
import subprocess

import gc
import torch
import torch.nn as nn

import timm
from timm.models.features import FeatureHooks


def seed_everything(seed=42):
    """Set random seed for reproducibility."""
    import random, os, torch, numpy as np
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def calculate_max_timesteps(train_step):
    """
    Calculate max timesteps for input based on train_step using a cosine increasing function.
    The timesteps grow from 2 to 17 over the first 3 epochs, and stay at 17 afterward.

    Args:
        train_step: Current training step.

    Returns:
        max_timesteps: Number of timesteps to use for training.
    """
    total_steps = 94 * 3  # Total steps for 3 epochs
    if train_step >= total_steps:
        return 17  # Cap timesteps at 17 after 3 epochs
    
    progress = train_step / total_steps  # Progress fraction [0, 1] over 3 epochs
    cos_value = 0.5 * (1 - torch.cos(torch.tensor(progress * torch.pi)))  # Flipped cosine
    max_timesteps = int(3 + cos_value * (17 - 3))  # Scale from 3 to 17
    return max_timesteps



def get_free_gpu():
    try:
        # Query GPU memory usage using nvidia-smi
        gpu_memory = subprocess.check_output(
            "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits", shell=True
        )
        gpu_memory = [int(x) for x in gpu_memory.decode("utf-8").strip().split("\n")]
        free_gpu = int(max(range(len(gpu_memory)), key=lambda i: gpu_memory[i]))
        return free_gpu
    except Exception as e:
        print("Could not find free GPU automatically:", e)
        # Fallback to first GPU if automatic detection fails
        return 0


def log_files():
    extension_depth_pairs = [('.py', 2), ('.yaml', 2)]
    collected_files = []

    base_dir = os.path.abspath(".")
    base_depth = base_dir.count(os.sep)

    for extension, max_depth in extension_depth_pairs:
        for root, dirs, files in os.walk("."):
            # Get absolute path of the current directory
            root_abs = os.path.abspath(root)
            current_depth = root_abs.count(os.sep) - base_depth

            # Skip directories that exceed the max depth
            if current_depth >= max_depth:
                dirs[:] = []  # Prune traversal by clearing 'dirs'
                continue  # Skip processing files in this directory

            # Collect files matching the extension
            for file in files:
                if file.endswith(extension):
                    collected_files.append(os.path.relpath(os.path.join(root, file)))

    return collected_files



def create_minimal_feature_model(config, feature_index):
    """
    Create a minimal feature extraction model that provides the exact output
    for the specified feature_index using TIMM's FeatureHooks mechanism.
    """
    # Step 1: Create the full model with features_only=True to get feature_info
    full_model = timm.create_model(
        config.encoder_backbone,
        pretrained=False,
        num_classes=0,
        in_chans=config.in_c,
        features_only=True
    )

    # Extract feature_info for the selected feature index
    selected_feature_layer = full_model.feature_info[feature_index]['module']
    selected_feature_channels = full_model.feature_info[feature_index]['num_chs']

    # Debugging: Print feature info and selected layer
    print("Selected Feature Layer:", selected_feature_layer)
    print("Feature Info:", full_model.feature_info)

    # Clean up full_model (not needed after extracting feature_info)
    del full_model
    gc.collect()

    # Step 2: Create the base model (no features_only)
    base_model = timm.create_model(
        config.encoder_backbone,
        pretrained=False,
        num_classes=0,
        in_chans=config.in_c
    )

    # Step 3: Construct named_modules dictionary
    named_modules = {name: module for name, module in base_model.named_modules() if name}

    # Debugging: Check if selected feature layer is in named_modules
    if selected_feature_layer not in named_modules:
        print("Available Layers in named_modules:")
        for name in named_modules.keys():
            print(name)
        raise ValueError(f"Selected feature layer '{selected_feature_layer}' not found in named_modules.")

    # Step 4: Use TIMM's FeatureHooks to capture the exact outputs
    hooks = FeatureHooks(
        hooks=[selected_feature_layer],  # List of target layer names
        named_modules=named_modules      # Valid dictionary of named modules
    )

    # Wrap the base model's forward function to capture features
    original_forward = base_model.forward

    def modified_forward(self, x):
        _ = original_forward(x)  # Run the forward pass
        return hooks.get_output([selected_feature_layer])[0]  # Extract the hooked output

    # Assign the modified forward method to the model
    base_model.forward = modified_forward.__get__(base_model, type(base_model))

    # Return the base model and the number of channels for the selected feature layer
    return base_model, selected_feature_channels



def log_embeddings_wandb(epoch, batch_idx, batch_states, batch_actions, enc_embeddings, pred_embeddings, timesteps=[0, 2, 4], phase="train", step=None):
    """
    Logs embeddings (encoded and predicted) along with input images and embedding values to Wandb.

    Args:
        epoch (int): Current epoch number.
        batch_idx (int): Current batch index.
        batch_states (torch.Tensor): Batch of input states (B, T, 2, H, W).
        batch_actions (torch.Tensor): Batch of actions (B, T-1, 2).
        enc_embeddings (torch.Tensor): Encoder output embeddings (B, T, EMBED_DIM).
        pred_embeddings (torch.Tensor): Predictor output embeddings (B, T, EMBED_DIM).
        timesteps (list): Specific timesteps to visualize for the first batch element.
        phase (str): Either "train" or "valid" to indicate the phase of logging.
        step (int, optional): Current training step for Wandb logging.
    """
    cmap = sns.color_palette("coolwarm", as_cmap=True)  # Gradient color map

    # Adjust timesteps to ensure t+1 does not exceed sequence length
    max_timestep = batch_states.shape[1] - 2  # Since we need t and t+1
    timesteps = [t for t in timesteps if t <= max_timestep]

    # Create a grid with 5 columns
    fig, axs = plt.subplots(len(timesteps), 5, figsize=(25, 5 * len(timesteps)))

    for row, t in enumerate(timesteps):
        # Input image at time t
        input_image = batch_states[0, t].permute(1, 2, 0).cpu().detach().numpy()

        # Target image at time t+1
        target_image = batch_states[0, t+1].permute(1, 2, 0).cpu().detach().numpy()

        # Identify the point of highest intensity (agent location) in channel 0 for input and target images
        agent_channel_input = input_image[..., 0]  # (H, W)
        max_pos_input = np.unravel_index(np.argmax(agent_channel_input), agent_channel_input.shape)  # (y, x)

        agent_channel_target = target_image[..., 0]  # (H, W)
        max_pos_target = np.unravel_index(np.argmax(agent_channel_target), agent_channel_target.shape)  # (y, x)

        # Actions
        action = batch_actions[0, t].cpu().detach().numpy() if t < batch_actions.shape[1] else None

        # Latent representations
        input_embed = enc_embeddings[0, t].cpu().detach().numpy()
        target_embed = enc_embeddings[0, t+1].cpu().detach().numpy()
        pred_embed = pred_embeddings[0, t+1].cpu().detach().numpy()

        # Normalize embeddings
        norm_input = Normalize()(input_embed)
        norm_target = Normalize()(target_embed)
        norm_pred = Normalize()(pred_embed)

        # Calculate distances
        squared_distance = np.sum((target_embed - pred_embed) ** 2)
        cosine_similarity = np.dot(target_embed, pred_embed) / (np.linalg.norm(target_embed) * np.linalg.norm(pred_embed) + 1e-8)

        # Create gradient-like images for embeddings
        input_emb_image = np.tile(norm_input, (5, 1))
        target_emb_image = np.tile(norm_target, (5, 1))
        pred_emb_image = np.tile(norm_pred, (5, 1))

        # Format embeddings and distance as text strings (truncated for brevity)
        input_emb_text = np.array2string(input_embed, precision=4, separator=", ", threshold=10)
        target_emb_text = np.array2string(target_embed, precision=4, separator=", ", threshold=10)
        pred_emb_text = np.array2string(pred_embed, precision=4, separator=", ", threshold=10)
        distance_text = f"Squared Distance: {squared_distance:.4f}\nCosine Similarity: {cosine_similarity:.4f}"

        # Input image and metadata
        axs[row, 0].imshow(agent_channel_input, cmap="gray")
        axs[row, 0].set_title(f"Input Image (Agent) - Timestep {t}", fontsize=14)
        axs[row, 0].axis("off")
        axs[row, 0].text(
            0.5,
            -0.2,
            f"Position: ({int(max_pos_input[0])}, {int(max_pos_input[1])})\nAction: {action if action is not None else 'None'}",
            transform=axs[row, 0].transAxes,
            fontsize=12,
            ha="center",
        )

        # Target image and metadata
        axs[row, 1].imshow(agent_channel_target, cmap="gray")
        axs[row, 1].set_title(f"Target Image (Agent) - Timestep {t+1}", fontsize=14)
        axs[row, 1].axis("off")
        axs[row, 1].text(
            0.5,
            -0.2,
            f"Position: ({int(max_pos_target[0])}, {int(max_pos_target[1])})",
            transform=axs[row, 1].transAxes,
            fontsize=12,
            ha="center",
        )

        # Input latent representation visualization
        axs[row, 2].imshow(input_emb_image, cmap=cmap, aspect=5)
        axs[row, 2].set_title("Input Latent Representation", fontsize=14)
        axs[row, 2].axis("off")
        axs[row, 2].text(
            0.5,
            -0.2,
            f"Embedding: {input_emb_text}",
            transform=axs[row, 2].transAxes,
            fontsize=10,
            ha="center",
            va="top",
            clip_on=False,
        )

        # Target latent representation visualization
        axs[row, 3].imshow(target_emb_image, cmap=cmap, aspect=5)
        axs[row, 3].set_title("Target Latent Representation", fontsize=14)
        axs[row, 3].axis("off")
        axs[row, 3].text(
            0.5,
            -0.2,
            f"Embedding: {target_emb_text}",
            transform=axs[row, 3].transAxes,
            fontsize=10,
            ha="center",
            va="top",
            clip_on=False,
        )

        # Predicted target representation visualization
        axs[row, 4].imshow(pred_emb_image, cmap=cmap, aspect=5)
        axs[row, 4].set_title("Predicted Target Representation", fontsize=14)
        axs[row, 4].axis("off")
        axs[row, 4].text(
            0.5,
            -0.2,
            f"Embedding: {pred_emb_text}\n{distance_text}",
            transform=axs[row, 4].transAxes,
            fontsize=10,
            ha="center",
            va="top",
            clip_on=False,
        )

    plt.tight_layout()
    # Log the figure to Wandb using epoch and batch index in the key
    wandb.log({f"{phase}_embedding_epoch_{epoch}_batch_{batch_idx}_grid": wandb.Image(fig)}, step=step)

    plt.close(fig)
