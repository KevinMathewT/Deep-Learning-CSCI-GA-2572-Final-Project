import os
import glob
import wandb
from accelerate import Accelerator
import torch
from optimized_dataset import WallDataset
from main import evaluate_model, load_data, load_expert_data
from models import get_model
from tqdm import tqdm
import numpy as np
from dataclasses import asdict

from configs import JEPAConfig
from utils import log_files, seed_everything, get_free_gpu
from engine import train_one_epoch, val_one_epoch
from omegaconf import OmegaConf
import argparse
from pprint import pprint

def setup(config):
    # Select a free GPU
    free_gpu = get_free_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(free_gpu)
    print(f"Using GPU {free_gpu}")

    acc = Accelerator()
    device = acc.device

    data_path = config.data_path

    # Load the dataset and split into train and validation datasets
    print("Loading dataset into memory...")
    full_dataset = WallDataset(data_path, probing=False) # , device=device)
    print("Dataset loaded.")
    train_size = int(0.99 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # Create train and validation data loaders
    tdl = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True, pin_memory=False, num_workers=4,
    )
    vdl = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False, pin_memory=False, num_workers=4,
    )

    # Calculate steps_per_epoch
    config.steps_per_epoch = len(tdl)

    # Get the model class from the model name
    ModelClass = get_model(config.model_type)
    model = ModelClass(config).to(device)

    print("---------------- Model Summary ----------------")
    print(model)
    print("----------------------------------------------")

    #Model Parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")

    # Prepare the components with the Accelerator
    model, tdl, vdl = acc.prepare(model, tdl, vdl)
    return acc, model, tdl, vdl

def train_jepa(config):
    # Convert the JEPAConfig dataclass to a dictionary
    config_dict = asdict(config)

    # Initialize wandb with the configuration
    print("Initializing WandB...")
    wandb.init(project="DL Final Project", config=config_dict, settings=wandb.Settings(code_dir="."))
    print("WandB initialized.")

    print("------ Configuration Parameters -----")
    pprint(config)
    print("-------------------------------------")

    print("Logging files to WandB...")
    for f in log_files():
        print(f"Logging file: {f}")
        wandb.save(f)
    print('WandB logging complete.')

    acc, model, tdl, vdl = setup(config)
    step = 0  # Initialize step counter locally

    acc.print("Setup complete. Starting training...")

    # prober setup
    probe_train_ds, probe_val_ds = load_data(acc.device)
    probe_train_expert_ds, probe_val_expert_ds = load_expert_data(acc.device)
    
    for epoch in range(config.epochs):
        step, avg_epoch_loss = train_one_epoch(epoch, model, tdl, vdl, acc, step, config, probe_train_ds, probe_val_ds, probe_train_expert_ds, probe_val_expert_ds, k=2)
        acc.print(f"[{epoch + 1}/{config.epochs}] train epoch loss: {avg_epoch_loss:.5f}")        

    # acc.save_state(f"weights/{config.model_type}_model_weights")
    wandb.finish()

# python -m train --config config/jepa_config.yaml
# python -m train --config config/adv_jepa_config.yaml
# python -m train --config config/areg_jepa_config.yaml
# python -m train --config config/areg_adv_jepa_config.yaml
# python -m train --config config/areg_adv_jepa_config_higher_delta.yaml
# python -m train --config config/areg_adv_jepa_config_lower_delta.yaml
# python -m train --config config/areg_jepa_2d_config.yaml
# python -m train --config config/areg_jepa_2d_config_higher_embed_dim.yaml
# python -m train --config config/areg_jepa_2d_config_lower_embed_dim.yaml
# python -m train --config config/areg_jepa_2d_config_lr_0_001.yaml
# python -m train --config config/areg_jepa_2d_config_covariance_0.yaml
# python -m train --config config/areg_jepa_2d_vicreg_config.yaml
# python -m train --config config/areg_jepa_2d_config_smaller_bs.yaml
# python -m train --config config/areg_jepa_2d_config_adamw.yaml
# python -m train --config config/areg_jepa_2d_config_adamw_linear.yaml
# python -m train --config config/areg_jepa_flex_2d_config.yaml
# python -m train --config config/areg_jepa_2d_v2.yaml
# python -m train --config config/areg_jepa_flex_2d_config_effnet.yaml
# python -m train --config config/areg_jepa_2d_v2_config.yaml
# python -m train --config config/areg_jepa_2d_config_v0.yaml
# python -m train --config config/areg_jepa_2d_config_v1.yaml
# python -m train --config config/areg_jepa_2d_config_v2.yaml
def main():
    parser = argparse.ArgumentParser(description='Train different models.')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    args = parser.parse_args()

    config = JEPAConfig.parse_from_file(args.config)

    seed_everything(42)  # Set the seed for reproducibility
    train_jepa(config)

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)

    main()
