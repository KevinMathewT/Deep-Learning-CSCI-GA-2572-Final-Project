# train.py

import wandb
from accelerate import Accelerator
import torch
from dataset import WallDataset
from models import get_model
from tqdm import tqdm
import numpy as np
from dataclasses import asdict

from configs import JEPAConfig
from utils import seed_everything
from engine import train_one_epoch, val_one_epoch
from omegaconf import OmegaConf
import argparse

def setup(config):
    acc = Accelerator()
    device = acc.device

    data_path = config.data_path

    # Load the dataset and split into train and validation datasets
    full_dataset = WallDataset(data_path, probing=False, device=device)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # Create train and validation data loaders
    tdl = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True, pin_memory=False
    )
    vdl = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False, pin_memory=False
    )

    # Get the model class from the model name
    ModelClass = get_model(config.model_type)
    model = ModelClass(config).to(device)

    # Prepare the components with the Accelerator
    model, tdl, vdl = acc.prepare(model, tdl, vdl)
    return acc, model, tdl, vdl

def train_jepa(config):
    # Convert the JEPAConfig dataclass to a dictionary
    config_dict = asdict(config)

    # Initialize wandb with the configuration
    wandb.init(project="DL Final Project", config=config_dict, settings=wandb.Settings(code_dir="."))
    # wandb.run.log_code(".")
    # Log only .py and .yaml files
    wandb.run.log_code(".", include=["*.py", "*.yaml"])

    acc, model, tdl, vdl = setup(config)
    step = 0  # Initialize step counter locally

    for epoch in range(config.epochs):
        step, avg_epoch_loss = train_one_epoch(epoch, model, tdl, vdl, acc, step, config, k=2)
        acc.print(f"[{epoch + 1}/{config.epochs}] train epoch loss: {avg_epoch_loss:.5f}")

    acc.save_state(f"weights/{config.model_type}_model_weights")
    wandb.finish()

# python -m train --config config/jepa_config.yaml
def main():
    parser = argparse.ArgumentParser(description='Train different models.')
    parser.add_argument('--model', type=str, default='JEPA', help='Model type: JEPA, AdversarialJEPA, InfoMaxJEPA')
    parser.add_argument('--config', type=str, default='config/jepa_config.yaml', help='Path to configuration file')
    args = parser.parse_args()

    config = JEPAConfig.parse_from_file(args.config)
    config.model_type = args.model  # Override model type if specified

    seed_everything(42)  # Set the seed for reproducibility
    train_jepa(config)

if __name__ == "__main__":
    main()
