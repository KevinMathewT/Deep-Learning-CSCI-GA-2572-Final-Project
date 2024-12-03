import glob
import wandb
from accelerate import Accelerator
import torch
from dataset import WallDataset
from main import evaluate_model, load_data
from models import get_model
from tqdm import tqdm
import numpy as np
from dataclasses import asdict

from configs import JEPAConfig
from utils import seed_everything, log_files_by_extensions
from engine import train_one_epoch, val_one_epoch
from omegaconf import OmegaConf
import argparse
from pprint import pprint

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

    # Calculate steps_per_epoch
    config.steps_per_epoch = len(tdl)

    # Get the model class from the model name
    ModelClass = get_model(config.model_type)
    model = ModelClass(config).to(device)

    # Prepare the components with the Accelerator
    model, tdl, vdl = acc.prepare(model, tdl, vdl)
    return acc, model, tdl, vdl

def train_jepa(config):
    # Convert the JEPAConfig dataclass to a dictionary
    config_dict = asdict(config)

    print("Initializing WandB...")

    # Initialize wandb with the configuration
    wandb.init(project="DL Final Project", config=config_dict, settings=wandb.Settings(code_dir="."))

    print("WandB initialized.")
    print("Logging files to WandB...")

    for f in log_files_by_extensions([".py", ".yaml", ".json", ".ipynb", ".md", ".txt", ".sh", ".gitignore", ".lock", ".toml"]):
        wandb.save(f)

    print('WandB logging complete.')

    acc, model, tdl, vdl = setup(config)
    step = 0  # Initialize step counter locally

    acc.print("Setup complete. Starting training...")

    for epoch in range(config.epochs):
        # step, avg_epoch_loss = train_one_epoch(epoch, model, tdl, vdl, acc, step, config, k=2)
        # acc.print(f"[{epoch + 1}/{config.epochs}] train epoch loss: {avg_epoch_loss:.5f}")

        acc.print(f"------ Running Probing Evaluator for epoch {epoch + 1} ------")
        # Evaluate the model using the probing evaluator
        probe_train_ds, probe_val_ds = load_data(acc.device)
        evaluate_model(acc.device, model, probe_train_ds, probe_val_ds)
        acc.print(f"-------------------------------------------------------------")
        

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

    print("------ Configuration Parameters -----")
    pprint(config)
    print("-------------------------------------")

    seed_everything(42)  # Set the seed for reproducibility
    train_jepa(config)

if __name__ == "__main__":
    main()
