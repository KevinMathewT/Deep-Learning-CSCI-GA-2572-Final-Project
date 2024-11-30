import wandb
from accelerate import Accelerator
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from dataset import WallDataset
from models import JEPA
from tqdm import tqdm
import numpy as np

from configs import *
from utils import seed_everything, log_embeddings_wandb
from engine import train_one_epoch, val_one_epoch


def setup():
    acc = Accelerator()
    device = acc.device

    data_path = "/scratch/DL24FA/train"
    
    # Load the dataset and split into train and validation datasets
    full_dataset = WallDataset(data_path, probing=False, device=device)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Create train and validation data loaders
    tdl = torch.utils.data.DataLoader(
        train_dataset, batch_size=BS, shuffle=True, drop_last=True, pin_memory=False
    )
    vdl = torch.utils.data.DataLoader(
        val_dataset, batch_size=BS, shuffle=False, drop_last=False, pin_memory=False
    )

    model = JEPA().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    # Prepare the components with the Accelerator
    model, opt, tdl, vdl = acc.prepare(model, opt, tdl, vdl)
    return acc, model, opt, tdl, vdl


def train_jepa():
    # Initialize wandb
    wandb.init(project="DL Final Project", config={"learning_rate": LR, "batch_size": BS, "epochs": EPOCHS})

    acc, model, opt, tdl, vdl = setup()
    step = 0  # Initialize step counter locally

    for epoch in range(EPOCHS):
        step, avg_epoch_loss = train_one_epoch(epoch, model, opt, tdl, vdl, acc, step, k=2)  # Correctly unpack
        acc.print(f"[{epoch + 1}/{EPOCHS}] train epoch loss: {avg_epoch_loss:.5f}")  # Log the average loss for the epoch

    acc.save_state("weights/jepa_model_weights")
    wandb.finish()



if __name__ == "__main__":
    seed_everything(42)  # Set the seed for reproducibility
    train_jepa()
