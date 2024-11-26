from accelerate import Accelerator
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from dataset import create_wall_dataloader
from tqdm import tqdm
from models import JEPA

def train_jepa():
    accelerator = Accelerator()
    device = accelerator.device

    # Data loaders
    data_path = "/scratch/DL24FA/train"
    train_loader = create_wall_dataloader(data_path, probing=False, device=device, batch_size=1028, train=True)

    # Initialize model and optimizer
    model = JEPA().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

    # Prepare with accelerator
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    # Training loop
    model.train()
    for epoch in range(20):
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/20"):
            states, actions = batch.states, batch.actions
            optimizer.zero_grad()

            # Forward pass
            predictions = model(states, actions)  # (B, T, 256)

            # Reshape states for encoding
            batch_size, seq_len, _, _, _ = states.shape
            states_reshaped = states.view(-1, *states.shape[2:])  # (B, T, 2, 65, 65) -> (B*T, 2, 65, 65)

            # Encode ground truth states
            encoded_states = model.encoder(states_reshaped)  # (B*T, 2, 65, 65) -> (B*T, 256)
            target_reprs = encoded_states.view(batch_size, seq_len, -1)  # (B*T, 256) -> (B, T, 256)

            # JEPA loss: Distance in representation space
            loss = F.mse_loss(predictions[:, :-1], target_reprs[:, 1:])  # Compare predictions and shifted target states

            # Backward pass
            accelerator.backward(loss)
            optimizer.step()

            total_loss += loss.item()

        accelerator.print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.5f}")

    accelerator.save_state("weights/jepa_model_weights")

if __name__ == "__main__":
    train_jepa()
