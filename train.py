from accelerate import Accelerator
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from dataset import create_wall_dataloader
from models import JEPA
from tqdm import tqdm

def setup():
    acc = Accelerator()
    device = acc.device

    data_path = "/scratch/DL24FA/train"
    dl = create_wall_dataloader(data_path, probing=False, device=device, batch_size=1028, train=True)

    model = JEPA().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.0002)

    model, opt, dl = acc.prepare(model, opt, dl)
    return acc, model, opt, dl

def train_jepa():
    acc, model, opt, dl = setup()
    model.train()

    for epoch in range(20):
        total_loss = 0
        for batch in tqdm(dl, desc=f"Epoch {epoch + 1}/20"):
            states, actions = batch.states, batch.actions
            opt.zero_grad()
            preds = model(states, actions)
            B, T, _, _, _ = states.shape
            enc = model.encoder(states.view(-1, *states.shape[2:])).view(B, T, -1)
            loss = F.mse_loss(preds[:, :-1], enc[:, 1:])
            acc.backward(loss)
            opt.step()
            total_loss += loss.item()
        acc.print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dl):.5f}")

    acc.save_state("weights/jepa_model_weights")

if __name__ == "__main__":
    train_jepa()
