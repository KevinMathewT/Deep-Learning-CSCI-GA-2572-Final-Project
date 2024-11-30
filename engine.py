import wandb
from accelerate import Accelerator
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from utils import log_embeddings_wandb
import numpy as np

from configs import *

def train_one_epoch(epoch, model, opt, tdl, vdl, acc, step, k=1):
    model.train()
    total_loss = 0
    last_5_losses = []

    for i, batch in enumerate(tdl):
        states, actions = batch.states, batch.actions  # Get states and actions
        opt.zero_grad()
        preds = model(states, actions)
        B, T, _, _, _ = states.shape
        enc = model.enc(states.view(-1, *states.shape[2:])).view(B, T, -1)  # Encode the states
        loss = F.mse_loss(preds[:, 1:], enc[:, 1:])
        acc.backward(loss)
        grad_norm = clip_grad_norm_(model.parameters(), max_norm=100000.0)  # Clip gradients
        opt.step()

        # Update loss trackers
        total_loss += loss.item()
        last_5_losses.append(loss.item())
        if len(last_5_losses) > 5:
            last_5_losses.pop(0)

        # Logging train metrics
        avg_train_loss = total_loss / (i + 1)
        wandb.log({
            "curr_train_loss": loss.item(),
            "grad_norm": grad_norm,
            "train_loss_last_5_batches": np.mean(last_5_losses),
            "avg_epoch_train_loss": avg_train_loss,
            "step": step,
        }, step=step)

        # Wandb embedding visualization during training
        if step % 20 == 0:  # Log every 20 steps
            log_embeddings_wandb(
                epoch=epoch,
                batch_idx=i,
                batch_states=states,
                batch_actions=actions,
                enc_embeddings=enc,
                pred_embeddings=preds,
                timesteps=[0, T // 3, 2 * T // 3],
                phase="train",
                step=step,  # Pass the current step for Wandb logging
            )

        step += 1  # Increment step counter

        # Print loss for every 5th batch or first/last batch
        if i == 0 or (i + 1) % 5 == 0 or i == (len(tdl) - 1):
            acc.print(f"[{epoch + 1}/{EPOCHS}][{i + 1}/{len(tdl)}] train batch loss: {loss.item():.5f}")

        # Periodic validation
        if (i + 1) % (len(tdl) // k) == 0:
            step, val_loss = val_one_epoch(epoch, model, vdl, acc, step, log_embeddings=True)
            acc.print(f"[{epoch + 1}/{EPOCHS}] valid epoch loss: {val_loss:.5f}")
            acc.print(f"\n---------------------------------------\n")
            model.train()

    avg_epoch_loss = total_loss / len(tdl)
    return step, avg_epoch_loss


def val_one_epoch(epoch, model, vdl, acc, step, log_embeddings=False):
    acc.print(f"\n------- valid for epoch {epoch} -------")
    model.eval()
    val_loss = 0
    last_5_losses = []

    with torch.no_grad():
        for i, batch in enumerate(vdl):
            states, actions = batch.states, batch.actions
            preds = model(states, actions)
            B, T, _, _, _ = states.shape
            enc = model.enc(states.view(-1, *states.shape[2:])).view(B, T, -1)
            loss = F.mse_loss(preds[:, :-1], enc[:, 1:])
            val_loss += loss.item()

            # Update loss trackers
            last_5_losses.append(loss.item())
            if len(last_5_losses) > 5:
                last_5_losses.pop(0)

            avg_val_loss = val_loss / (i + 1)
            wandb.log({
                "curr_val_loss": loss.item(),
                "val_loss_last_5_batches": np.mean(last_5_losses),
                "avg_epoch_val_loss": avg_val_loss,
                "step": step,
            }, step=step)

            # Wandb embedding visualization during validation
            if log_embeddings and i == 0:  # Log embeddings for the first batch only
                log_embeddings_wandb(
                    epoch=epoch,
                    batch_idx=i,
                    batch_states=states,
                    batch_actions=actions,
                    enc_embeddings=enc,
                    pred_embeddings=preds,
                    timesteps=[0, T // 3, 2 * T // 3],
                    phase="valid",
                    step=step,  # Pass the current step for Wandb logging
                )

            step += 1  # Increment step after each validation batch

            # Print validation loss for every 5th batch or first/last batch
            if i == 0 or (i + 1) % 5 == 0 or i == (len(vdl) - 1):
                acc.print(f"[{epoch + 1}/{EPOCHS}][{i + 1}/{len(vdl)}] valid batch loss: {loss.item():.5f}")

    avg_val_loss = val_loss / len(vdl)
    return step, avg_val_loss

