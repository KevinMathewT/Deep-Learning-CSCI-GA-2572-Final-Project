import wandb
from accelerate import Accelerator
import torch
import torch.nn.functional as F
from utils import log_embeddings_wandb
import numpy as np

def train_one_epoch(epoch, model, tdl, vdl, acc, step, config, k=1):
    model.train()
    total_loss = 0

    for i, batch in enumerate(tdl):
        outputs = model.training_step(batch, device=acc.device)

        # Extract non-loggable data
        non_logs = outputs.pop('non_logs', {})
        states = non_logs.get('states')
        actions = non_logs.get('actions')
        enc_embeddings = non_logs.get('enc_embeddings')
        pred_embeddings = non_logs.get('pred_embeddings')

        # Rename 'loss' to 'train_loss'
        outputs['train_loss'] = outputs.pop('loss')

        total_loss += outputs['train_loss']

        # Log all other outputs directly
        outputs['step'] = step
        wandb.log(outputs, step=step)

        # Wandb embedding visualization during training
        if step % 20 == 0 and states is not None and enc_embeddings is not None and pred_embeddings is not None:
            T = states.shape[1]
            # log_embeddings_wandb(
            #     epoch=epoch,
            #     batch_idx=i,
            #     batch_states=states,
            #     batch_actions=actions,
            #     enc_embeddings=enc_embeddings,
            #     pred_embeddings=pred_embeddings,
            #     timesteps=[0, T // 3, 2 * T // 3],
            #     phase="train",
            #     step=step,
            # )

        step += 1

        # Print loss for every 5th batch or first/last batch
        if i == 0 or (i + 1) % 5 == 0 or i == (len(tdl) - 1):
            acc.print(f"[{epoch + 1}/{config.epochs}][{i + 1}/{len(tdl)}] train batch loss: {outputs['train_loss']:.5f}")

        # Periodic validation
        if (i + 1) % (len(tdl) // k) == 0:
            step, val_loss = val_one_epoch(epoch, model, vdl, acc, step, config, log_embeddings=True)
            acc.print(f"[{epoch + 1}/{config.epochs}] valid epoch loss: {val_loss:.5f}")
            acc.print(f"\n---------------------------------------\n")
            model.train()

    avg_epoch_loss = total_loss / len(tdl)

    # Log avg_epoch_train_loss at the end of the epoch
    wandb.log({"avg_epoch_train_loss": avg_epoch_loss, "epoch": epoch + 1}, step=step)

    return step, avg_epoch_loss


def val_one_epoch(epoch, model, vdl, acc, step, config, log_embeddings=False):
    acc.print(f"\n------- valid for epoch {epoch} -------")
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(vdl):
            outputs = model.validation_step(batch)

            # Extract non-loggable data
            non_logs = outputs.pop('non_logs', {})
            states = non_logs.get('states')
            actions = non_logs.get('actions')
            enc_embeddings = non_logs.get('enc_embeddings')
            pred_embeddings = non_logs.get('pred_embeddings')

            # Rename 'loss' to 'val_loss'
            outputs['val_loss'] = outputs.pop('loss')

            val_loss += outputs['val_loss']

            # Log all other outputs directly
            outputs['step'] = step
            wandb.log(outputs, step=step)

            # Wandb embedding visualization during validation
            if log_embeddings and i == 0 and states is not None and enc_embeddings is not None and pred_embeddings is not None:
                T = states.shape[1]
                # log_embeddings_wandb(
                #     epoch=epoch,
                #     batch_idx=i,
                #     batch_states=states,
                #     batch_actions=actions,
                #     enc_embeddings=enc_embeddings,
                #     pred_embeddings=pred_embeddings,
                #     timesteps=[0, T // 3, 2 * T // 3],
                #     phase="valid",
                #     step=step,
                # )

            step += 1

            # Print validation loss for every 5th batch or first/last batch
            if i == 0 or (i + 1) % 5 == 0 or i == (len(vdl) - 1):
                acc.print(f"[{epoch + 1}/{config.epochs}][{i + 1}/{len(vdl)}] valid batch loss: {outputs['val_loss']:.5f}")

    avg_val_loss = val_loss / len(vdl)

    # Log avg_epoch_val_loss at the end of the validation
    wandb.log({"avg_epoch_val_loss": avg_val_loss, "epoch": epoch + 1}, step=step)

    return step, avg_val_loss
