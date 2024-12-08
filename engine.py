import wandb
from accelerate import Accelerator
import torch
import torch.nn.functional as F
from utils import log_embeddings_wandb
import numpy as np

from main import evaluate_model

best_normal_loss = float("inf")
best_wall_loss = float("inf")


def train_one_epoch(
    epoch, model, tdl, vdl, acc, step, config, probe_train_ds, probe_val_ds, k=1, l=1
):
    global best_normal_loss, best_wall_loss
    
    model.train()
    total_loss = 0

    for i, batch in enumerate(tdl):
        outputs = model.training_step(batch, device=acc.device)

        # Extract non-loggable data
        non_logs = outputs.pop("non_logs", {})
        states = non_logs.get("states")
        actions = non_logs.get("actions")
        enc_embeddings = non_logs.get("enc_embeddings")
        pred_embeddings = non_logs.get("pred_embeddings")

        # Rename 'loss' to 'train_loss'
        outputs["train_loss"] = outputs.pop("loss")

        total_loss += outputs["train_loss"]

        # Log all other outputs directly
        outputs["step"] = step
        wandb.log(outputs, step=step)

        # Wandb embedding visualization during training
        if (
            step % 20 == 0
            and states is not None
            and enc_embeddings is not None
            and pred_embeddings is not None
        ):
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
            acc.print(
                f"[{epoch + 1}/{config.epochs}][{i + 1}/{len(tdl)}] train batch loss: {outputs['train_loss']:.5f}"
            )

        # Periodic validation
        if (i + 1) % (len(tdl) // k) == 0:
            step, val_loss = val_one_epoch(
                epoch, model, vdl, acc, step, config, log_embeddings=True
            )
            acc.print(f"[{epoch + 1}/{config.epochs}] valid epoch loss: {val_loss:.5f}")
            acc.print(f"\n---------------------------------------\n")
            model.train()
        
        if (i + 1) % (len(tdl) // l) == 0:
            acc.print(f"------ Running Probing Evaluator for epoch {epoch + 1} ------")
            # Evaluate the model using the probing evaluator
            avg_losses = evaluate_model(acc.device, model, probe_train_ds, probe_val_ds)

            # Check and save the best model for "normal"
            if avg_losses["normal"] < best_normal_loss:
                best_normal_loss = avg_losses["normal"]
                normal_model_path = f"weights/best_normal_model_epoch_{epoch + 1}_train_iter_{i}_normal_loss_{best_normal_loss:.5f}_wall_loss_{avg_losses['wall']:.5f}.pt"
                acc.save(model.state_dict(), normal_model_path)
                wandb.save(normal_model_path)
                acc.print(
                    f"Saved best normal model with normal loss {best_normal_loss:.5f} and wall loss {avg_losses['wall']:.5f} at {normal_model_path}"
                )

            # Check and save the best model for "wall"
            if avg_losses["wall"] < best_wall_loss:
                best_wall_loss = avg_losses["wall"]
                wall_model_path = f"weights/best_wall_model_epoch_{epoch + 1}_train_iter_{i}_normal_loss_{avg_losses['normal']:.5f}_wall_loss_{best_wall_loss:.5f}.pt"
                acc.save(model.state_dict(), wall_model_path)
                wandb.save(wall_model_path)
                acc.print(
                    f"Saved best wall model with normal loss {avg_losses['normal']:.5f} and wall loss {best_wall_loss:.5f} at {wall_model_path}"
                )

            acc.print(f"-------------------------------------------------------------")

    avg_epoch_loss = total_loss / len(tdl)
    avg_losses.update({"avg_epoch_train_loss": avg_epoch_loss, "epoch": epoch + 1})

    # Log avg_epoch_train_loss at the end of the epoch
    wandb.log(avg_losses, step=step)

    return step, avg_epoch_loss


def val_one_epoch(epoch, model, vdl, acc, step, config, log_embeddings=False):
    acc.print(f"\n------- valid for epoch {epoch + 1} -------")
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(vdl):
            outputs = model.validation_step(batch)

            # Extract non-loggable data
            non_logs = outputs.pop("non_logs", {})
            states = non_logs.get("states")
            actions = non_logs.get("actions")
            enc_embeddings = non_logs.get("enc_embeddings")
            pred_embeddings = non_logs.get("pred_embeddings")

            # Rename 'loss' to 'val_loss'
            outputs["val_loss"] = outputs.pop("loss")

            val_loss += outputs["val_loss"]

            # Log all other outputs directly
            outputs["step"] = step
            wandb.log(outputs, step=step)

            # Wandb embedding visualization during validation
            if (
                log_embeddings
                and i == 0
                and states is not None
                and enc_embeddings is not None
                and pred_embeddings is not None
            ):
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
                acc.print(
                    f"[{epoch + 1}/{config.epochs}][{i + 1}/{len(vdl)}] valid batch loss: {outputs['val_loss']:.5f}"
                )

    avg_val_loss = val_loss / len(vdl)

    # Log avg_epoch_val_loss at the end of the validation
    wandb.log({"avg_epoch_val_loss": avg_val_loss, "epoch": epoch + 1}, step=step)

    return step, avg_val_loss
