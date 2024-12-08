import torch

def get_optimizer(config, parameters):
    if config.optimizer_type == "adam":
        optimizer = torch.optim.Adam(
            parameters, 
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=1e-5,
        )
    elif config.optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(
            parameters, 
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=1e-5,
        )
    elif config.optimizer_type == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, 
            lr=config.learning_rate,
            alpha=0.99,
            eps=1e-08,
            weight_decay=0,
            momentum=0
        )
    elif config.optimizer_type == "sgd":
        optimizer = torch.optim.SGD(
            parameters, 
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=0.0005
        )
    else:
        raise ValueError("Invalid optimizer type")
    return optimizer


def get_scheduler(optimizer, config):
    scheduler_type = config.scheduler_type
    if scheduler_type == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            steps_per_epoch=config.steps_per_epoch,
            epochs=config.epochs,
            pct_start=0.05,
            anneal_strategy="cos",
        )
    elif scheduler_type == "linear":
        warmup_epochs = 1
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs),
                torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=config.epochs - warmup_epochs)
            ],
            milestones=[warmup_epochs]
        )
    elif scheduler_type == "cosine":
        warmup_epochs = 1
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs),
                torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs - warmup_epochs, eta_min=0)
            ],
            milestones=[warmup_epochs]
        )
    else:
        raise ValueError("Invalid scheduler type")
    return scheduler