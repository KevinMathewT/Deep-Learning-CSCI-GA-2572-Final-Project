from typing import NamedTuple
import torch
import numpy as np


class WallSample(NamedTuple):
    states: torch.Tensor
    locations: torch.Tensor
    actions: torch.Tensor


import random

class WallDataset:
    def __init__(self, data_path, probing=False, cache_probability=0):
        self.states_mmap = np.load(f"{data_path}/states.npy", mmap_mode="r")
        self.actions_mmap = np.load(f"{data_path}/actions.npy", mmap_mode="r")

        if probing:
            self.locations_mmap = np.load(f"{data_path}/locations.npy", mmap_mode="r")
        else:
            self.locations_mmap = None

        self.states_cache = {}
        self.actions_cache = {}
        self.locations_cache = {} if probing else None
        self.cache_probability = cache_probability  # Probability of caching

    def __len__(self):
        return len(self.states_mmap)

    def __getitem__(self, i):
        # Cache states based on probability
        if i not in self.states_cache and random.random() < self.cache_probability:
            self.states_cache[i] = torch.from_numpy(np.array(self.states_mmap[i])).float()
        states = self.states_cache.get(i, torch.from_numpy(np.array(self.states_mmap[i])).float())

        # Cache actions based on probability
        if i not in self.actions_cache and random.random() < self.cache_probability:
            self.actions_cache[i] = torch.from_numpy(np.array(self.actions_mmap[i])).float()
        actions = self.actions_cache.get(i, torch.from_numpy(np.array(self.actions_mmap[i])).float())

        # Cache locations if probing, based on probability
        if self.locations_mmap is not None:
            if i not in self.locations_cache and random.random() < self.cache_probability:
                self.locations_cache[i] = torch.from_numpy(np.array(self.locations_mmap[i])).float()
            locations = self.locations_cache.get(i, torch.from_numpy(np.array(self.locations_mmap[i])).float())
        else:
            locations = torch.empty(0)

        return WallSample(states=states, locations=locations, actions=actions)





def create_wall_dataloader(
    data_path,
    probing=False,
    device="cuda",
    batch_size=64,
    train=True,
):
    ds = WallDataset(
        data_path=data_path,
        probing=probing,
        # device=device,
    )

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size,
        shuffle=train,
        drop_last=True,
        pin_memory=False,
        num_workers=4,
    )

    return loader

if __name__ == "__main__":
    # test for best num_workers

    import time
    from tqdm import tqdm
    from torch.utils.data import DataLoader

    data_path = "/scratch/DL24FA"

    ds = WallDataset(
        data_path=f"{data_path}/probe_normal/train",
        probing=True,
        # device=device,
    )

    for workers in [0, 2, 4, 8, 16, 24]:
        for pin_memory in [True, False]:
            start_time = time.time()
            loader = DataLoader(ds, batch_size=64, num_workers=workers, pin_memory=pin_memory)
            for _ in range(5):
                for _ in tqdm(loader, leave=False):
                    pass
            print(f"original | num_workers={workers}, pin_memory={pin_memory}, time={time.time() - start_time:.2f}s")
            del loader

    
    for workers in [0, 2, 4, 8, 16, 24]:
        for pin_memory in [True, False]:
            start_time = time.time()
            full_dataset = WallDataset(f"{data_path}/train", probing=False) # , device=device)
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

            # Create train and validation data loaders
            tdl = torch.utils.data.DataLoader(
                train_dataset, batch_size=1024, shuffle=True, drop_last=True, pin_memory=pin_memory, num_workers=workers,
            )
            vdl = torch.utils.data.DataLoader(
                val_dataset, batch_size=1024, shuffle=False, drop_last=False, pin_memory=pin_memory, num_workers=workers,
            )
            for _ in range(2):
                for _ in tqdm(tdl, leave=False):
                    pass
                for _ in tqdm(vdl, leave=False):
                    pass
            print(f"tdl-vdl | num_workers={workers}, pin_memory={pin_memory}, time={time.time() - start_time:.2f}s")
