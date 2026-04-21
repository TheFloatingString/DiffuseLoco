import copy
import pathlib
import sys
from typing import Dict, List, Optional

import numpy as np
import torch

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer

# Make src/ importable regardless of working directory
_src_path = str(pathlib.Path(__file__).parents[3] / "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)
from dataloader import GrandTourDataloader


class GrandTourDataset(BaseLowdimDataset):
    """
    Wraps GrandTourDataloader into the BaseLowdimDataset interface.

    dataset_path should point to a single mission directory, e.g.
        datasets/grand_tour/LEICA-1
    The parent directory must contain config.json with mission metadata.
    """

    def __init__(
        self,
        dataset_path: str,
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        val_ratio: float = 0.0,
        seed: int = 42,
        frequency: int = 100,
        mission_names: Optional[List[str]] = None,
    ):
        super().__init__()

        dataset_path = pathlib.Path(dataset_path)
        data_base_path = str(dataset_path.parent)
        mission_name_short = dataset_path.name

        loader = GrandTourDataloader(
            frequency=frequency,
            mission_name_short=mission_name_short if mission_names is None else None,
            mission_names=mission_names,
            data_base_path=data_base_path,
        )

        # Build an in-memory ReplayBuffer; each mission becomes one episode
        replay_buffer = ReplayBuffer.create_empty_numpy()
        missions = list(loader.missions_data.keys())
        for mission in missions:
            obs = loader.get_observations_isaac_lab_format(mission_name=mission)
            action = loader.get_actions_isaac_lab_format(mission_name=mission, shift_by_one=True)
            # obs and action are both (N-1, D); treat the full mission as one episode
            replay_buffer.add_episode({"obs": obs.astype(np.float32), "action": action.astype(np.float32)})

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = ~val_mask

        self.sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.replay_buffer = replay_buffer
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "obs": self.replay_buffer["obs"],
            "action": self.replay_buffer["action"],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"])

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        return dict_apply(sample, torch.from_numpy)
