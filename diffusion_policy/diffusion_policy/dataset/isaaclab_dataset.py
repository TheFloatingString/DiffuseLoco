import copy
import pathlib
from typing import Dict, Optional

import numpy as np
import torch

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer


class IsaacLabDataset(BaseLowdimDataset):
    """
    Wraps IsaacLab-generated CSV data into the BaseLowdimDataset interface.

    Expects a directory containing:
        observations.csv      -- (N, obs_dim)
        actions.csv           -- (N, action_dim)
        next_observations.csv -- (N, obs_dim)
        rewards.csv           -- (N, 1)
        dones.csv             -- (N, 1)
        env_ids.csv           -- (N, 1)
        metadata.json         -- metadata dict

    Each robot (env_id) is treated as its own episode.
    """

    def __init__(
        self,
        dataset_path: str,
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        val_ratio: float = 0.0,
        seed: int = 42,
        obs_key: str = "obs",
        action_key: str = "action",
        offset: int = 0,
    ):
        super().__init__()

        dataset_path = pathlib.Path(dataset_path)
        if not dataset_path.is_dir():
            raise FileNotFoundError(f"IsaacLab dataset directory not found: {dataset_path}")

        # Load CSVs
        obs = np.loadtxt(dataset_path / "observations.csv", delimiter=",", dtype=np.float32)
        action = np.loadtxt(dataset_path / "actions.csv", delimiter=",", dtype=np.float32)
        dones = np.loadtxt(dataset_path / "dones.csv", delimiter=",", dtype=np.int64)
        env_ids = np.loadtxt(dataset_path / "env_ids.csv", delimiter=",", dtype=np.int64)

        # Only keep first 36 obs dimensions (exclude prev_actions)
        obs = obs[:, :36]

        if obs.ndim == 1:
            obs = obs.reshape(-1, 1)
        if action.ndim == 1:
            action = action.reshape(-1, 1)
        if dones.ndim == 1:
            dones = dones.reshape(-1, 1)
        if env_ids.ndim == 1:
            env_ids = env_ids.reshape(-1, 1)

        # dones may be bool or 0/1; flatten to 1D
        dones = dones.ravel()
        env_ids = env_ids.ravel()

        # Split into per-robot episodes
        replay_buffer = ReplayBuffer.create_empty_numpy()
        unique_envs = np.unique(env_ids)
        for env_id in unique_envs:
            mask = env_ids == env_id
            env_obs = obs[mask]
            env_action = action[mask]
            env_dones = dones[mask]
            if offset > 0:
                env_obs = env_obs[:-offset]
                env_action = env_action[offset:]
                env_dones = env_dones[:-offset]
            elif offset < 0:
                env_obs = env_obs[-offset:]
                env_action = env_action[:offset]
                env_dones = env_dones[-offset:]
            # episode ends where done==1 or at the last step of this robot's data
            start_idx = 0
            for i in range(len(env_dones)):
                if env_dones[i] == 1:
                    ep_obs = env_obs[start_idx : i + 1]
                    ep_action = env_action[start_idx : i + 1]
                    if len(ep_obs) > 0:
                        replay_buffer.add_episode({obs_key: ep_obs, action_key: ep_action})
                    start_idx = i + 1
            if start_idx < len(env_dones):
                ep_obs = env_obs[start_idx:]
                ep_action = env_action[start_idx:]
                if len(ep_obs) > 0:
                    replay_buffer.add_episode({obs_key: ep_obs, action_key: ep_action})

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
        self.obs_key = obs_key
        self.action_key = action_key

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
            self.obs_key: self.replay_buffer[self.obs_key],
            self.action_key: self.replay_buffer[self.action_key],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer[self.action_key])

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        return dict_apply(sample, torch.from_numpy)
