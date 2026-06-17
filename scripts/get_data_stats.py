"""
Print Grand Tour dataset statistics without training.

Usage:
    python get_data_stats.py [--preset full_mix|flat_mix|rough_mix] [--frequency 30]
"""

import sys
import argparse
import json
import pathlib

import numpy as np

# Make src/ importable
sys.path.insert(0, str(pathlib.Path(__file__).parent.joinpath("..", "src")))
sys.path.insert(0, str(pathlib.Path(__file__).parent.joinpath("..", "diffusion_policy")))

from dataloader import GrandTourDataloader

PRESETS = {
    "full_mix": None,  # None = all missions
    "flat_mix": ["ETH-1", "ETH-2", "ETH-3", "SPX-3", "HAUS-1", "LEE-1", "ARC-5", "ARC-6", "LEICA-1", "LEICA-2"],
    "rough_mix": ["SNOW-1", "SNOW-2", "SNOW-3", "EIG-1", "EIG-2", "GRI-1", "CYN-1", "CYN-2", "HIL-1", "PIL-1", "PIL-2", "ROOT-1", "HOB-1", "HOB-2", "HEAP-1", "KAB-1", "KAB-2", "KAB-3", "TRIM-1", "ALB-1", "ALB-2", "ALB-3", "LMB-1", "LMB-2"],
}

parser = argparse.ArgumentParser()
parser.add_argument("--preset", default="flat_mix", choices=list(PRESETS.keys()))
parser.add_argument("--frequency", type=int, default=30)
parser.add_argument("--upsample-factor", type=int, default=1)
parser.add_argument("--val-mission", default="SPX-2")
args = parser.parse_args()

data_base = pathlib.Path("datasets/grand_tour")
with open(data_base / "config.json", "r") as f:
    mission_configs = json.load(f)

# Discover available missions
available_missions = [
    mission
    for mission, cfg_m in mission_configs.items()
    if mission != args.val_mission
    and (data_base / mission).is_dir()
    and (data_base / mission / "anymal_state_odometry" / "timestamp").exists()
]

# Apply preset filter
mission_filter = PRESETS[args.preset]
if mission_filter is not None:
    selected_missions = [m for m in available_missions if m in mission_filter]
else:
    selected_missions = available_missions

print(f"=== Grand Tour Dataset Stats ===")
print(f"Preset:        {args.preset}")
print(f"Frequency:     {args.frequency} Hz")
print(f"Val mission:   {args.val_mission}")
print(f"Selected missions ({len(selected_missions)}): {selected_missions}")
print()

loader = GrandTourDataloader(
    frequency=args.frequency,
    mission_name_short=None,
    mission_names=selected_missions,
    data_base_path=str(data_base),  
    upsample_factor=args.upsample_factor,
)

total_obs_samples = 0
total_action_samples = 0

for mission_name, data in loader.missions_data.items():
    obs = loader.get_observations_isaac_lab_format(mission_name=mission_name)
    action = loader.get_actions_isaac_lab_format(mission_name=mission_name, shift_by_one=False)

    print(f"Mission {mission_name}:")
    print(f"  obs shape:    {obs.shape}  (N={obs.shape[0]}, dim={obs.shape[1]})")
    print(f"  action shape: {action.shape}  (N={action.shape[0]}, dim={action.shape[1]})")
    print(f"  duration:     {obs.shape[0] / args.frequency:.1f}s @ {args.frequency}Hz")
    print()

    total_obs_samples += obs.shape[0]
    total_action_samples += action.shape[0]

print(f"=== Totals ===")
print(f"Missions:            {len(loader.missions_data)}")
print(f"Total obs samples:   {total_obs_samples}")
print(f"Total action samples:{total_action_samples}")
print(f"Total duration:      {total_obs_samples / args.frequency:.1f}s @ {args.frequency}Hz")
