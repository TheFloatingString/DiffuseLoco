"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import sys
import argparse
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

# Parse custom --ds argument before Hydra
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--ds', default='default', choices=['default', 'grand_tour', 'isaaclab'],
                    help='Dataset selection')
parser.add_argument('--grand_tour_goal_cond', action='store_true', help='Enable goal conditioning for grand_tour dataset')
args, remaining_argv = parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining_argv

if args.ds != 'grand_tour':
    try:
        from isaacgym.torch_utils import *
    except:
        print("Isaac Gym Not Installed")

import torch
torch.set_float32_matmul_precision('medium')
torch.backends.cuda.matmul.allow_tf32 = True

import hydra
from omegaconf import OmegaConf, open_dict
import pathlib

from diffusion_policy.workspace.base_workspace import BaseWorkspace
if args.ds != 'grand_tour':
    from diffusion_policy.env_runner.cyber_runner import LeggedRunner


# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("../..", "diffusion_policy","config_files")),
    config_name="diffusion_policy_tf_2.yaml"   # was cyber_diffusion_policy_medium_model.yaml
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    # Swap dataset to GrandTourDataset when --ds grand_tour is specified
    if args.ds == 'grand_tour':
        import json
        from omegaconf import OmegaConf as OC
        ds_cfg = cfg.task.dataset

        # Discover all missions, excluding the hold-out validation mission
        data_base = pathlib.Path('datasets/grand_tour')
        with open(data_base / 'config.json', 'r') as f:
            mission_configs = json.load(f)
        val_mission = 'SPX-2'
        dataset_paths = [
            str(data_base / mission)
            for mission, cfg_m in mission_configs.items()
            if mission != val_mission
            # rough terrain mix
            # and mission in ["SNOW-1", "SNOW-2", "SNOW-3", "EIG-1", "EIG-2", "GRI-1", "CYN-1", "CYN-2", "HIL-1", "PIL-1", "PIL-2", "ROOT-1", "HOB-1", "HOB-2", "HEAP-1", "KAB-1", "KAB-2", "KAB-3", "TRIM-1", "ALB-1","ALB-2", "ALB-3", "LMB-1", "LMB-2"]
            # and mission in ["HAUS-1", "ARC-5", "ARC-6", "LEICA-1", "LEICA-2"] # only use these missions for now
            # flat terrain mix 
            and mission in ["ETH-1", "ETH-2", "ETH-3", "SPX-3", "HAUS-1", "LEE-1", "ARC-5", "ARC-6", "LEICA-1", "LEICA-2"]
            and (data_base / mission).is_dir()
            and (data_base / mission / 'anymal_state_odometry' / 'timestamp').exists()
        ]

        cfg.task.dataset = OC.create({
            '_target_': 'diffusion_policy.dataset.grand_tour_dataset.GrandTourDataset',
            'dataset_path': dataset_paths,
            'horizon': ds_cfg.horizon,
            'pad_before': ds_cfg.pad_before,
            'pad_after': ds_cfg.pad_after,
            'val_ratio': ds_cfg.val_ratio,
            'seed': ds_cfg.seed,
            'frequency': 30,
        })
        cfg.task.env_runner = OC.create({
            '_target_': 'diffusion_policy.env_runner.null_runner.NullRunner',
        })
        # grand_tour obs is 36-dim (vs 45 for CyberDog): update model input sizes
        cfg.obs_dim = 36
        cfg.policy.obs_dim = 36
        cfg.policy.model.cond_dim = 36
        cfg.task.obs_dim = 36
        # hold-out mission for per-epoch RMSE validation
        with open_dict(cfg.task):
            cfg.task.grand_tour_val_mission = val_mission

        if args.grand_tour_goal_cond:
            with open_dict(cfg.policy.model):
                cfg.policy.model.separate_goal_conditioning = True
                cfg.policy.model.goal_indices = [9, 10, 11]
            # Ensure cond_dim matches obs_dim (already set to 36 above)
            cfg.policy.model.cond_dim = 36

    # --- IsaacLab generated data handler ---
    elif args.ds == 'isaaclab':
        from omegaconf import OmegaConf as OC
        ds_cfg = cfg.task.dataset

        isaaclab_data_path = pathlib.Path('../../grand_tour_code/generated_data')
        if not isaaclab_data_path.is_dir():
            # fallback: look relative to current working directory
            isaaclab_data_path = pathlib.Path('generated_data')

        cfg.task.dataset = OC.create({
            '_target_': 'diffusion_policy.dataset.isaaclab_dataset.IsaacLabDataset',
            'dataset_path': str(isaaclab_data_path.resolve()),
            'horizon': cfg.horizon,
            'pad_before': cfg.task.dataset.pad_before,
            'pad_after': cfg.task.dataset.pad_after,
            'val_ratio': ds_cfg.val_ratio,
            'seed': ds_cfg.seed,
        })
        cfg.task.env_runner = OC.create({
            '_target_': 'diffusion_policy.env_runner.null_runner.NullRunner',
        })
        # IsaacLab obs: first 36 dims (excludes prev_actions), action is 12-dim
        cfg.obs_dim = 36
        cfg.policy.obs_dim = 36
        cfg.policy.model.cond_dim = 36
        cfg.action_dim = 12
        cfg.task.obs_dim = 36
        cfg.task.action_dim = 12

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
