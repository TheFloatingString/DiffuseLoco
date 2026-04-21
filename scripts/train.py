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
parser.add_argument('--ds', default='default', choices=['default', 'grand_tour'],
                    help='Dataset selection')
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
from omegaconf import OmegaConf
import pathlib

from diffusion_policy.workspace.base_workspace import BaseWorkspace
if args.ds != 'grand_tour':
    from diffusion_policy.env_runner.cyber_runner import LeggedRunner


# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("../..", "diffusion_policy","config_files")),
    config_name="cyber_diffusion_policy_medium_model.yaml"
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    # Swap dataset to GrandTourDataset when --ds grand_tour is specified
    if args.ds == 'grand_tour':
        from omegaconf import OmegaConf as OC
        ds_cfg = cfg.task.dataset
        cfg.task.dataset = OC.create({
            '_target_': 'diffusion_policy.dataset.grand_tour_dataset.GrandTourDataset',
            'dataset_path': 'datasets/grand_tour/LEICA-1',
            'horizon': ds_cfg.horizon,
            'pad_before': ds_cfg.pad_before,
            'pad_after': ds_cfg.pad_after,
            'val_ratio': ds_cfg.val_ratio,
            'seed': ds_cfg.seed,
            'frequency': 100,
        })
        cfg.task.env_runner = OC.create({
            '_target_': 'diffusion_policy.env_runner.null_runner.NullRunner',
        })

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
