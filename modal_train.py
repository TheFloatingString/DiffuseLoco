"""
Run DiffuseLoco grand_tour training on Modal.

One-time setup:

    modal volume put diffuseloco-grand-tour-data ./datasets/grand_tour grand_tour
    modal secret create wandb-secret WANDB_API_KEY=<key> WANDB_ENTITY=<entity>

Usage:

    export PYTHONIOENCODING=utf-8
    modal run modal_train.py
    modal run modal_train.py -- --offset 100 --frequency 50 --grand-tour-goal-cond
"""

import subprocess
import sys

import modal

APP_DIR = "/root/DiffuseLoco"

app = modal.App("diffuseloco-train")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch",
        "torchvision",
        "torchaudio",
        "numpy==1.23.5",
        "numba",
        "scipy",
        "matplotlib",
        "zarr",
        "numcodecs",
        "hydra-core",
        "omegaconf",
        "einops",
        "tqdm",
        "dill",
        "wandb",
        "diffusers",
        "threadpoolctl",
        "termcolor",
    )
    .add_local_dir(
        ".",
        remote_path=APP_DIR,
        ignore=[
            "datasets",
            ".git",
            "legged_gym",
            "rsl_rl",
            "csrc",
            "source_ckpts",
            "docs",
            "outputs",
            "*.png",
        ],
        copy=True,
    )
    .run_commands(f"pip install -e {APP_DIR}/diffusion_policy")
)

dataset_volume = modal.Volume.from_name("diffuseloco-grand-tour-data", create_if_missing=True)
outputs_volume = modal.Volume.from_name("diffuseloco-outputs", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    volumes={
        f"{APP_DIR}/datasets/grand_tour": dataset_volume,
        f"{APP_DIR}/outputs": outputs_volume,
    },
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=6 * 60 * 60,
)
def train_remote(extra_args: list[str]):
    import os

    os.chdir(APP_DIR)
    try:
        subprocess.run(
            [sys.executable, "scripts/train.py", "--ds", "grand_tour", *extra_args],
            check=True,
        )
    finally:
        outputs_volume.commit()


@app.local_entrypoint()
def main(
    offset: int = 0,
    frequency: int = 30,
    grand_tour_goal_cond: bool = False,
    upsample_factor: int = 1,
):
    args = [
        "--offset", str(offset),
        "--frequency", str(frequency),
        "--upsample_factor", str(upsample_factor),
    ]
    if grand_tour_goal_cond:
        args.append("--grand_tour_goal_cond")

    train_remote.remote(args)
