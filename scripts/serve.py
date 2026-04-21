"""
Usage:
    python scripts/serve.py --checkpoint <path_to_ckpt> [--device cuda:0] [--host 0.0.0.0] [--port 8000]

REST API:
    POST /predict   {"obs": [[36 floats] * n_obs_steps]}
                    → {"action": [[12 floats] * n_action_steps]}
    GET  /health    → {"status": "ok", "obs_dim": 36, "action_dim": 12, ...}
"""

import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import pathlib
import click
import dill
import hydra
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from omegaconf import OmegaConf
from pydantic import BaseModel
from typing import List

from diffusion_policy.workspace.base_workspace import BaseWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)

app = FastAPI(title="DiffuseLoco Inference Server")

# Global policy — populated at startup
policy = None
cfg = None


# ---------- request / response schemas ----------

class PredictRequest(BaseModel):
    obs: List[List[float]]  # shape: (n_obs_steps, obs_dim)

class PredictResponse(BaseModel):
    action: List[List[float]]  # shape: (n_action_steps, action_dim)


# ---------- endpoints ----------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "obs_dim": policy.obs_dim,
        "action_dim": policy.action_dim,
        "n_obs_steps": policy.n_obs_steps,
        "n_action_steps": policy.n_action_steps,
        "horizon": policy.horizon,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    obs = np.array(req.obs, dtype=np.float32)  # (n_obs_steps, obs_dim)

    if obs.ndim != 2:
        raise HTTPException(status_code=422, detail="obs must be a 2-D array")
    if obs.shape[0] != policy.n_obs_steps:
        raise HTTPException(
            status_code=422,
            detail=f"obs must have {policy.n_obs_steps} timesteps, got {obs.shape[0]}"
        )
    if obs.shape[1] != policy.obs_dim:
        raise HTTPException(
            status_code=422,
            detail=f"obs_dim must be {policy.obs_dim}, got {obs.shape[1]}"
        )

    # add batch dim → (1, n_obs_steps, obs_dim)
    obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(policy.device)

    with torch.no_grad():
        result = policy.predict_action({"obs": obs_tensor})

    # (1, n_action_steps, action_dim) → list
    action = result["action"].squeeze(0).cpu().numpy().tolist()
    return PredictResponse(action=action)


# ---------- CLI ----------

@click.command()
@click.option("-c", "--checkpoint", required=True, help="Path to .ckpt file")
@click.option("-d", "--device", default="cuda:0")
@click.option("--host", default="0.0.0.0")
@click.option("--port", default=8000)
def main(checkpoint, device, host, port):
    global policy, cfg

    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    OmegaConf.set_struct(cfg, False)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg, output_dir=str(pathlib.Path(checkpoint).parent))
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    policy.to(torch.device(device))
    policy.eval()

    print(f"Loaded policy from {checkpoint}")
    print(f"  obs_dim={policy.obs_dim}, action_dim={policy.action_dim}")
    print(f"  n_obs_steps={policy.n_obs_steps}, n_action_steps={policy.n_action_steps}")
    print(f"Serving on http://{host}:{port}")

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
