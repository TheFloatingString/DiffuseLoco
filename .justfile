set dotenv-load := true

# read WANDB API KEY from .env
# MAKE sure that there is a .env file in the same directory as the .justfile
WANDB_KEY_VALUE := env("LAURENCE_WANDB_API_KEY", "")
WANDB_ENTITY_VALUE := env("WANDB_ENTITY", "")

generate_standup_data:
    python ./scripts/eval.py --checkpoint=./cyberdog_final.ckpt --task=cyber2_stand --online=false --generate_data=true

generate_trot_data:
    python ./scripts/eval.py --checkpoint=./cyberdog_final.ckpt --task=cyber2_trot --online=true --generate_data=true

train_grand_tour:
    WANDB_API_KEY={{WANDB_KEY_VALUE}} WANDB_ENTITY={{WANDB_ENTITY_VALUE}} python ./scripts/train.py --ds grand_tour

train_isaaclab:
    #!/bin/bash
    conda activate diffuseloco
    WANDB_API_KEY={{WANDB_KEY_VALUE}} WANDB_ENTITY={{WANDB_ENTITY_VALUE}} python ./scripts/train.py --ds isaaclab