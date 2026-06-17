set dotenv-load := true

# read WANDB API KEY from .env
# MAKE sure that there is a .env file in the same directory as the .justfile
WANDB_KEY_VALUE := env("LAURENCE_WANDB_API_KEY", "")
WANDB_ENTITY_VALUE := env("WANDB_ENTITY", "")

generate_standup_data:
    python ./scripts/eval.py --checkpoint=./cyberdog_final.ckpt --task=cyber2_stand --online=false --generate_data=true

generate_trot_data:
    python ./scripts/eval.py --checkpoint=./cyberdog_final.ckpt --task=cyber2_trot --online=true --generate_data=true

train_grand_tour offset="0" frequency="30":
    WANDB_API_KEY={{WANDB_KEY_VALUE}} WANDB_ENTITY={{WANDB_ENTITY_VALUE}} python ./scripts/train.py --ds grand_tour --offset {{offset}} --frequency {{frequency}}

train_grand_tour_with_upsampling_50_hz offset="0":
    WANDB_API_KEY={{WANDB_KEY_VALUE}} WANDB_ENTITY={{WANDB_ENTITY_VALUE}} python ./scripts/train.py --ds grand_tour --upsample_factor 5 --frequency 50 --offset {{offset}}

train_grand_tour_goal_conditioning offset="0":
    WANDB_API_KEY={{WANDB_KEY_VALUE}} WANDB_ENTITY={{WANDB_ENTITY_VALUE}} python ./scripts/train.py --ds grand_tour --grand_tour_goal_cond --frequency 50 --offset {{offset}}

train_isaaclab offset="0":
    WANDB_API_KEY={{WANDB_KEY_VALUE}} WANDB_ENTITY={{WANDB_ENTITY_VALUE}} python ./scripts/train.py --ds isaaclab --offset {{offset}}