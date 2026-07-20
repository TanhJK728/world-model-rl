#!/usr/bin/env bash
set -euo pipefail

# CPU is the conservative default on macOS. Use DEVICE=mps bash run_pipeline.sh to opt into Apple GPU.
DEVICE=${DEVICE:-cpu}

# macOS + conda: avoid the "libomp already initialized" crash (multiple OpenMP runtimes).
export KMP_DUPLICATE_LIB_OK=${KMP_DUPLICATE_LIB_OK:-TRUE}

# Phase 1: three real-environment PPO seeds.
# Config that actually solves Pendulum (~-150 deterministic return, optimal ~-130):
#   --normalize-reward 1  reward normalization on the training signal (obs untouched, pipeline-safe);
#                         the decisive fix -- without it PPO plateaus at ~-1000, barely above random.
#   --ent-coef 0.01       keep exploration (ent-coef 0 collapses to ~-1000).
#   --anneal-lr 0         constant LR beats annealing-to-0 here.
# Converges via a sharp phase transition ~400-500k steps; 1M gives margin (~90s/seed on CPU).
for seed in 0 1 2; do
  python train_ppo.py \
    --seed "$seed" \
    --total-timesteps 1000000 \
    --num-envs 8 \
    --num-steps 256 \
    --anneal-lr 0 \
    --ent-coef 0.01 \
    --normalize-reward 1 \
    --output-dir "runs/ppo_seed${seed}" \
    --device "$DEVICE"
done

python evaluate_ppo.py \
  --checkpoints \
    runs/ppo_seed0/checkpoint.pt \
    runs/ppo_seed1/checkpoint.pt \
    runs/ppo_seed2/checkpoint.pt \
  --episodes 30 \
  --output runs/ppo_summary/evaluation.csv \
  --device "$DEVICE"

python plot_ppo_results.py \
  --metrics \
    runs/ppo_seed0/metrics.csv \
    runs/ppo_seed1/metrics.csv \
    runs/ppo_seed2/metrics.csv \
  --evaluation runs/ppo_summary/evaluation.csv \
  --output-dir runs/ppo_summary

# Phase 2: train on policy data; reserve random and OOD for distribution-shift evaluation.
python collect_transitions.py \
  --mode policy \
  --checkpoint runs/ppo_seed0/checkpoint.pt \
  --steps 60000 \
  --output data/policy.npz \
  --device "$DEVICE"

python collect_transitions.py \
  --mode random \
  --steps 30000 \
  --output data/random.npz \
  --device "$DEVICE"

python collect_transitions.py \
  --mode ood \
  --steps 30000 \
  --output data/ood.npz \
  --device "$DEVICE"

python train_world_model.py \
  --datasets data/policy.npz \
  --ensemble-size 5 \
  --epochs 80 \
  --output-dir runs/world_model \
  --device "$DEVICE"

python evaluate_world_model.py \
  --checkpoint runs/world_model/world_model.pt \
  --datasets data/policy.npz data/random.npz data/ood.npz \
  --horizons 1,5,10,20 \
  --output-dir runs/world_model_eval \
  --device "$DEVICE"

# Phase 3: imagined-policy refinement ablations.
python train_imagined_ppo.py \
  --world-model runs/world_model/world_model.pt \
  --state-dataset data/policy.npz \
  --init-checkpoint runs/ppo_seed0/checkpoint.pt \
  --method fixed \
  --horizon 5 \
  --output-dir runs/imagined_fixed_h5 \
  --device "$DEVICE"

python train_imagined_ppo.py \
  --world-model runs/world_model/world_model.pt \
  --state-dataset data/policy.npz \
  --init-checkpoint runs/ppo_seed0/checkpoint.pt \
  --method fixed \
  --horizon 20 \
  --output-dir runs/imagined_fixed_h20 \
  --device "$DEVICE"

python train_imagined_ppo.py \
  --world-model runs/world_model/world_model.pt \
  --state-dataset data/policy.npz \
  --init-checkpoint runs/ppo_seed0/checkpoint.pt \
  --method uncertainty \
  --horizon 20 \
  --output-dir runs/imagined_uncertainty \
  --device "$DEVICE"

python train_imagined_ppo.py \
  --world-model runs/world_model/world_model.pt \
  --state-dataset data/policy.npz \
  --init-checkpoint runs/ppo_seed0/checkpoint.pt \
  --method weighted \
  --horizon 20 \
  --beta 1.0 \
  --output-dir runs/imagined_weighted \
  --device "$DEVICE"

# Phase 4: same-initial-state 200-step model-versus-real comparison.
python compare_imagined_methods.py \
  --world-model runs/world_model/world_model.pt \
  --state-dataset data/policy.npz \
  --checkpoints \
    runs/ppo_seed0/checkpoint.pt \
    runs/imagined_fixed_h5/checkpoint.pt \
    runs/imagined_fixed_h20/checkpoint.pt \
    runs/imagined_uncertainty/checkpoint.pt \
    runs/imagined_weighted/checkpoint.pt \
  --episodes 50 \
  --output-dir runs/comparison \
  --device "$DEVICE"
