# File map

| File | Responsibility |
|---|---|
| `ppo_core.py` | Shared actor-critic, checkpoint loading, deterministic evaluation, Pendulum reward |
| `train_ppo.py` | Real-environment PPO training |
| `evaluate_ppo.py` | Deterministic PPO and random-policy evaluation |
| `plot_ppo_results.py` | Multi-seed learning curve and evaluation plot |
| `collect_transitions.py` | Policy/random/OOD transition collection |
| `world_model.py` | Normalized five-member dynamics ensemble |
| `train_world_model.py` | Bootstrap training, validation, uncertainty calibration |
| `evaluate_world_model.py` | One-step and multi-step error analysis |
| `imagined_env.py` | Ensemble-backed imagined vector environment |
| `train_imagined_ppo.py` | Fixed, uncertainty-terminated, or weighted imagined PPO |
| `compare_imagined_methods.py` | Matched model-versus-real return and exploitation gap |
| `aggregate_comparison.py` | Seed-level mean±std aggregation of the comparison + error-bar plots |
| `run_pipeline.sh` | End-to-end command sequence |
