# PATD3 / R-PATD3 — DRL-Guided Rollout for Semiconductor Packaging Scheduling

Code for the methods proposed in *"A Deep Reinforcement Learning-Guided Rollout Approach for
Adaptable Scheduling of Multiplicity Flexible Job Shops in Semiconductor Packaging."*

| Method | What it is | Where it lives |
| --- | --- | --- |
| **PATD3** | Proxy action-based TD3. The actor emits a proxy action `â`; every feasible job–machine pair is scored by the inner product of its action features with `â`, and the highest-scoring pair is dispatched (eq. 10). Trained with TD3 (Algorithm 1). | `patd3_agent.py`, `episode.py`, `Simulator.py` |
| **R-PATD3** | Rollout guided by the trained PATD3 actor. Two-phase branching (Algorithm 2) selects candidate pairs; each candidate is evaluated by a full look-ahead under the actor, and the search advances along the best complete schedule found (Algorithm 3). | `Rollout.py` |

## Run

```bash
pip install -r requirements.txt
python run_RL.py
```

This trains PATD3 on SD1, selects the checkpoint with the lowest mean validation makespan, tests it
on SD1–SD5, and runs R-PATD3 on the same problems, five times with seeds 42–46. Makespans are
printed; checkpoints go to `saved_models/`. Python 3.8 or later; runs on CPU.

## Data

`data/` holds the 450 problem instances used in the experiments (SD, MD, LD; 5 datasets × 30
problems each), as tab-separated text. Each dataset's problems 1, 2–10 and 11–30 are the training,
validation and test problems; problems 11–30 carry processing times with ±30% variation.

| File | Content |
| --- | --- |
| `problem.txt` | one row per problem: production requirements, initial machine setups, machine-to-type assignment, processing times |
| `jobTypes.txt` | operation sequence of each job type |
| `operationTypes.txt` | machine type that can process each operation type |
| `setupTime.txt` | sequence-dependent setup times |
| `machineTypes.txt` | machine type names |

## Settings (`params` in `run_RL.py`)

| Key | Default | Symbol / meaning |
| --- | --- | --- |
| `train_group` / `test_group` | `'SD'` / `'SD'` | dataset group trained / tested on: `SD`, `MD` or `LD` (Table I) |
| `NUM_EPISODE` | `1000` | `N_E` |
| `MEMORY_SIZE` | `100000` | `\|B\|` |
| `batch_size` | `256` | `N_TR` |
| `gamma` | `0.99` | discount factor |
| `actor_lr` / `critic_lr` | `3e-4` / `3e-4` | Adam learning rates |
| `tau` | `0.005` | `ρ`, soft-update coefficient |
| `policy_freq` | `2` | `ψ`, delayed policy update |
| `expl_noise` | `0.1` | exploration noise |
| `policy_noise` / `noise_clip` | `0.2` / `0.5` | target policy smoothing noise and its clip |
| `TRAINING_START` | `25000` | warm-up steps with uniform random actions |
| `save_interval` | `10` | checkpoint interval (validation candidates) |
| `train_validation_test` | `[1, 9, 20]` | training / validation / test problems per dataset |
| `state_size` | `4` | state `(s^P, s^C, s^W, s^I)` in the order `get_state()` returns it, Section IV-A-1 |
| `action_feature` | `['ST','RO','RL','AS','MS']` | action features `(a^S, a^R, a^J, a^A, a^M)`, Section IV-A-2 |
| `branching_value` | `1.0` | `p_b` in Algorithm 2 (1.0 on SD, 0.05 on MD and LD in the paper) |
| `total_seconds` | `600` | rollout time limit per problem (seconds) |

`dataset` and `test_dataset` at the top of `run_RL.py` choose the datasets (1–5) within the groups.

## Files

| File | Role |
| --- | --- |
| `run_RL.py` | settings and the Train / Validation / Test / Rollout driver |
| `patd3_agent.py` | actor, twin critics, replay buffer, TD3 update (Algorithm 1) |
| `episode.py` | one PATD3 episode: state, proxy action, dispatch, reward, update |
| `Rollout.py` | R-PATD3: branching (Algorithm 2) and the rollout loop (Algorithm 3) |
| `Simulator.py` | scheduling environment: `get_state()`, `cal_action()` (eq. 10), `step()` |
| `factory_status.py` | shop-floor state: waiting jobs, idle machines, completion times, schedule |
| `problem.py` | problem data structures (`ProblemInstance`, `Lot`, `Machine`) |
| `read_dataset.py` | reads one problem instance from `data/` |
