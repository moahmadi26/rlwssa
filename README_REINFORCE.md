# REINFORCE for Weighted SSA

This repository implements a REINFORCE algorithm for controlling stochastic chemical reaction networks using weighted SSA (dwSSA).

## Features

- **Combined Reward Structure**: Progress rewards + importance weight penalties + success bonuses
- **Parallel Training & Evaluation**: Fully parallelized across multiple CPUs
- **Adaptive Exploration**: Exploration decreases as states are visited more frequently
- **Fine-grained State Discretization**: Adaptive binning based on distance to target

## Usage

1. Activate the virtual environment:
```bash
source venv_address.sh
```

2. Run the algorithm:
```bash
cd src
python main.py ../crns/<model_folder>/<config_file>.json
```

## Configuration

All models are configured via JSON files with the following structure:
```json
{
    "model_path": "../crns/model_folder/model.sm",
    "target_variable": "species_name",
    "target_value": "100",
    "max_time": "10.0",
    "algorithm": "reinforce_combined"
}
```

## Reward Components

1. **Progress Reward** (0-40 points): Rewards getting closer to the target
2. **Weight Penalty** (-10 to 0 points): Penalizes high-variance trajectories
3. **Success Bonus** (100-200 points): Large reward for reaching the target with efficiency bonus

## Files

- `main.py`: Entry point and experiment runner
- `reinforce_dwssa_combined.py`: Core REINFORCE algorithm with combined rewards
- `reinforce_parallel_combined.py`: Parallel implementation for training and evaluation
- `prism_parser.py`: Parser for PRISM model files
- `suppress.py`: Utility for suppressing C output

## Hyperparameters

Configured in `main.py`:
- `N_train`: Number of training episodes (default: 50,000)
- `batch_size`: Batch size for policy updates (default: 100)
- `K`: Number of independent runs for evaluation (default: 4)
- `N`: Episodes per evaluation run (default: 20,000)