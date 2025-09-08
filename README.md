# Rare Event Simulation for Stochastic Reaction Networks

This repository implements three advanced importance sampling methods for estimating rare event probabilities in stochastic reaction networks.

## Methods

### 1. **dwSSA** - Cross-entropy method for importance sampling
### 2. **dwSSA++** - Enhanced dwSSA with polynomial leaping acceleration  
### 3. **Learning-based IS** - Gradient-based optimization with sigmoid ansatz

## Repository Structure

```
rlwssa/
├── src/                           # Source code and main scripts
└── models/                       # Biochemical reaction network models
```

## How to Use

### Prerequisites
- Python 3.7+ with NumPy
- Stormpy (for PRISM model parsing)

### Running the Methods

Each method takes a JSON configuration file that specifies:
- `model_path`: Path to the .sm PRISM model file
- `target_variable`: Which species to monitor (e.g., "X", "s5")  
- `target_value`: Threshold value for the rare event
- `max_time`: Simulation time horizon
- `model_name`: Descriptive name

Navigate to `src/` directory and run:

```bash
# dwSSA (cross-entropy method)
python run_dwssa.py ../models/enzymatic_futile_cycle/low_s5.json

# dwSSA++ (with polynomial leaping)
python run_dwssa_plus_plus.py ../models/enzymatic_futile_cycle/low_s5.json

# Learning-based importance sampling
python run_learning_is.py ../models/enzymatic_futile_cycle/low_s5.json
```

### Creating Your Own Models

1. **Create a PRISM model file** (`.sm`) defining your reaction network:
```prism
ctmc
const double k1 = 1.0;
module model_name
    X : int init 100;
    [reaction] X>0 -> (k1*X) : (X'=X-1);
endmodule
```

2. **Create a JSON configuration file** specifying simulation parameters:
```json
{
    "model_path": "path/to/your_model.sm",
    "target_variable": "X", 
    "target_value": "50",
    "max_time": "1.0",
    "model_name": "Your Model Name"
}
```

3. **Run your simulation** with any of the three methods above.

## Output

Each method provides:
- **Probability estimate** with standard error
- **Variance reduction** compared to standard simulation
- **Convergence information** and computational performance
- **Results saved** to JSON/text files for analysis

