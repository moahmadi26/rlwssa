# Plotting Tools for RLWSSA Results

This directory contains tools for visualizing probability estimates from the RLWSSA simulation results.

## Features

- Plot probability estimates with shaded error regions
- Support for single file, batch, and comparison plotting
- Automatic scientific notation for small probabilities
- High-resolution PNG output (300 DPI)

## Usage

### As a Python Module

```python
from plot_results import plot_probability_estimates, plot_all_results, plot_comparison

# Plot a single CSV file
plot_probability_estimates("../results/circuit.csv")

# Plot all CSV files in a directory
plot_all_results("../results")

# Compare multiple CSV files
plot_comparison(
    ["../results/circuit.csv", "../results/yeast.csv"],
    labels=["Circuit", "Yeast"]
)
```

### Command Line Interface

```bash
# Plot a single file
python plot_results.py single -i ../results/circuit.csv

# Plot all files in a directory
python plot_results.py all -i ../results

# Compare multiple files
python plot_results.py compare -f ../results/circuit.csv ../results/yeast.csv -l Circuit Yeast

# Get help
python plot_results.py -h
```

### Running Without Arguments

If you run the script without any arguments, it will automatically plot all CSV files in the `../results` directory:

```bash
python plot_results.py
```

## CSV Format

The script expects CSV files with the following columns:
- `simulations`: Number of simulations
- `probability_estimate`: Estimated probability value
- `variance`: Variance of the estimate
- `error`: Error margin

## Output

- Individual plots are saved as `{filename}_plot.png`
- Comparison plots are saved as `comparison_plot.png` by default
- All plots are saved with 300 DPI resolution for publication quality

## Examples

See `example_usage.py` for detailed examples of how to use the plotting functions.