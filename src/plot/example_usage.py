#!/usr/bin/env python3
"""
Example usage of the plot_results module.
"""

from plot_results import plot_probability_estimates, plot_all_results, plot_comparison
from pathlib import Path

# Example 1: Plot a single CSV file
print("Example 1: Plotting a single CSV file")
single_file = Path("../results/circuit.csv")
if single_file.exists():
    output = plot_probability_estimates(single_file)
    print(f"Single plot saved to: {output}")
else:
    print(f"File {single_file} not found")

print("\n" + "="*50 + "\n")

# Example 2: Plot all CSV files in results directory
print("Example 2: Plotting all CSV files in results directory")
results_dir = Path("../results")
if results_dir.exists():
    plot_all_results(results_dir)
else:
    print(f"Directory {results_dir} not found")

print("\n" + "="*50 + "\n")

# Example 3: Compare multiple CSV files
print("Example 3: Comparing multiple CSV files")
comparison_files = [
    "../results/circuit.csv",
    "../results/circuit_updated.csv"
]

# Check which files exist
existing_files = [f for f in comparison_files if Path(f).exists()]

if len(existing_files) >= 2:
    output = plot_comparison(
        existing_files[:2],
        output_file="circuit_comparison.png",
        labels=["Original", "Updated"]
    )
    print(f"Comparison plot saved to: {output}")
else:
    print("Not enough files found for comparison")

print("\n" + "="*50 + "\n")

# Example 4: Command line usage examples
print("Example 4: Command line usage")
print("\nTo plot a single file:")
print("  python plot_results.py single -i ../results/circuit.csv")

print("\nTo plot all files in a directory:")
print("  python plot_results.py all -i ../results")

print("\nTo compare multiple files:")
print("  python plot_results.py compare -f ../results/circuit.csv ../results/yeast.csv -l Circuit Yeast")

print("\nFor help:")
print("  python plot_results.py -h")