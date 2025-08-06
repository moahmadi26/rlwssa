#!/usr/bin/env python3
"""
Script to plot probability estimates from CSV files with shaded error regions.
Reads CSV files from the results directory and saves plots as PNG files.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import matplotlib as mpl

# Set default font sizes
mpl.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 24,
    'axes.labelsize': 22,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'figure.titlesize': 26
})

# True probability values for different models
TRUE_PROBABILITIES = {
    'motil': 2.20E-7,
    'enzym': 1.71E-7,
    'yeast': 1.202E-6,
    'circuit': 6.292E-4
}


def plot_probability_estimates(csv_file, output_dir=None):
    """
    Plot probability estimates with shaded error regions from a CSV file.
    
    Parameters:
    -----------
    csv_file : str or Path
        Path to the CSV file containing the data
    output_dir : str or Path, optional
        Directory to save the plot. If None, saves in the same directory as the script
    
    Returns:
    --------
    str : Path to the saved plot file
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Extract data
    simulations = df['simulations']
    probability_estimate = df['probability_estimate']
    error = df['error']
    
    # Calculate upper and lower bounds for error region
    upper_bound = probability_estimate + error
    lower_bound = probability_estimate - error
    
    # Create the plot
    plt.figure(figsize=(10, 7))
    
    # Plot the probability estimate line
    plt.plot(simulations, probability_estimate, 'b-', linewidth=2, label='Probability Estimate')
    
    # Fill the error region in grey
    plt.fill_between(simulations, lower_bound, upper_bound, 
                     alpha=0.3, color='grey', label='Error')
    
    # Add true probability line if available
    filename = Path(csv_file).stem
    for model_name, true_prob in TRUE_PROBABILITIES.items():
        if model_name in filename:
            plt.axhline(y=true_prob, color='red', linestyle='--', linewidth=2, 
                       label='True Probability')
            break
    
    # Customize the plot
    plt.xlabel('Number of Simulations')
    plt.ylabel('Probability Estimate')
    # No title
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    
    # Use scientific notation for y-axis if values are very small
    if probability_estimate.max() < 0.01:
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    # Set x-axis to show actual simulation counts
    plt.ticklabel_format(style='plain', axis='x')
    
    # Tight layout
    plt.tight_layout()
    
    # Save the plot
    if output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{Path(csv_file).stem}_plot.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(output_file)


def plot_all_results(results_dir, output_dir=None):
    """
    Plot all CSV files in the results directory.
    
    Parameters:
    -----------
    results_dir : str or Path
        Directory containing CSV files
    output_dir : str or Path, optional
        Directory to save plots. If None, saves in the plot directory
    """
    results_dir = Path(results_dir)
    
    # Find all CSV files
    csv_files = list(results_dir.glob('*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {results_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files to plot")
    
    # Plot each file
    for csv_file in csv_files:
        try:
            output_file = plot_probability_estimates(csv_file, output_dir)
            print(f"✓ Plotted {csv_file.name} -> {output_file}")
        except Exception as e:
            print(f"✗ Error plotting {csv_file.name}: {str(e)}")


def plot_comparison(csv_files, output_file=None, labels=None):
    """
    Plot multiple CSV files on the same graph for comparison.
    
    Parameters:
    -----------
    csv_files : list of str or Path
        List of CSV files to compare
    output_file : str or Path, optional
        Output file name. If None, saves as 'comparison_plot.png'
    labels : list of str, optional
        Labels for each plot. If None, uses file names
    """
    plt.figure(figsize=(12, 8))
    
    # Default colors for multiple plots
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'olive', 'cyan']
    
    # Track which true probabilities to add
    models_to_plot = set()
    
    for i, csv_file in enumerate(csv_files):
        df = pd.read_csv(csv_file)
        
        simulations = df['simulations']
        probability_estimate = df['probability_estimate']
        error = df['error']
        
        # Calculate bounds
        upper_bound = probability_estimate + error
        lower_bound = probability_estimate - error
        
        # Use provided label or file name
        label = labels[i] if labels and i < len(labels) else Path(csv_file).stem
        color = colors[i % len(colors)]
        
        # Plot line and error region
        plt.plot(simulations, probability_estimate, color=color, linewidth=2, label=label)
        plt.fill_between(simulations, lower_bound, upper_bound, 
                         alpha=0.2, color='grey')
        
        # Check which model this is
        filename = Path(csv_file).stem
        for model_name in TRUE_PROBABILITIES:
            if model_name in filename:
                models_to_plot.add(model_name)
    
    # Add true probability lines
    for model_name in models_to_plot:
        true_prob = TRUE_PROBABILITIES[model_name]
        plt.axhline(y=true_prob, color='red', linestyle='--', linewidth=2, 
                   label=f'True {model_name}')
    
    # Customize the plot
    plt.xlabel('Number of Simulations')
    plt.ylabel('Probability Estimate')
    # No title
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    
    # Use scientific notation for y-axis if needed
    ax = plt.gca()
    y_max = max([pd.read_csv(f)['probability_estimate'].max() for f in csv_files])
    if y_max < 0.01:
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    plt.ticklabel_format(style='plain', axis='x')
    plt.tight_layout()
    
    # Save the plot
    if output_file is None:
        output_file = Path(__file__).parent / 'comparison_plot.png'
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(output_file)


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description='Plot probability estimates from CSV files with error regions'
    )
    
    parser.add_argument(
        'action',
        choices=['single', 'all', 'compare'],
        help='Action to perform: plot single file, all files, or compare multiple files'
    )
    
    parser.add_argument(
        '--input', '-i',
        help='Input CSV file (for single) or directory (for all)'
    )
    
    parser.add_argument(
        '--files', '-f',
        nargs='+',
        help='List of CSV files to compare (for compare action)'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output directory or file'
    )
    
    parser.add_argument(
        '--labels', '-l',
        nargs='+',
        help='Labels for comparison plots'
    )
    
    args = parser.parse_args()
    
    # Handle different actions
    if args.action == 'single':
        if not args.input:
            print("Error: --input required for single file plotting")
            sys.exit(1)
        
        output_file = plot_probability_estimates(args.input, args.output)
        print(f"Plot saved to: {output_file}")
    
    elif args.action == 'all':
        # Default to results directory if not specified
        input_dir = args.input or '../results'
        plot_all_results(input_dir, args.output)
    
    elif args.action == 'compare':
        if not args.files:
            print("Error: --files required for comparison plotting")
            sys.exit(1)
        
        output_file = plot_comparison(args.files, args.output, args.labels)
        print(f"Comparison plot saved to: {output_file}")


if __name__ == '__main__':
    # If no arguments provided, plot all results in the default directory
    if len(sys.argv) == 1:
        print("No arguments provided. Plotting all CSV files in ../results/")
        plot_all_results('../results')
    else:
        main()
