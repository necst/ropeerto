import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Colour codes
# Blue #085cd8
# Green #39bc47
# Grey #6a6b6e
# Yellow #e5c255
# Orange #e67c2f
# Purple #6a2eb5
# Black #000000

# Plot formating
plt.rcParams.update({
  'font.size'        : 10,
  'font.family'      : ['Helvetica Neue'],
  'figure.figsize'   : (5,2.075),
  'axes.linewidth'   : 1,
  'xtick.direction'  : 'in',
  'xtick.major.size' : '2',
  'xtick.major.width': '1',
  'ytick.right'      : True,
  'ytick.direction'  : 'in',
  'ytick.major.size' : '2',
  'ytick.major.width': '1',
  'grid.linewidth'   : '1',
  'grid.color'       : 'black',
  'grid.linestyle'   : ':',
  'legend.fancybox'  : False,
  'legend.framealpha': 1,
  'legend.edgecolor' : 'black',
  'axes.autolimit_mode': 'round_numbers',
  'lines.linewidth'  : 1,
  'lines.markersize' : 5,
})

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Plot throughput scaling from CSV file')
parser.add_argument('csv_file', type=str, help='Path to the CSV file containing benchmark results')
args = parser.parse_args()

# Read CSV file
df = pd.read_csv(args.csv_file)

# Extract unique transfer counts (N values) and sort them
n_values = sorted(df['transfers'].unique())

# Initialize data structure to store values for each N and mode
values_p2p = []
values_base = []

for n in n_values:
    # Filter data for this N value
    n_data = df[df['transfers'] == n]
    
    # Separate by mode
    p2p_data = n_data[n_data['mode'] == 1].sort_values('size')
    base_data = n_data[n_data['mode'] == 0].sort_values('size')
    
    # Get sizes and throughputs
    p2p_sizes = p2p_data['size'].values
    p2p_throughputs = p2p_data['avg_throughput'].values
    
    base_sizes = base_data['size'].values
    base_throughputs = base_data['avg_throughput'].values
    
    values_p2p.append(p2p_throughputs.tolist())
    values_base.append(base_throughputs.tolist())

# Get x values from the first dataset (assuming all have same sizes) & convert to KiB
if len(values_p2p) > 0 and len(p2p_data) > 0:
    x_vals = p2p_data['size'].values
    x_vals_kb = [x / 1024 for x in x_vals]
else:
    x_vals_kb = []

fig, axs = plt.subplots(2, 2, constrained_layout=True, figsize=(5, 5), sharex=True, sharey=True)

# Plot sub-plots for varying N
for i in range(2):
    for j in range(2):
        ax = axs[i, j]
        ax.plot(x_vals_kb, values_p2p[i * 2 + j], marker='o', color='#085cd8', label='P2P')
        ax.plot(x_vals_kb, values_base[i * 2 + j], marker='x', color='#e67c2f', label='Baseline')
        ax.axhline(y=12.5, color='#ff0000', linestyle='-', label='CPU - FPGA max. bandwidth')
        ax.set_xscale('log', base=2)
        ax.set_xticks(x_vals_kb[::2])
        ax.set_xlim(0.125, 64 * 1024 * 1.5)
        ax.set_ylim(0, 13.5)
        ax.grid(True, axis='y')
        ax.set_title(f'N = {n_values[i * 2 + j]}', fontsize=10)

# Plot title, legend, axis
fig.text(0.5, -0.05, 'Message Size (KiB)', ha='center')
fig.text(-0.05, 0.5, 'Throughput (GBps)', va='center', rotation='vertical')
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1.015), ncol=3)

# Save figure
script_dir = Path(__file__).parent
figures_dir = script_dir.parent / 'figures'
figures_dir.mkdir(exist_ok=True)
fig.savefig(figures_dir / 'p2p-throughput-scaling.pdf', bbox_inches='tight', pad_inches=.03)
fig.savefig(figures_dir / 'p2p-throughput-scaling.png', bbox_inches='tight', pad_inches=.03)
