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
parser = argparse.ArgumentParser(description='Plot latency comparison from CSV file')
parser.add_argument('csv_file', type=str, help='Path to the CSV file containing benchmark results')
args = parser.parse_args()

# Read CSV file
df = pd.read_csv(args.csv_file)

# Filter and group data by operation and size, then calculate average throughput
reads_data = df[df['operation'] == 0].groupby('size')['avg_throughput'].mean().reset_index()
writes_data = df[df['operation'] == 1].groupby('size')['avg_throughput'].mean().reset_index()

# Extract x and y values
x_vals = sorted(reads_data['size'].unique().tolist() + writes_data['size'].unique().tolist())
x_vals = list(set(x_vals))              # Remove duplicates
x_vals.sort()
x_vals_kb = [x / 1024 for x in x_vals]  # Convert to KiB for plotting

# Create dictionaries for easy lookup
reads_dict = dict(zip(reads_data['size'], reads_data['avg_throughput']))
writes_dict = dict(zip(writes_data['size'], writes_data['avg_throughput']))

# Get values for each size (use NaN if not present)
values_reads = [reads_dict.get(size, np.nan) for size in x_vals]
values_writes = [writes_dict.get(size, np.nan) for size in x_vals]

# Plot
fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(5, 5))
axs.plot(x_vals_kb, values_reads, marker='o', color='#085cd8', label='Reads')
axs.plot(x_vals_kb, values_writes, marker='x', color='#e67c2f', label='Writes')
axs.set_xlabel('Message Size (KiB)')
axs.set_ylabel('Throughput (GBps)')
axs.set_xscale('log', base=2)
axs.set_xticks(x_vals_kb[::2])
axs.set_xlim(196 / 1024, 64 * 1024 * 1.5)
axs.set_ylim(0, 12)
axs.grid(True, axis='y')
axs.set_title('P2P reads and writes throughput comparison', fontsize=10)
axs.legend(loc='upper left', ncol=2)

# Create figures directory one level above script directory
script_dir = Path(__file__).parent
figures_dir = script_dir.parent / 'figures'
figures_dir.mkdir(exist_ok=True)
fig.savefig(figures_dir / 'p2p-read-writes-comparison.pdf', bbox_inches='tight', pad_inches=.03)
fig.savefig(figures_dir / 'p2p-read-writes-comparison.png', bbox_inches='tight', pad_inches=.03)
