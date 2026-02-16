import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

### Check if all five csv files exist, otherwise terminate
required_files = ['size.csv', 'baseline_lat.csv', 'p2p_lat.csv', 'baseline_thr.csv', 'p2p_thr.csv']
for file in required_files:
    if not os.path.exists(file):
        print(f"Error: {file} not found.")
        exit(1)

### LATENCY PLOT

# Load data from CSV
df1 = pd.read_csv('size.csv')
df2 = pd.read_csv('baseline_lat.csv')
df3 = pd.read_csv('p2p_lat.csv')

# Merge dataframes to a single object
df = pd.concat([df1, df2, df3], axis=1)
df.columns = df.columns.str.strip()

# Plotting
plt.figure(figsize=(10, 7))

x_labels = df['Size'].tolist()          # actual x-values (e.g., [64, 128, 256, ...])
# Remove last element from the x_lables
x_pos = np.arange(len(x_labels))   # equidistant positions [0, 1, 2, 3, ...]

print(x_labels)

# Set the bar width
bar_width = 0.25

# Define series metadata: (label, color, column suffix, marker, hatch)
series_meta = [
    ('Baseline: Copies from host memory to GPUs', 'skyblue', 'baseline', 'o', '.'),
    ('FPGA-initiated P2P to GPUs', 'red', 'p2p', 'x', '\\'),
]

# Plot each series
for label, color, suffix, marker, hatch in series_meta:
    avg = df[f'avg_{suffix}']/1000
    x = x_pos

    # Sanity check
    for i in range(len(avg)):

        # Calculate x positions per bar 
        bar_offset = 0
        if(label == 'Baseline: Copies from host memory to GPUs'):
            bar_offset = -(bar_width)
        elif(label == 'FPGA-initiated P2P to GPUs'):
            bar_offset = 0

        bar_positions = x + bar_offset

    # Draw bars 
    bars = plt.bar(bar_positions, avg, width=bar_width, label=label, color=color, edgecolor='black', zorder=3)




# Format the plot
plt.xticks(ticks=x_pos, labels=[r'$2^{'+str(int(i))+'}$' if not np.isnan(i) else '' for i in x_labels])  # Show explicit x entries
plt.xlabel('Size of the transmitted buffer [Byte]', fontsize=22)
plt.ylabel('Latency [us]', fontsize=22)
plt.yscale("log")
plt.legend(fontsize=18, title = 'RDMA READ SCATTER TO GPU-BUFFERS', title_fontsize=20, loc='upper left')
plt.grid(True, zorder=0)
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
plt.tight_layout()

# Increase tick label font size
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

# Increase legend font size
# plt.show()
plt.savefig('P2P_SCATTER_LAT.pdf', dpi=900, bbox_inches='tight')


### THROUGHPUT PLOT

df2 = pd.read_csv('baseline_thr.csv')
df3 = pd.read_csv('p2p_thr.csv')

# Merge dataframes to a single object
df = pd.concat([df1, df2, df3], axis=1)
df.columns = df.columns.str.strip()

# Plotting
plt.figure(figsize=(10, 7))

x_labels = df['Size'].tolist()          # actual x-values (e.g., [64, 128, 256, ...])
x_pos = list(range(len(x_labels)))   # equidistant positions [0, 1, 2, 3, ...]


# Define series metadata: (label, color, column suffix, marker,)
series_meta = [
    ('Baseline: Copies from host memory to GPUs', 'skyblue', 'baseline', 'o'),
    ('FPGA-initiated P2P to GPUs', 'red', 'p2p', 'x'),
]

# Plot each series
for label, color, suffix, marker in series_meta:
    avg = df[f'avg_{suffix}']
    x = x_pos

    # Plot line with markers and error bars

    plt.errorbar(
        x, avg,
        fmt= marker + '-',              # circle markers + solid lines
        markersize=8,
        markerfacecolor='none',  # make circle hollow
        markeredgecolor=color,   # outline in your series col
        capsize=4,             # small line on error bar ends
        label=label,
        color=color, 
        zorder = 3
    )

# Format the plot
plt.xticks(ticks=x_pos, labels=[r'$2^{'+str(i)+'}$' for i in x_labels])  # Show explicit x entries
plt.xlabel('Size of the transmitted buffer [Byte]', fontsize=22)
plt.ylabel('Throughput [MB/s]', fontsize=22)
plt.legend(fontsize=17, title = 'RDMA READ SCATTER TO GPU-BUFFERS', title_fontsize=20, loc='center right')
plt.grid(True, zorder=0)
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
plt.tight_layout()

# Increase tick label font size
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

# Increase legend font size
# plt.show()
plt.savefig('P2P_SCATTER_THR.pdf', dpi=900, bbox_inches='tight')



