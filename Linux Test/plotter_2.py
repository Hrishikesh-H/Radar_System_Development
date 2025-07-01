import pandas as pd
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='CSV file')
parser.add_argument('--file', default="radar_data.csv", help='CSV file with original and despiked radar data')
args = parser.parse_args()

# Load the CSV
df = pd.read_csv(args.file)

# Line plots for each axis
axes = ['x', 'y', 'z']
fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

for i, axis in enumerate(axes):
    orig_col = f"original_{axis}"
    desp_col = f"despiked_{axis}"
    
    axs[i].plot(df[orig_col], label='Original', color='red', alpha=0.6)
    axs[i].plot(df[desp_col], label='Despiked', color='green', alpha=0.6)
    axs[i].set_ylabel(f'{axis.upper()} (m)', fontsize=12)
    axs[i].legend()
    axs[i].grid(True)

axs[2].set_xlabel('Index', fontsize=12)
fig.suptitle('Line Plot: Original vs Despiked Radar Data (X, Y, Z)', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
