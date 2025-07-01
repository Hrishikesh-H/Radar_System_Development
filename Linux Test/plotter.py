import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

parser = argparse.ArgumentParser(description='CSV file')
parser.add_argument('--file', default="radar_data.csv", help='Look for the CSV file you want to plot in org/desp format')
args = parser.parse_args()
# Load the radar data CSV
df = pd.read_csv(args.file)  # Make sure this file is in your working directory

# Extract original and despiked coordinates
orig_x = df["original_x"]
orig_y = df["original_y"]
orig_z = df["original_z"]

despiked_x = df["despiked_x"]
despiked_y = df["despiked_y"]
despiked_z = df["despiked_z"]

# Create a 3D scatter plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot original data points in red
ax.scatter(orig_x, orig_y, orig_z, c='red', label='Original', alpha=0.6)

# Plot despiked data points in green
ax.scatter(despiked_x, despiked_y, despiked_z, c='green', label='Despiked', alpha=0.6)

# Set axis labels and title
ax.set_title('3D Scatter Plot: Original vs Despiked Radar Points', fontsize=14)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')

# Add legend
ax.legend()

# Optimize layout and display
plt.tight_layout()
plt.show()
