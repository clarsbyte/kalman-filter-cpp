#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

# Read sigma points
df = pd.read_csv('sigma_points.csv')
mean_points = df[df['type'] == 'mean']
sigma_points = df[df['type'] == 'sigma']

# Read covariance matrix
P = np.loadtxt('covariance.csv', delimiter=',')

# Compute ellipse
eigenvalues, eigenvectors = np.linalg.eig(P)
angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

fig, ax = plt.subplots(figsize=(10, 8))

# Plot mean
if not mean_points.empty:
    ax.plot(mean_points['x'].values[0], mean_points['y'].values[0],
            'go', markersize=12, label='Mean', zorder=5)

# Plot sigma points
ax.scatter(sigma_points['x'], sigma_points['y'],
          c='red', s=100, marker='x', linewidths=2, label='Sigma Points', zorder=4)

# Draw covariance ellipse (1-sigma)
ellipse1 = patches.Ellipse(
    xy=(mean_points['x'].values[0], mean_points['y'].values[0]),
    width=2*np.sqrt(eigenvalues[0]),
    height=2*np.sqrt(eigenvalues[1]),
    angle=angle,
    facecolor='blue',
    alpha=0.2,
    edgecolor='blue',
    linewidth=2,
    label='Covariance (1-σ)'
)
ax.add_patch(ellipse1)

# Draw 2-sigma ellipse
ellipse2 = patches.Ellipse(
    xy=(mean_points['x'].values[0], mean_points['y'].values[0]),
    width=4*np.sqrt(eigenvalues[0]),
    height=4*np.sqrt(eigenvalues[1]),
    angle=angle,
    facecolor='none',
    edgecolor='blue',
    linewidth=1,
    linestyle='--',
    label='Covariance (2-σ)',
    alpha=0.5
)
ax.add_patch(ellipse2)

ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Sigma Points and Covariance Ellipse')

plt.tight_layout()
plt.savefig('sigma_points_plot.png', dpi=150, bbox_inches='tight')
print("Plot saved to sigma_points_plot.png")
plt.show()
