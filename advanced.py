import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

plt.close('all')

def cohensd(x, y):
    mnx = np.mean(x)
    mny = np.mean(y)
    sdx = np.std(x)
    sdy = np.std(y)
    return (mnx - mny) / np.sqrt(((sdx**2) + (sdy**2)) / 2)

# Vowels to analyze
vowels = ['aa', 'ih', 'eh', 'ey', 'ow']

# Initialize data structure for storing results
results = np.zeros((len(vowels), len(vowels), 8, 25))  # Each vowel pair, per dialect, per layer

# Load data and compute Cohen's d
for d in range(1, 9):
    HR = {v: np.load(f'/Users/ameenqureshi/Desktop/487seg/HS_{d}_{v}.npy') for v in vowels}
    for (i, v1), (j, v2) in combinations(enumerate(vowels), 2):
        if i >= j:  # Ensure we only compute each combination once
            continue
        for token in range(20):  # Number of tokens
            hr1 = HR[v1][:,:,np.random.choice(HR[v1].shape[2], 100, replace=False)]
            hr2 = HR[v2][:,:,np.random.choice(HR[v2].shape[2], 100, replace=False)]
            for e in range(25):  # Encoder layers
                CD = cohensd(hr1[e, :, :], hr2[e, :, :])
                results[i, j, d-1, e] += (np.abs(CD) > 0.5)  # Increment if distinguishing

# Normalize results by number of tokens
results /= 20

# Plot results using heatmaps
fig, axes = plt.subplots(nrows=8, ncols=1, figsize=(10, 40))  # One column per dialect
for d in range(8):
    ax = axes[d]
    im = ax.imshow(results[:, :, d, :].sum(axis=0), cmap='hot', interpolation='nearest', aspect='auto')
    ax.set_title(f'Heatmap of Distinguishing Features for Dialect {d+1}')
    ax.set_xlabel('Encoder Layers')
    ax.set_ylabel('Vowel Pairs')
    ax.set_yticks(np.arange(len(vowels)))
    ax.set_yticklabels(vowels)
    fig.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()
