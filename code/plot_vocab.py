import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Load SIFT features from the pickle file
with open('vocab_400.pkl', 'rb') as f:
    sift_features = pickle.load(f)

# Convert features to a NumPy array if not already
sift_features = np.array(sift_features)

# Print shape to ensure correct dimensions
print(f"Loaded SIFT features with shape: {sift_features.shape}")

# Perform t-SNE to reduce dimensions to 2D
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
sift_features_2d = tsne.fit_transform(sift_features)

# Plot the t-SNE results
plt.figure(figsize=(10, 7))
plt.scatter(sift_features_2d[:, 0], sift_features_2d[:, 1], s=5, alpha=0.7, c='blue')

# Add title and axis labels
plt.title("t-SNE Visualization of SIFT Features", fontsize=16)
plt.xlabel("t-SNE Dimension 1", fontsize=14)
plt.ylabel("t-SNE Dimension 2", fontsize=14)

# Optional: Add grid
plt.grid(True, linestyle='--', alpha=0.6)

# Save and show the plot
plt.savefig("tsne_sift_features.png", dpi=300)
plt.show()
