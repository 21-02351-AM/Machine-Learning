import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import path

# Define paths
PROJECT_ROOT = path.abspath(path.dirname(path.dirname(__file__)))
DATA_DIR = path.join(PROJECT_ROOT, "Datasets")
CLUSTERED_FILE = path.join(DATA_DIR, "plant_growth_clustered.csv")

# Load clustered dataset
df = pd.read_csv(CLUSTERED_FILE)

# Extract actual labels and predicted clusters
true_labels = df["Growth_Milestone"].values  # Assuming "Growth_Milestone" is the actual class column
predicted_clusters = df["Cluster"].values  # K-Means assigned clusters

# Map clusters to actual labels using majority voting
cluster_to_label = {}
for cluster in np.unique(predicted_clusters):
    cluster_points = df[df["Cluster"] == cluster]
    most_common_label = cluster_points["Growth_Milestone"].mode()[0]  # Most frequent label in this cluster
    cluster_to_label[cluster] = most_common_label

# Assign new predicted labels based on the mapping
mapped_predictions = np.array([cluster_to_label[c] for c in predicted_clusters])

# Create confusion matrix manually
unique_labels = np.unique(true_labels)  # Find unique classes (0 and 1)
conf_matrix = np.zeros((len(unique_labels), len(unique_labels)), dtype=int)

# Populate the confusion matrix
for true, pred in zip(true_labels, mapped_predictions):
    conf_matrix[true, pred] += 1

# Extract values from the confusion matrix
TP = conf_matrix[1, 1]  # True Positives
TN = conf_matrix[0, 0]  # True Negatives
FP = conf_matrix[0, 1]  # False Positives
FN = conf_matrix[1, 0]  # False Negatives

# Compute precision, recall, and F1-score manually
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Display results
print("\nConfusion Matrix:")
print(conf_matrix)

print(f"\nPrecision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1_score:.2f}")

# Plot confusion matrix
plt.figure(figsize=(6, 5))
plt.imshow(conf_matrix, cmap="Blues", interpolation="nearest")
plt.colorbar()
plt.xticks([0, 1], labels=["Not Reached (0)", "Reached (1)"])
plt.yticks([0, 1], labels=["Not Reached (0)", "Reached (1)"])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()
