import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from os import path

# Define paths
PROJECT_ROOT = path.abspath(path.dirname(path.dirname(__file__)))
DATA_DIR = path.join(PROJECT_ROOT, "Datasets")
CLUSTERED_FILE = path.join(DATA_DIR, "plant_growth_clustered.csv")

# Load clustered dataset
df = pd.read_csv(CLUSTERED_FILE)

# Extract actual labels and predicted clusters
true_labels = df["Growth_Milestone"].values  # Assuming "Plant_Type" is the actual class column
predicted_clusters = df["Cluster"].values  # K-Means assigned clusters

# Map clusters to actual labels using majority voting
cluster_to_label = {}
for cluster in np.unique(predicted_clusters):
    cluster_points = df[df["Cluster"] == cluster]
    most_common_label = cluster_points["Growth_Milestone"].mode()[0]  # Most frequent label in this cluster
    cluster_to_label[cluster] = most_common_label

# Assign new predicted labels based on the mapping
mapped_predictions = np.array([cluster_to_label[c] for c in predicted_clusters])

# Generate confusion matrix
conf_matrix = confusion_matrix(true_labels, mapped_predictions)

# Compute precision, recall, and F1-score
precision = precision_score(true_labels, mapped_predictions, average="weighted")
recall = recall_score(true_labels, mapped_predictions, average="weighted")
f1 = f1_score(true_labels, mapped_predictions, average="weighted")

# Display results
print("\nConfusion Matrix:")
print(conf_matrix)

print(f"\nPrecision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")

# Plot confusion matrix
plt.figure(figsize=(6, 5))
plt.imshow(conf_matrix, cmap="Blues", interpolation="nearest")
plt.colorbar()
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()
