# Import necessary libraries
import pandas as pd
import random
import numpy as np
from os import path
from math import sqrt
import matplotlib.pyplot as plt

# ------------------------- Define Paths -------------------------
# Define project directory and dataset path
PROJECT_ROOT = path.abspath(path.dirname(path.dirname(__file__)))
DATA_DIR = path.join(PROJECT_ROOT, "Datasets")
DATA_FILE = path.join(DATA_DIR, "plant_growth_data.csv")

# ------------------------- Load Dataset -------------------------
# Load the CSV file into a pandas dataframe
df = pd.read_csv(DATA_FILE)

# ------------------------- Data Preprocessing -------------------------
# Encode categorical feature "Water_Frequency" (converting text labels to numerical values)
water_mapping = {"bi-weekly": 1, "weekly": 2, "daily": 3}
df["Water_Frequency_Encoded"] = df["Water_Frequency"].map(water_mapping)

# Select features for clustering (Sunlight_Hours, Humidity, and Water Frequency)
features = df[["Sunlight_Hours", "Humidity", "Water_Frequency_Encoded"]].values

# ------------------------- K-Means Clustering Functions -------------------------

# Function to calculate Euclidean distance between two points
def euclidean_distance(p1, p2):
    return sqrt(sum((p1[i] - p2[i]) ** 2 for i in range(len(p1))))

# Function to initialize k random centroids
def initialize_centroids(data, k):
    return random.sample(data.tolist(), k)

# Function to assign each data point to the nearest centroid
def assign_clusters(data, centroids):
    clusters = [[] for _ in range(len(centroids))]
    labels = []
    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        cluster_index = distances.index(min(distances))  # Assign point to the nearest centroid
        clusters[cluster_index].append(point)
        labels.append(cluster_index)
    return clusters, labels

# Function to compute new centroids by averaging cluster points
def compute_centroids(clusters):
    new_centroids = []
    for cluster in clusters:
        if cluster:  # Check if cluster is not empty
            new_centroids.append(np.mean(cluster, axis=0).tolist())
    return new_centroids

# Function to check if centroids have converged (i.e., minimal movement)
def has_converged(old_centroids, new_centroids):
    return all(euclidean_distance(old, new) < 1e-6 for old, new in zip(old_centroids, new_centroids))

# K-Means Clustering Algorithm
def k_means(data, k, max_iters=100):
    centroids = initialize_centroids(data, k)  # Step 1: Initialize centroids randomly
    for _ in range(max_iters):
        clusters, labels = assign_clusters(data, centroids)  # Step 2: Assign data points to nearest centroid
        new_centroids = compute_centroids(clusters)  # Step 3: Compute new centroids
        if has_converged(centroids, new_centroids):  # Step 4: Check convergence
            break
        centroids = new_centroids  # Update centroids
    return labels, centroids, clusters

# Function to compute Within-Cluster Sum of Squares (WCSS) for evaluation
def compute_wcss(data, labels, centroids):
    wcss = 0
    for i, centroid in enumerate(centroids):
        cluster_points = [data[j] for j in range(len(data)) if labels[j] == i]
        wcss += sum(euclidean_distance(point, centroid) ** 2 for point in cluster_points)
    return wcss

# ------------------------- Elbow Method & Finding Optimal K -------------------------

def elbow_method_and_plot(data, max_k=10, threshold=0.5):
    wcss_values = []  # Store WCSS values for different values of k

    for k in range(1, max_k + 1):
        labels, centroids, _ = k_means(data, k)
        wcss = compute_wcss(data, labels, centroids)
        wcss_values.append(wcss)

    # Compute the percentage drop in WCSS for each k
    percentage_drops = [(wcss_values[i - 1] - wcss_values[i]) / wcss_values[i - 1] for i in range(1, len(wcss_values))]

    # Find the optimal k where the drop in WCSS is below the threshold
    for i, drop in enumerate(percentage_drops):
        if drop < threshold:
            optimal_k = i + 1
            break
    else:
        optimal_k = max_k  # If no clear elbow is found, choose max_k

    print(f"Optimal number of clusters (k): {optimal_k}")
    return wcss_values, optimal_k

# ------------------------- Run K-Means & Plot Both Graphs -------------------------

# Step 1: Use the Elbow Method to determine the best number of clusters
wcss_values, optimal_k = elbow_method_and_plot(features)

# Step 2: Perform K-Means clustering using the optimal number of clusters
labels, centroids, clusters = k_means(features, optimal_k)

# Save the results to a CSV file
df["Cluster"] = labels
output_file = path.join(DATA_DIR, "plant_growth_clustered.csv")
df.to_csv(output_file, index=False)
print(f"Clustering complete. Results saved to {output_file}.")

# ------------------------- Plot Both Graphs Together -------------------------

def plot_both_graphs(data, labels, centroids, wcss_values):
    # Create a figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ------------------ Plot Elbow Method ------------------
    axes[0].plot(range(1, len(wcss_values) + 1), wcss_values, marker='o', linestyle='--')
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("WCSS")
    axes[0].set_title("Elbow Method for Optimal k")

    # ------------------ Plot Final Clustering ------------------
    data = np.array(data)
    centroids = np.array(centroids)
    unique_clusters = set(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_clusters)))

    for cluster, color in zip(unique_clusters, colors):
        cluster_points = data[np.array(labels) == cluster]  # Select data points belonging to the cluster
        axes[1].scatter(cluster_points[:, 0], cluster_points[:, 1], c=[color], label=f"Cluster {cluster}", alpha=0.6)

    # Plot centroids as red X markers
    axes[1].scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label="Centroids")
    axes[1].set_xlabel("Sunlight Hours")
    axes[1].set_ylabel("Humidity")
    axes[1].set_title("Final Clustering Results")
    axes[1].legend()

    plt.tight_layout()  # Adjust layout for better visibility
    plt.show()

# Show both graphs side by side
plot_both_graphs(features, labels, centroids, wcss_values)
