import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

def visualize_clusters(dataset_file):
    """Performs hierarchical clustering on the provided dataset and visualizes the clusters."""

    try:
        dataset = pd.read_csv(dataset_file)
    except pd.errors.ParserError as e:
        print(f"Error parsing CSV: {e}")
        return None  # Indicate an error occurred

    X = dataset.iloc[:, [3, 4]].values  # Extract relevant features

    # Create a dendrogram (optional for visual exploration)
    dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
    plt.title('Dendrogram')
    plt.xlabel('Customers')
    plt.ylabel('Euclidean distances')
    plt.show()  # Display dendrogram (optional)

    # Perform hierarchical clustering with 5 clusters
    hc = AgglomerativeClustering(n_clusters=5,  linkage='ward')
    y_hc = hc.fit_predict(X)

    # Create the cluster visualization
    plt.figure(figsize=(8, 6))  # Adjust plot size for better visibility
    colors = ['red', 'blue', 'green', 'cyan', 'magenta']
    for i in range(5):
        plt.scatter(X[y_hc == i, 0], X[y_hc == i, 1], s=100, c=colors[i], label=f'Cluster {i+1}')

    plt.title('Clusters of Customers')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()

    plt.tight_layout()
    plt.savefig("clusters.png")  # Save plot to a temporary file

    return gr.Image("clusters.png")

# Create the Gradio interface
iface = gr.Interface(
    visualize_clusters,
    inputs=[gr.File(label="Upload your CSV dataset (expected format: annual income, spending score)", type="filepath")],
    outputs=gr.Image(label="Cluster Visualization"),
    title="Hierarchical Clustering with Gradio",
    description="Visualize clusters in your CSV dataset containing annual income and spending score columns."
)

iface.launch()