import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from kneed import KneeLocator
import os
from tqdm import tqdm
import subprocess
from math import radians, sin, cos, sqrt, atan2

# Maximum cluster diameter in kilometers
MAX_CLUSTER_DIAMETER = 20

def load_data(file_path):
    """
    Load port data from CSV file
    """
    try:
        # Assuming the delimiter is '|' as shown in the example
        df = pd.read_csv(file_path, delimiter='|')
        print(f"Successfully loaded {len(df)} ports from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(df):
    """
    Preprocess the data for clustering
    """
    # Check for missing values
    if df.isnull().sum().sum() > 0:
        print(f"Found {df.isnull().sum().sum()} missing values, dropping those rows")
        df = df.dropna()
    
    # Extract coordinates for clustering
    coordinates = df[['lat', 'long']].values
    
    return df, coordinates

def find_optimal_k_binary_search(coordinates, max_k=30):
    """
    Find the optimal number of clusters using binary search to minimize iterations.
    """
    print("Finding optimal number of clusters using binary search...")
    low, high = 2, min(max_k, len(coordinates) // 2)
    best_k = high
    smallest_diameter = float('inf')

    while low <= high:
        mid = (low + high) // 2
        print(f"Trying with {mid} clusters...")
        labels, max_diameter = perform_clustering_with_diameter_check(coordinates, n_clusters=mid)

        if max_diameter <= MAX_CLUSTER_DIAMETER:
            best_k = mid
            smallest_diameter = max_diameter
            high = mid - 1  # Try fewer clusters
        else:
            low = mid + 1  # Try more clusters

    print(f"Optimal number of clusters: {best_k} with diameter: {smallest_diameter:.2f} km")
    return best_k

def perform_clustering_with_diameter_check(coordinates, n_clusters):
    """
    Perform K-Means clustering and calculate the maximum cluster diameter.
    """
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)  # Removed n_jobs
    labels = kmeans.fit_predict(coordinates)

    # Check maximum diameter of each cluster
    max_diameter = 0
    for cluster_id in range(n_clusters):
        cluster_points = coordinates[labels == cluster_id]
        if len(cluster_points) < 2:
            continue

        # Calculate maximum distance between any two points in cluster
        for i in range(len(cluster_points)):
            for j in range(i + 1, len(cluster_points)):
                dist = haversine_distance(
                    cluster_points[i][0], cluster_points[i][1],
                    cluster_points[j][0], cluster_points[j][1]
                )
                max_diameter = max(max_diameter, dist)

    return labels, max_diameter

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the Earth in kilometers.
    """
    R = 6371  # Earth's radius in kilometers

    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Compute differences
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c

def perform_clustering(coordinates, method='kmeans', n_clusters=10):
    """Perform clustering on port coordinates with maximum area constraint"""
    if method == 'kmeans':
        print(f"Performing K-Means clustering with {n_clusters} clusters...")
        labels, max_diameter = perform_clustering_with_diameter_check(coordinates, n_clusters)

        if max_diameter > MAX_CLUSTER_DIAMETER:
            print(f"Warning: Could not find clustering with diameter <= {MAX_CLUSTER_DIAMETER} km")
            print(f"Smallest maximum diameter found: {max_diameter:.2f} km")
            if n_clusters < len(coordinates) // 2:
                print("Trying again with more clusters...")
                return perform_clustering(coordinates, method='kmeans', n_clusters=n_clusters + 5)

        return labels
    
    elif method == 'dbscan':
        # DBSCAN is another option that doesn't require specifying the number of clusters
        # but requires tuning of eps and min_samples parameters
        scaler = StandardScaler()
        scaled_coordinates = scaler.fit_transform(coordinates)
        
        dbscan = DBSCAN(eps=0.3, min_samples=5)
        cluster_labels = dbscan.fit_predict(scaled_coordinates)
        
        # Handle the noise points (labeled as -1) by assigning them to a new cluster
        cluster_labels[cluster_labels == -1] = np.max(cluster_labels) + 1
        
        return cluster_labels
    
    else:
        raise ValueError(f"Unsupported clustering method: {method}")

def extend_data_with_clusters(df, cluster_labels):
    """
    Add cluster information to the original dataframe
    """
    df_extended = df.copy()
    df_extended['cluster'] = cluster_labels
    return df_extended

def save_results(df_extended, output_file):
    """
    Save the extended dataframe to a CSV file
    """
    df_extended.to_csv(output_file, sep='|', index=False)
    print(f"Results saved to {output_file}")

def visualize_clusters(df_extended, output_file='cluster_visualization.png'):
    """
    Create a visualization of the clusters
    """
    print("Creating visualization...")
    
    plt.figure(figsize=(12, 8))
    
    # Create a scatter plot colored by cluster
    scatter = plt.scatter(
        df_extended['long'], 
        df_extended['lat'],
        c=df_extended['cluster'], 
        cmap='viridis', 
        alpha=0.6,
        s=10
    )
    
    # Add a colorbar for the cluster labels
    plt.colorbar(scatter, label='Cluster')
    
    # Set labels and title
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Clusters of Ports Based on Geographical Proximity')
    
    # Save the visualization
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Visualization saved to {output_file}")

def main():
    # Define file paths
    input_file = input('input file name: ')  # Replace with your actual input file
    output_file = 'ports_clustered.csv'
    
    # Step 1: Load and preprocess data
    df = load_data(input_file)
    if df is None:
        return
    
    df, coordinates = preprocess_data(df)
    
    # Step 2: Find optimal number of clusters using binary search
    optimal_k = find_optimal_k_binary_search(coordinates)
    
    # Step 3: Perform clustering
    cluster_labels = perform_clustering(coordinates, method='kmeans', n_clusters=optimal_k)
    
    # Step 4: Extend the data with cluster information
    df_extended = extend_data_with_clusters(df, cluster_labels)
    
    # Step 5: Save the results
    save_results(df_extended, output_file)
    
    # Step 6: Visualize the clusters (optional)
    visualize_clusters(df_extended)
    
    print("Port clustering completed successfully!")
    # Print some statistics about the clusters
    cluster_stats = df_extended.groupby('cluster').size().reset_index(name='count')
    print("\nCluster statistics:")
    print(f"Number of clusters: {len(cluster_stats)}")
    print(f"Largest cluster: {cluster_stats['count'].max()} ports")
    print(f"Smallest cluster: {cluster_stats['count'].min()} ports")
    print(f"Average cluster size: {cluster_stats['count'].mean():.2f} ports")

    # Step 7: Run visualize_on_map.py
    try:
        print("Running visualize_on_map.py to generate the map...")
        subprocess.run(
            ["python", "c:\\Users\\arvid\\Documents\\python\\visualize_on_map.py"], 
            check=True
        )
        print("Map visualization completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error running visualize_on_map.py: {e}")

if __name__ == "__main__":
    main()
