import logging
from math import radians, sin, cos, sqrt, atan2
from typing import Tuple, Any, Optional

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Maximum cluster diameter in kilometers
MAX_CLUSTER_DIAMETER = 20

def load_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    Load port data from CSV file with '|' delimiter.
    """
    try:
        df = pd.read_csv(file_path, delimiter='|')
        logging.info(f"Successfully loaded {len(df)} ports from {file_path}")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None

def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Any]:
    """
    Preprocess data: drop missing values and extract coordinates.
    """
    if df.isnull().sum().sum() > 0:
        logging.warning(f"Found {df.isnull().sum().sum()} missing values, dropping those rows")
        df = df.dropna()
    
    coordinates = df[['lat', 'long']].values
    return df, coordinates

def find_optimal_k_binary_search(coordinates: Any, max_k: int = 30) -> int:
    """
    Use binary search to find the smallest k such that the maximum cluster diameter <= MAX_CLUSTER_DIAMETER.
    """
    logging.info("Finding optimal number of clusters using binary search...")
    low, high = 2, min(max_k, len(coordinates) // 2)
    best_k = high
    smallest_diameter = float('inf')

    while low <= high:
        mid = (low + high) // 2
        logging.info(f"Trying with {mid} clusters...")
        labels, max_diameter = perform_clustering_with_diameter_check(coordinates, n_clusters=mid)

        if max_diameter <= MAX_CLUSTER_DIAMETER:
            best_k = mid
            smallest_diameter = max_diameter
            high = mid - 1  # try fewer clusters
        else:
            low = mid + 1  # try more clusters

    logging.info(f"Optimal number of clusters: {best_k} with diameter: {smallest_diameter:.2f} km")
    return best_k

def perform_clustering_with_diameter_check(coordinates: Any, n_clusters: int) -> Tuple[Any, float]:
    """
    Perform K-Means clustering and calculate the maximum pairwise cluster distance (cluster diameter).
    """
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(coordinates)
    max_diameter = 0

    for cluster_id in range(n_clusters):
        cluster_points = coordinates[labels == cluster_id]
        if len(cluster_points) < 2:
            continue
        for i in range(len(cluster_points)):
            for j in range(i + 1, len(cluster_points)):
                dist = haversine_distance(
                    cluster_points[i][0], cluster_points[i][1],
                    cluster_points[j][0], cluster_points[j][1]
                )
                max_diameter = max(max_diameter, dist)
    return labels, max_diameter

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute the great-circle distance between two points on the Earth.
    """
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def perform_clustering(coordinates: Any, method: str = 'kmeans', n_clusters: int = 10) -> Any:
    """
    Perform clustering on port coordinates using KMeans or DBSCAN.
    """
    if method == 'kmeans':
        logging.info(f"Performing K-Means clustering with {n_clusters} clusters...")
        labels, max_diameter = perform_clustering_with_diameter_check(coordinates, n_clusters)

        if max_diameter > MAX_CLUSTER_DIAMETER:
            logging.warning(f"Cluster diameter {max_diameter:.2f} km exceeds maximum allowed {MAX_CLUSTER_DIAMETER} km")
            if n_clusters < len(coordinates) // 2:
                logging.info("Retrying with more clusters...")
                return perform_clustering(coordinates, method='kmeans', n_clusters=n_clusters + 5)
        return labels
    
    elif method == 'dbscan':
        scaler = StandardScaler()
        scaled_coordinates = scaler.fit_transform(coordinates)        
        dbscan = DBSCAN(eps=0.3, min_samples=5)
        cluster_labels = dbscan.fit_predict(scaled_coordinates)
        cluster_labels[cluster_labels == -1] = np.max(cluster_labels) + 1
        return cluster_labels
    else:
        raise ValueError(f"Unsupported clustering method: {method}")

def extend_data_with_clusters(df: pd.DataFrame, cluster_labels: Any) -> pd.DataFrame:
    """
    Extend the original dataframe with a new column for cluster labels.
    """
    df_extended = df.copy()
    df_extended['cluster'] = cluster_labels
    return df_extended

def save_results(df_extended: pd.DataFrame, output_file: str) -> None:
    """
    Save the dataframe with cluster labels to CSV.
    """
    df_extended.to_csv(output_file, sep='|', index=False)
    logging.info(f"Results saved to {output_file}")

def visualize_clusters(df_extended: pd.DataFrame, output_file: str = 'cluster_visualization.png') -> None:
    """
    Generate and save a scatter plot of the clusters.
    """
    logging.info("Creating visualization...")
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        df_extended['long'], 
        df_extended['lat'],
        c=df_extended['cluster'], 
        cmap='viridis', 
        alpha=0.6,
        s=10
    )
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Clusters of Ports Based on Geographical Proximity')
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    logging.info(f"Visualization saved to {output_file}")

def main():
    # Define file paths
    input_file = input('Input file name: ')
    output_file = 'ports_clustered.csv'
    
    # Load and preprocess data
    df = load_data(input_file)
    if df is None:
        return
    df, coordinates = preprocess_data(df)
    
    # Find optimal k and perform clustering
    optimal_k = find_optimal_k_binary_search(coordinates)
    cluster_labels = perform_clustering(coordinates, method='kmeans', n_clusters=optimal_k)
    
    # Extend data with clusters and save results
    df_extended = extend_data_with_clusters(df, cluster_labels)
    save_results(df_extended, output_file)
    visualize_clusters(df_extended)
    
    logging.info("Port clustering completed successfully!")
    cluster_stats = df_extended.groupby('cluster').size().reset_index(name='count')
    logging.info(f"Number of clusters: {len(cluster_stats)}")
    logging.info(f"Largest cluster: {cluster_stats['count'].max()} ports")
    logging.info(f"Smallest cluster: {cluster_stats['count'].min()} ports")
    logging.info(f"Average cluster size: {cluster_stats['count'].mean():.2f} ports")

    # Run map visualization
    try:
        import subprocess
        logging.info("Running visualize_on_map.py to generate the map...")
        subprocess.run(["python", "c:\\Users\\arvid\\Documents\\python\\visualize_on_map.py"], check=True)
        logging.info("Map visualization completed successfully!")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running visualize_on_map.py: {e}")

if __name__ == "__main__":
    main()
