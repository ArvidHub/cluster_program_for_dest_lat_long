import folium
import csv

# Input file path
input_file = "c:\\Users\\arvid\\Documents\\python\\ports_clustered.csv"

# Initialize map centered at an arbitrary location
map_center = [0, 0]
mymap = folium.Map(location=map_center, zoom_start=2)

# Define a list of colors for clusters
colors = ['green', 'blue', 'red', 'purple', 'orange', 'yellow', 'brown', 'black']

# Read the file and add markers
with open(input_file, 'r') as file:
    reader = csv.DictReader(file, delimiter='|')
    for row in reader:
        lat = float(row['lat'])
        lon = float(row['long'])
        destination = row['destination']
        cluster_number = row.get('cluster', 'Unknown')  # Assuming 'cluster' column exists in the CSV
        
        # Determine the color using modulo if cluster_number is numeric
        if cluster_number.isdigit():
            marker_color = colors[int(cluster_number) % len(colors)]
        else:
            marker_color = 'black'  # Default color for non-numeric or unknown clusters
        
        # Add marker with cluster number in the popup and specific color
        folium.Marker(
            [lat, lon],
            popup=f"Destination: {destination}, Cluster: {cluster_number}",
            icon=folium.Icon(color=marker_color)
        ).add_to(mymap)

# Save the map to an HTML file
output_file = "c:\\Users\\arvid\\Documents\\python\\map.html"
mymap.save(output_file)

print(f"Map has been saved to {output_file}. Open it in your browser to view.")
