import folium
import csv
import os

def main():
    input_file = "c:\\Users\\arvid\\Documents\\python\\ports_clustered.csv"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} does not exist.")
        return

    # Read CSV file into list for processing and collect lat/long for centering map
    markers = []
    lat_list, long_list = [], []
    with open(input_file, 'r') as file:
        reader = csv.DictReader(file, delimiter='|')
        for row in reader:
            try:
                lat = float(row['lat'])
                lon = float(row['long'])
                # Use a fallback value if destination is missing
                destination = row.get('destination', 'Unknown')
                cluster_number = row.get('cluster', 'Unknown')
                markers.append((lat, lon, destination, cluster_number))
                lat_list.append(lat)
                long_list.append(lon)
            except (ValueError, KeyError) as e:
                print(f"Skipping row due to error: {e}")

    # Compute dynamic map center if markers exist
    if lat_list and long_list:
        map_center = [sum(lat_list)/len(lat_list), sum(long_list)/len(long_list)]
    else:
        map_center = [0, 0]

    mymap = folium.Map(location=map_center, zoom_start=2)
    colors = ['green', 'blue', 'red', 'purple', 'orange', 'yellow', 'brown', 'black']

    for lat, lon, destination, cluster_number in markers:
        if str(cluster_number).isdigit():
            marker_color = colors[int(cluster_number) % len(colors)]
        else:
            marker_color = 'black'
        folium.Marker(
            [lat, lon],
            popup=f"Destination: {destination}, Cluster: {cluster_number}",
            icon=folium.Icon(color=marker_color)
        ).add_to(mymap)
        folium.map.Marker(
            [lat, lon],
            icon=folium.DivIcon(
                html=f'<div style="font-size: 12px; font-weight: bold; color: black; transform: translate(-50%, -50px);">{cluster_number}</div>'
            )
        ).add_to(mymap)

    output_file = "c:\\Users\\arvid\\Documents\\python\\map.html"
    mymap.save(output_file)
    print(f"Map has been saved to {output_file}. Open it in your browser to view.")

if __name__ == "__main__":
    main()
