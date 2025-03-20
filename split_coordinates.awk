BEGIN {
    FS = ","; OFS = ",";  # Set input and output field separators
    print "land,port,location,city,region,runway,approach,usage,opstatus,lat,long";  # Print header
}

NR > 1 {
    # Extract coordinates and split into latitude and longitude
    if ($11 ~ /^[0-9]+[NS] [0-9]+[EW]$/) {
        split($11, coord, " ");
        lat = substr(coord[1], 1, length(coord[1]) - 1) * (index(coord[1], "S") ? -1 : 1);
        long = substr(coord[2], 1, length(coord[2]) - 1) * (index(coord[2], "W") ? -1 : 1);
    } else {
        lat = ""; long = "";  # Handle missing or invalid coordinates
    }
    # Print the updated row with lat and long
    print $2, $3, $4, $5, $6, $7, $8, $9, $10, lat, long;
}
