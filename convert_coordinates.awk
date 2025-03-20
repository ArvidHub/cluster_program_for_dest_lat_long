BEGIN {
    FS = "|"; OFS = "|";
    print "land", "port", "location", "city", "region", "runway", "approach", "usage", "opstatus", "lat", "long";
}

NR > 1 {
    # Check if coordinates exist and are in the format DDDMMN DDDMME
    if ($11 ~ /^[0-9]+[NS] [0-9]+[EW]$/) {
        lat_deg = substr($11, 1, 2);
        lat_min = substr($11, 3, 2);
        lat_dir = substr($11, 5, 1);
        lon_deg = substr($11, 7, 3);
        lon_min = substr($11, 10, 2);
        lon_dir = substr($11, 12, 1);

        # Convert to decimal degrees
        lat = lat_deg + (lat_min / 60);
        lon = lon_deg + (lon_min / 60);

        # Adjust for direction
        if (lat_dir == "S") lat = -lat;
        if (lon_dir == "W") lon = -lon;

        # Replace coordinates with lat and long
        $11 = lat;
        $12 = lon;
    } else {
        # If coordinates are missing or invalid, set lat and long to empty
        $11 = ""; $12 = "";
    }
    print $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12;
}
