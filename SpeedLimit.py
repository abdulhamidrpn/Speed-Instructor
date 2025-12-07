import geopandas as gpd  # Import GeoPandas for handling geospatial data
import pandas as pd  # Import Pandas for handling CSV data
from shapely.geometry import Point  # Import Point class for creating point geometries
import time  # Import time module to measure query execution time
from datetime import datetime  # Import datetime for handling time-based logic
import math  # Import math for distance calculations

PARQUET_FILE = "NSW.parquet"  # Define the path to the GeoParquet file containing road geometries
CAMERA_CSV_FILE = "Sorted.csv"  # Define the path to the CSV file containing camera locations


def load_geoparquet():
    """Load the GeoParquet file into a GeoDataFrame."""
    print(f"Loading {PARQUET_FILE}...")
    gdf = gpd.read_parquet(PARQUET_FILE)
    gdf.sindex
    print(f"Loaded {len(gdf)} geometries.")
    return gdf


def load_camera_data():
    """Load camera locations from CSV file into a GeoDataFrame."""
    print(f"Loading camera data from {CAMERA_CSV_FILE}...")
    df = pd.read_csv(CAMERA_CSV_FILE)
    df.columns = df.columns.str.strip()
    print(f"CSV columns: {list(df.columns)}")
    expected_columns = ['Speed', 'Latitude', 'Longitude']
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        raise ValueError(f"CSV file is missing required columns: {missing_columns}")
    df = df[df['Latitude'] != 'Speed']
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    df = df.dropna(subset=['Latitude', 'Longitude'])
    geometry = [Point(lon, lat) for lon, lat in zip(df['Longitude'], df['Latitude'])]
    camera_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    print(f"Loaded {len(camera_gdf)} camera locations:")
    print(f"  - Speed cameras: {len(camera_gdf[camera_gdf['Speed'] == 'Speed'])}")
    print(f"  - Red light cameras: {len(camera_gdf[camera_gdf['Speed'] == 'Red Light'])}")
    return camera_gdf


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on Earth in meters."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371000
    return c * r


def check_nearby_cameras(camera_gdf, lat, lon, warning_distance=300):
    """Check for nearby cameras within the warning distance."""
    nearby_cameras = []
    for _, camera in camera_gdf.iterrows():
        distance = haversine_distance(lat, lon, camera['Latitude'], camera['Longitude'])
        if distance <= warning_distance:
            camera_type = "Speed Camera" if camera['Speed'] == 'Speed' else "Red Light Camera"
            nearby_cameras.append({
                'type': camera_type,
                'distance': distance,
                'lat': camera['Latitude'],
                'lon': camera['Longitude']
            })
    nearby_cameras.sort(key=lambda x: x['distance'])
    return nearby_cameras


def is_school_zone_time():
    """Check if current time is within school zone hours (8-10 AM or 2-4 PM) on weekdays."""
    now = datetime.now()
    current_time = now.time()
    current_weekday = now.weekday()  # 0=Monday, 6=Sunday

    # Only apply school zones on weekdays (Monday to Friday)
    if current_weekday >= 5:  # Saturday=5, Sunday=6
        return False

    morning_start = datetime.strptime("08:00", "%H:%M").time()
    morning_end = datetime.strptime("10:00", "%H:%M").time()
    afternoon_start = datetime.strptime("14:00", "%H:%M").time()
    afternoon_end = datetime.strptime("16:00", "%H:%M").time()

    return (morning_start <= current_time <= morning_end) or (afternoon_start <= current_time <= afternoon_end)


def get_speed_limit_with_fallback(gdf, lon, lat, buffer_distances=[0.00005, 0.000075, 0.0001],
                                  bbox_distance=0.01):
    """
    Get speed limit for a given longitude and latitude.
    Returns up to 3 closest roads for analysis.
    """
    point = Point(lon, lat)
    point_gdf = gpd.GeoDataFrame(geometry=[point], crs="EPSG:4326")
    minx, miny, maxx, maxy = lon - bbox_distance, lat - bbox_distance, lon + bbox_distance, lat + bbox_distance
    gdf_subset = gdf.cx[minx:maxx, miny:maxy]

    if gdf_subset.empty:
        return None

    # Try normal buffer distances
    for buffer_distance in buffer_distances:
        point_buffer = point.buffer(buffer_distance)
        candidates = gdf_subset[gdf_subset.geometry.intersects(point_buffer)]

        if not candidates.empty:
            projected_point = point_gdf.to_crs("EPSG:32756").geometry.iloc[0]
            projected_roads = candidates.to_crs("EPSG:32756")
            projected_roads['distance'] = projected_roads.geometry.distance(projected_point)
            sorted_roads = projected_roads.sort_values('distance')

            results = []
            seen_types = set()
            seen_speeds = set()

            for _, road in sorted_roads.iterrows():
                road_type = road['Type']
                speed = road['Speed']
                if (road_type not in seen_types or speed not in seen_speeds) and len(results) < 3:
                    results.append((speed, road_type))
                    seen_types.add(road_type)
                    seen_speeds.add(speed)
                if len(results) == 3:
                    break

            return results if results else None

    return None


def get_nearest_non_school_road(gdf, lon, lat, fallback_distances=[0.00015, 0.0002, 0.0003, 0.0005, 0.001]):
    """
    Specifically search for non-school roads using larger buffer distances.
    """
    point = Point(lon, lat)
    point_gdf = gpd.GeoDataFrame(geometry=[point], crs="EPSG:4326")

    # Use a larger bounding box for fallback search
    bbox_distance = 0.02
    minx, miny, maxx, maxy = lon - bbox_distance, lat - bbox_distance, lon + bbox_distance, lat + bbox_distance
    gdf_subset = gdf.cx[minx:maxx, miny:maxy]

    if gdf_subset.empty:
        return None

    for buffer_distance in fallback_distances:
        point_buffer = point.buffer(buffer_distance)
        candidates = gdf_subset[gdf_subset.geometry.intersects(point_buffer)]

        # Filter out school roads
        non_school_candidates = candidates[candidates['Type'] != 'School']

        if not non_school_candidates.empty:
            projected_point = point_gdf.to_crs("EPSG:32756").geometry.iloc[0]
            projected_roads = non_school_candidates.to_crs("EPSG:32756")
            projected_roads['distance'] = projected_roads.geometry.distance(projected_point)

            # Get the nearest non-school road
            nearest_road = projected_roads.loc[projected_roads['distance'].idxmin()]
            return (nearest_road['Speed'], nearest_road['Type'])

    return None


def format_camera_warning(nearby_cameras):
    """Format camera warning message."""
    if not nearby_cameras:
        return "None"
    warnings = []
    for camera in nearby_cameras:
        warnings.append(f"{camera['type']} ({camera['distance']:.0f}m)")
    return ", ".join(warnings)


def process_coordinates(gdf, camera_gdf):
    """Process user-provided coordinates to retrieve and display speed limits and camera warnings."""
    print("Enter 'lat, lon' (e.g., -33.900064, 151.065112) or 'quit' to exit.")

    while True:
        user_input = input("> ").strip()
        if user_input.lower() == 'quit':
            break
        if not user_input:
            print("Enter next 'lat, lon' or 'quit' to exit.")
            continue

        try:
            lat, lon = map(float, user_input.split(','))
        except (ValueError, IndexError):
            print("Invalid format. Please enter coordinates as 'lat, lon' (e.g., -33.900064, 151.065112)")
            continue

        start = time.time()

        # Check for nearby cameras
        nearby_cameras = check_nearby_cameras(camera_gdf, lat, lon)
        camera_warning = format_camera_warning(nearby_cameras)

        # Get speed limit for the coordinates
        results = get_speed_limit_with_fallback(gdf, lon, lat)

        # Initialize output variables
        final_speed_limit = "Not available"
        road_type = "Not available"
        school_zone = "No"
        school_zone_speed = "Not available"
        non_school_zone_speed = "Not available"

        if results:
            school_road = None
            non_school_road = None

            # Identify school and non-school roads from results
            for speed, r_type in results:
                if r_type == 'School':
                    school_road = (speed, r_type)
                elif r_type != 'School' and non_school_road is None:
                    non_school_road = (speed, r_type)

            is_school_time = is_school_zone_time()

            if school_road:
                school_zone_speed = school_road[0]

                # Always try to find non-school speed, regardless of school zone status
                if non_school_road:
                    non_school_zone_speed = non_school_road[0]
                else:
                    # Search specifically for nearest non-school road with larger buffers
                    nearest_non_school = get_nearest_non_school_road(gdf, lon, lat)
                    if nearest_non_school:
                        non_school_zone_speed = nearest_non_school[0]

                if is_school_time:
                    # School zone is active
                    final_speed_limit = school_road[0]
                    road_type = school_road[1]
                    school_zone = "Yes (Active)"
                else:
                    # School zone is inactive
                    school_zone = "Yes (Inactive)"

                    # Use non-school road speed for final speed limit
                    if non_school_road:
                        final_speed_limit = non_school_road[0]
                        road_type = non_school_road[1]
                    else:
                        # Search specifically for nearest non-school road with larger buffers
                        nearest_non_school = get_nearest_non_school_road(gdf, lon, lat)
                        if nearest_non_school:
                            final_speed_limit = nearest_non_school[0]
                            road_type = nearest_non_school[1]
                        else:
                            final_speed_limit = "Not available (No nearby non-school roads found)"
                            road_type = "Not available"
            else:
                # No school road found, use the first available result
                if non_school_road:
                    final_speed_limit = non_school_road[0]
                    road_type = non_school_road[1]
                    non_school_zone_speed = non_school_road[0]
                elif results:
                    final_speed_limit = results[0][0]
                    road_type = results[0][1]
                    non_school_zone_speed = results[0][0]

        query_time = f"{(time.time() - start) * 1000:.2f}ms"

        # Print formatted output
        print(f"Input Location: ({lat}, {lon})")
        print(f"Final Speed Limit: {final_speed_limit}")
        print(f"Road Type: {road_type}")
        print(f"School Zone: {school_zone}")
        print(f"School Zone Speed: {school_zone_speed}")
        print(f"Non School Zone Speed: {non_school_zone_speed}")
        print(f"Camera Warning: {camera_warning}")
        print(f"Query Time: {query_time}")
        print("\nEnter next 'lat, lon' or 'quit' to exit.")


if __name__ == "__main__":
    gdf = load_geoparquet()
    camera_gdf = load_camera_data()
    process_coordinates(gdf, camera_gdf)