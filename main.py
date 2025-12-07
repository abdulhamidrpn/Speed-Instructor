# main.py
"""
FastAPI Backend Service - Geospatial Speed Limit Query
Integrates SpeedLimit.py for efficient coordinate-based speed limit lookups
Loads data once at startup for optimal performance
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from typing import List, Optional
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import time
import math
from datetime import datetime
import logging

# ===================== LOGGING SETUP =====================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===================== CONSTANTS =====================

PARQUET_FILE = "NSW.parquet"
CAMERA_CSV_FILE = "Sorted.csv"

# ===================== PYDANTIC MODELS =====================

class CoordinateRequest(BaseModel):
    """
    Request model for speed limit query
    Accepts latitude and longitude coordinates
    """
    latitude: float = Field(..., ge=-90, le=90, description="Latitude (-90 to 90)")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude (-180 to 180)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "latitude": -33.900064,
                "longitude": 151.065112
            }
        }


class CameraWarning(BaseModel):
    """
    Camera detection model
    Contains camera type, distance, and location
    """
    type: str = Field(..., description="Speed Camera or Red Light Camera")
    distance: float = Field(..., description="Distance in meters")
    lat: float = Field(..., description="Camera latitude")
    lon: float = Field(..., description="Camera longitude")


class SpeedLimitResponse(BaseModel):
    """
    Complete geospatial analysis response
    Returns speed limits, road types, school zones, and camera alerts
    """
    input_location: str = Field(..., description="Query coordinates as string")
    final_speed_limit: str = Field(..., description="Speed limit (e.g., '50 km/h')")
    road_type: str = Field(..., description="Type of road (Local, Highway, etc.)")
    school_zone: str = Field(..., description="School zone status")
    school_zone_speed: str = Field(..., description="Speed limit in school zone")
    non_school_zone_speed: str = Field(..., description="Speed limit outside school zone")
    camera_warnings: List[CameraWarning] = Field(default_factory=list, description="Nearby cameras")
    query_time: str = Field(..., description="Query execution time")
    
    class Config:
        json_schema_extra = {
            "example": {
                "input_location": "(-33.900064, 151.065112)",
                "final_speed_limit": "50",
                "road_type": "Local",
                "school_zone": "Yes (Inactive)",
                "school_zone_speed": "40",
                "non_school_zone_speed": "50",
                "camera_warnings": [
                    {
                        "type": "Speed Camera",
                        "distance": 145.5,
                        "lat": -33.9001,
                        "lon": 151.0652
                    }
                ],
                "query_time": "45.23ms"
            }
        }


class HealthCheckResponse(BaseModel):
    """Health check response model"""
    status: str
    data_loaded: bool
    parquet_records: int = 0
    camera_records: int = 0


class StatsResponse(BaseModel):
    """Backend statistics response model"""
    total_roads: int
    total_cameras: int
    speed_cameras: int
    red_light_cameras: int


class ErrorResponse(BaseModel):
    """Error response model"""
    detail: str
    status_code: int


# ===================== GLOBAL STATE (LOADED ONCE) =====================

gdf_global: Optional[gpd.GeoDataFrame] = None
camera_gdf_global: Optional[gpd.GeoDataFrame] = None

# ===================== DATA LOADING FUNCTIONS =====================

def load_geoparquet() -> gpd.GeoDataFrame:
    """
    Load the GeoParquet file into a GeoDataFrame.
    Builds spatial index for efficient queries.
    
    Returns:
        GeoDataFrame: Loaded road geometry data
    """
    logger.info(f"📂 Loading {PARQUET_FILE}...")
    try:
        gdf = gpd.read_parquet(PARQUET_FILE)
        gdf.sindex  # Build spatial index
        logger.info(f"✅ Loaded {len(gdf)} road geometries with spatial index")
        return gdf
    except FileNotFoundError:
        logger.error(f"❌ File not found: {PARQUET_FILE}")
        raise
    except Exception as e:
        logger.error(f"❌ Error loading parquet: {e}")
        raise


def load_camera_data() -> gpd.GeoDataFrame:
    """
    Load camera locations from CSV file.
    Creates GeoDataFrame with spatial geometry.
    
    Returns:
        GeoDataFrame: Camera locations with geometry
    
    Raises:
        ValueError: If required columns missing
        FileNotFoundError: If CSV file not found
    """
    logger.info(f"📂 Loading camera data from {CAMERA_CSV_FILE}...")
    try:
        df = pd.read_csv(CAMERA_CSV_FILE)
        df.columns = df.columns.str.strip()
        
        # Validate columns
        expected_columns = ['Speed', 'Latitude', 'Longitude']
        missing_columns = [col for col in expected_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Missing columns: {missing_columns}")
            logger.error(f"Available columns: {list(df.columns)}")
            raise ValueError(f"CSV missing columns: {missing_columns}")
        
        # Data cleaning
        df = df[df['Latitude'] != 'Speed']
        df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
        df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
        df = df.dropna(subset=['Latitude', 'Longitude'])
        
        # Create GeoDataFrame
        geometry = [Point(lon, lat) for lon, lat in zip(df['Longitude'], df['Latitude'])]
        camera_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
        
        speed_count = len(camera_gdf[camera_gdf['Speed'] == 'Speed'])
        red_light_count = len(camera_gdf[camera_gdf['Speed'] == 'Red Light'])
        
        logger.info(f"✅ Loaded {len(camera_gdf)} camera locations")
        logger.info(f"   - Speed cameras: {speed_count}")
        logger.info(f"   - Red light cameras: {red_light_count}")
        
        return camera_gdf
    except FileNotFoundError:
        logger.error(f"❌ File not found: {CAMERA_CSV_FILE}")
        raise
    except Exception as e:
        logger.error(f"❌ Error loading cameras: {e}")
        raise


# ===================== GEOSPATIAL CALCULATION FUNCTIONS =====================

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great circle distance between two points using Haversine formula.
    
    Args:
        lat1, lon1: First point coordinates (degrees)
        lat2, lon2: Second point coordinates (degrees)
    
    Returns:
        float: Distance in meters
    """
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371000  # Earth radius in meters
    return c * r


def check_nearby_cameras(lat: float, lon: float, warning_distance: float = 300) -> List[CameraWarning]:
    """
    Check for nearby cameras within warning distance.
    Uses Haversine distance for accurate geographic distance.
    
    Args:
        lat: Query latitude
        lon: Query longitude
        warning_distance: Search radius in meters (default 300m)
    
    Returns:
        List[CameraWarning]: Nearby cameras sorted by distance
    """
    if camera_gdf_global is None:
        return []
    
    nearby_cameras: List[CameraWarning] = []
    
    for _, camera in camera_gdf_global.iterrows():
        distance = haversine_distance(lat, lon, camera['Latitude'], camera['Longitude'])
        
        if distance <= warning_distance:
            camera_type = "Speed Camera" if camera['Speed'] == 'Speed' else "Red Light Camera"
            nearby_cameras.append(CameraWarning(
                type=camera_type,
                distance=distance,
                lat=camera['Latitude'],
                lon=camera['Longitude']
            ))
    
    # Sort by distance (closest first)
    nearby_cameras.sort(key=lambda x: x.distance)
    return nearby_cameras


def is_school_zone_time() -> bool:
    """
    Check if current time is within school zone hours.
    School zones active weekdays 8-10 AM and 2-4 PM.
    
    Returns:
        bool: True if currently in school zone time
    """
    now = datetime.now()
    current_time = now.time()
    weekday = now.weekday()  # 0=Monday, 6=Sunday
    
    # Only weekdays (Monday-Friday)
    if weekday >= 5:  # Saturday=5, Sunday=6
        return False
    
    morning_start = datetime.strptime("08:00", "%H:%M").time()
    morning_end = datetime.strptime("10:00", "%H:%M").time()
    afternoon_start = datetime.strptime("14:00", "%H:%M").time()
    afternoon_end = datetime.strptime("16:00", "%H:%M").time()
    
    return (morning_start <= current_time <= morning_end) or \
           (afternoon_start <= current_time <= afternoon_end)


def get_speed_limit_with_fallback(
    lon: float,
    lat: float,
    buffer_distances: Optional[List[float]] = None,
    bbox_distance: float = 0.01
) -> Optional[List[tuple]]:
    """
    Get speed limit for coordinates using spatial indexing.
    Returns up to 3 closest roads with different types/speeds.
    
    Uses progressive buffer expansion for robustness:
    1. First tries small buffer (0.00005°)
    2. Then medium buffer (0.000075°)
    3. Finally larger buffer (0.0001°)
    
    Args:
        lon: Longitude coordinate
        lat: Latitude coordinate
        buffer_distances: List of search radius values in degrees
        bbox_distance: Bounding box search radius in degrees
    
    Returns:
        List of (speed, road_type) tuples, or None if not found
    """
    if gdf_global is None:
        return None
    
    if buffer_distances is None:
        buffer_distances = [0.00005, 0.000075, 0.0001]
    
    point = Point(lon, lat)
    point_gdf = gpd.GeoDataFrame(geometry=[point], crs="EPSG:4326")
    
    # Bounding box query for faster filtering
    minx = lon - bbox_distance
    miny = lat - bbox_distance
    maxx = lon + bbox_distance
    maxy = lat + bbox_distance
    gdf_subset = gdf_global.cx[minx:maxx, miny:maxy]
    
    if gdf_subset.empty:
        return None
    
    # Try increasing buffer distances
    for buffer_distance in buffer_distances:
        point_buffer = point.buffer(buffer_distance)
        candidates = gdf_subset[gdf_subset.geometry.intersects(point_buffer)]
        
        if not candidates.empty:
            # Project to UTM for accurate distance calculations
            projected_point = point_gdf.to_crs("EPSG:32756").geometry.iloc[0]
            projected_roads = candidates.to_crs("EPSG:32756")
            projected_roads['distance'] = projected_roads.geometry.distance(projected_point)
            
            sorted_roads = projected_roads.sort_values('distance')
            
            # Collect up to 3 unique results
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


def get_nearest_non_school_road(
    lon: float,
    lat: float,
    fallback_distances: Optional[List[float]] = None
) -> Optional[tuple]:
    """
    Find nearest non-school road using larger buffer.
    Fallback when school road doesn't have non-school alternative.
    
    Args:
        lon: Longitude coordinate
        lat: Latitude coordinate
        fallback_distances: List of search radius values in degrees
    
    Returns:
        (speed, road_type) tuple or None
    """
    if gdf_global is None:
        return None
    
    if fallback_distances is None:
        fallback_distances = [0.00015, 0.0002, 0.0003, 0.0005, 0.001]
    
    point = Point(lon, lat)
    point_gdf = gpd.GeoDataFrame(geometry=[point], crs="EPSG:4326")
    
    bbox_distance = 0.02
    minx = lon - bbox_distance
    miny = lat - bbox_distance
    maxx = lon + bbox_distance
    maxy = lat + bbox_distance
    gdf_subset = gdf_global.cx[minx:maxx, miny:maxy]
    
    if gdf_subset.empty:
        return None
    
    for buffer_distance in fallback_distances:
        point_buffer = point.buffer(buffer_distance)
        candidates = gdf_subset[gdf_subset.geometry.intersects(point_buffer)]
        
        # Filter out school roads
        non_school_candidates = candidates[candidates['Type'] != 'School']
        
        if not non_school_candidates.empty:
            # Project and find nearest
            projected_point = point_gdf.to_crs("EPSG:32756").geometry.iloc[0]
            projected_roads = non_school_candidates.to_crs("EPSG:32756")
            projected_roads['distance'] = projected_roads.geometry.distance(projected_point)
            
            nearest_road = projected_roads.loc[projected_roads['distance'].idxmin()]
            return (nearest_road['Speed'], nearest_road['Type'])
    
    return None


# ===================== MAIN PROCESSING FUNCTION =====================

def process_coordinates(latitude: float, longitude: float) -> SpeedLimitResponse:
    """
    Main processing function combining all calculations.
    Performs speed limit lookup, school zone detection, and camera detection.
    
    Algorithm:
    1. Check nearby cameras (300m radius)
    2. Get speed limits for coordinates
    3. Identify school vs non-school roads
    4. Apply school zone logic (time-based)
    5. Return comprehensive analysis
    
    Args:
        latitude: Query latitude
        longitude: Query longitude
    
    Returns:
        SpeedLimitResponse: Complete analysis with all data
    
    Raises:
        Exception: If data not loaded or query fails
    """
    start_time = time.time()
    
    if gdf_global is None or camera_gdf_global is None:
        raise Exception("Geospatial data not loaded")
    
    # Initialize response
    response = SpeedLimitResponse(
        input_location=f"({latitude}, {longitude})",
        final_speed_limit="Not available",
        road_type="Not available",
        school_zone="No",
        school_zone_speed="Not available",
        non_school_zone_speed="Not available",
        camera_warnings=[],
        query_time="0ms"
    )
    
    # Check for nearby cameras
    response.camera_warnings = check_nearby_cameras(latitude, longitude)
    
    # Get speed limits
    road_results = get_speed_limit_with_fallback(longitude, latitude)
    
    if road_results:
        school_road = None
        non_school_road = None
        
        # Separate school and non-school roads
        for speed, road_type in road_results:
            if road_type == 'School':
                school_road = (speed, road_type)
            elif non_school_road is None:
                non_school_road = (speed, road_type)
        
        is_school_time = is_school_zone_time()
        
        if school_road:
            response.school_zone_speed = school_road[0]
            
            # Find non-school speed
            if non_school_road:
                response.non_school_zone_speed = non_school_road[0]
            else:
                nearest_non_school = get_nearest_non_school_road(longitude, latitude)
                if nearest_non_school:
                    response.non_school_zone_speed = nearest_non_school[0]
            
            # Apply school zone logic
            if is_school_time:
                response.final_speed_limit = school_road[0]
                response.road_type = school_road[1]
                response.school_zone = "Yes (Active)"
            else:
                response.school_zone = "Yes (Inactive)"
                if non_school_road:
                    response.final_speed_limit = non_school_road[0]
                    response.road_type = non_school_road[1]
                else:
                    nearest_non_school = get_nearest_non_school_road(longitude, latitude)
                    if nearest_non_school:
                        response.final_speed_limit = nearest_non_school[0]
                        response.road_type = nearest_non_school[1]
        else:
            # No school road found
            if non_school_road:
                response.final_speed_limit = non_school_road[0]
                response.road_type = non_school_road[1]
                response.non_school_zone_speed = non_school_road[0]
            elif road_results:
                response.final_speed_limit = road_results[0][0]
                response.road_type = road_results[0][1]
                response.non_school_zone_speed = road_results[0][0]
    
    # Calculate query time
    query_time_ms = (time.time() - start_time) * 1000
    response.query_time = f"{query_time_ms:.2f}ms"
    
    return response


# ===================== FASTAPI LIFECYCLE =====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager
    Loads data on startup, cleans up on shutdown
    """
    global gdf_global, camera_gdf_global
    
    # ========== STARTUP ==========
    logger.info("🚀 FastAPI application starting...")
    try:
        logger.info("📦 Loading geospatial data...")
        gdf_global = load_geoparquet()
        camera_gdf_global = load_camera_data()
        logger.info("✅ All data loaded successfully")
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield  # Application runs here
    
    # ========== SHUTDOWN ==========
    logger.info("🛑 FastAPI application shutting down...")
    logger.info("✅ Cleanup completed")


# ===================== FASTAPI APP =====================

app = FastAPI(
    title="Speed Limit Query Service",
    description="FastAPI backend for geospatial speed limit, road type, school zone, and camera detection",
    version="1.0.0",
    lifespan=lifespan
)

# ===================== MIDDLEWARE =====================

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== ENDPOINTS =====================

@app.get("/", tags=["Info"])
async def read_root() -> dict:
    """
    Root endpoint with API information
    
    Returns:
        dict: Welcome message and API version
    """
    return {
        "message": "Welcome to Speed Limit Query API!",
        "version": "1.0.0",
        "documentation": "/docs"
    }


@app.get("/about", tags=["Info"])
async def read_about() -> dict:
    """
    API information endpoint
    
    Returns:
        dict: API description and capabilities
    """
    return {
        "message": "Speed Limit Query API 1.0.0",
        "description": "This API provides geospatial speed limit lookups with school zone and camera detection",
        "features": [
            "Speed limit lookup by coordinates",
            "Road type detection",
            "School zone detection",
            "Camera detection (Speed & Red Light)",
            "Query performance metrics"
        ]
    }


@app.post(
    "/query",
    response_model=SpeedLimitResponse,
    status_code=status.HTTP_200_OK,
    tags=["Query"]
)
async def query_speed_limit(request: CoordinateRequest) -> SpeedLimitResponse:
    """
    Query speed limit for given coordinates.
    
    Main endpoint for geospatial analysis.
    
    **Request:**
    ```
    {
        "latitude": -33.900064,
        "longitude": 151.065112
    }
    ```
    
    **Response:**
    ```
    {
        "input_location": "(-33.900064, 151.065112)",
        "final_speed_limit": "50",
        "road_type": "Local",
        "school_zone": "Yes (Inactive)",
        "school_zone_speed": "40",
        "non_school_zone_speed": "50",
        "camera_warnings": [...],
        "query_time": "45.23ms"
    }
    ```
    
    Args:
        request: CoordinateRequest with latitude and longitude
    
    Returns:
        SpeedLimitResponse: Complete geospatial analysis
    
    Raises:
        HTTPException 400: Invalid coordinates
        HTTPException 503: Data not loaded
        HTTPException 500: Processing error
    """
    try:
        # Validate data loaded
        if gdf_global is None or camera_gdf_global is None:
            logger.warning("Query attempted but data not loaded")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Geospatial data not loaded"
            )
        
        # Process coordinates
        logger.info(f"🔍 Processing query: ({request.latitude}, {request.longitude})")
        result = process_coordinates(request.latitude, request.longitude)
        logger.info(f"✅ Query completed in {result.query_time}")
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}"
        )


@app.get("/health", response_model=HealthCheckResponse, tags=["System"])
async def health_check() -> HealthCheckResponse:
    """
    Health check endpoint.
    Verifies backend status and data availability.
    
    Returns:
        HealthCheckResponse: Status and data statistics
    """
    data_loaded = gdf_global is not None and camera_gdf_global is not None
    
    return HealthCheckResponse(
        status="healthy" if data_loaded else "unhealthy",
        data_loaded=data_loaded,
        parquet_records=len(gdf_global) if gdf_global is not None else 0,
        camera_records=len(camera_gdf_global) if camera_gdf_global is not None else 0
    )


@app.get("/stats", response_model=StatsResponse, tags=["System"])
async def get_stats() -> StatsResponse:
    """
    Get backend statistics.
    Returns data summary for monitoring.
    
    Returns:
        StatsResponse: Statistics about loaded data
    
    Raises:
        HTTPException 503: Data not loaded
    """
    if gdf_global is None or camera_gdf_global is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Data not loaded"
        )
    
    speed_cameras = len(camera_gdf_global[camera_gdf_global['Speed'] == 'Speed'])
    red_light_cameras = len(camera_gdf_global[camera_gdf_global['Speed'] == 'Red Light'])
    
    return StatsResponse(
        total_roads=len(gdf_global),
        total_cameras=len(camera_gdf_global),
        speed_cameras=speed_cameras,
        red_light_cameras=red_light_cameras
    )


# ===================== ERROR HANDLERS =====================

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle validation errors"""
    logger.error(f"Validation error: {exc}")
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=str(exc)
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected errors"""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Internal server error"
    )


# ===================== RUN APPLICATION =====================

if __name__ == "__main__":
    import uvicorn
    
    # Start server
    print("🚀 Starting FastAPI server...")
    print("📍 API available at: http://localhost:8000")
    print("📚 Documentation at: http://localhost:8000/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
