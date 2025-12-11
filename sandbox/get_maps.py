#!/usr/bin/env python3
"""
Download high-resolution satellite imagery for a particular location.

This script uses free tile services (no API key required) to download satellite imagery.
By default, it uses Esri World Imagery, which is free and publicly available.

Usage:
    python get_maps.py --lat 32.882449 --lon -117.2348557 --zoom 18 --output location.png
    python get_maps.py --coordinates "32.882449,-117.2348557" --zoom 18 --tiles 5x5
    python get_maps.py --lat 32.882449 --lon -117.2348557 --zoom 20 --tiles 10x10 --output highres.png
"""

import argparse
import math
import os
import sys
import time
from io import BytesIO
from typing import Tuple

try:
    import requests
except ImportError:
    print("Error: 'requests' library is required.")
    print("Install it with: pip install requests")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("Error: 'Pillow' library is required.")
    print("Install it with: pip install Pillow")
    sys.exit(1)


def deg2num(lat_deg: float, lon_deg: float, zoom: int) -> Tuple[int, int]:
    """
    Convert latitude/longitude to tile coordinates.
    
    Args:
        lat_deg: Latitude in degrees
        lon_deg: Longitude in degrees
        zoom: Zoom level
        
    Returns:
        Tuple of (tile_x, tile_y)
    """
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    tile_x = int((lon_deg + 180.0) / 360.0 * n)
    tile_y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return tile_x, tile_y


def num2deg(tile_x: int, tile_y: int, zoom: int) -> Tuple[float, float]:
    """
    Convert tile coordinates to latitude/longitude of top-left corner.
    
    Args:
        tile_x: Tile X coordinate
        tile_y: Tile Y coordinate
        zoom: Zoom level
        
    Returns:
        Tuple of (latitude, longitude)
    """
    n = 2.0 ** zoom
    lon_deg = tile_x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * tile_y / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg


def get_tile_url(tile_x: int, tile_y: int, zoom: int, provider: str = "esri", maptype: str = "satellite") -> str:
    """
    Generate tile URL for various free satellite imagery providers.
    
    Args:
        tile_x: Tile X coordinate
        tile_y: Tile Y coordinate
        zoom: Zoom level
        provider: Provider to use - 'esri' (Esri World Imagery) or 'google' (Google Maps)
        maptype: Map type - 'satellite' or 'hybrid'
        
    Returns:
        Tile URL string
    """
    if provider == "esri":
        # Esri World Imagery - free, no API key required
        # Uses Web Mercator projection (same as Google Maps)
        # Correct URL format for Esri ArcGIS Online
        url = f"https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{tile_y}/{tile_x}"
    elif provider == "google":
        # Google Maps tile (may have ToS restrictions)
        # Try multiple URL patterns
        subdomain = (tile_x + tile_y) % 4
        if maptype == "satellite":
            # Try the standard Google Maps satellite tile format
            url = f"https://mt{subdomain}.google.com/vt/lyrs=s&x={tile_x}&y={tile_y}&z={zoom}"
        else:  # hybrid
            url = f"https://mt{subdomain}.google.com/vt/lyrs=y&x={tile_x}&y={tile_y}&z={zoom}"
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
    return url


def download_tile(tile_x: int, tile_y: int, zoom: int, provider: str = "esri", maptype: str = "satellite", retries: int = 3) -> Image.Image:
    """
    Download a single tile from satellite imagery provider.
    
    Args:
        tile_x: Tile X coordinate
        tile_y: Tile Y coordinate
        zoom: Zoom level
        provider: Provider to use ('esri' or 'google')
        maptype: Map type ('satellite' or 'hybrid')
        retries: Number of retry attempts
        
    Returns:
        PIL Image object
    """
    url = get_tile_url(tile_x, tile_y, zoom, provider, maptype)
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Check if we got an image
            if response.headers.get('content-type', '').startswith('image/'):
                img = Image.open(BytesIO(response.content))
                return img
            else:
                raise ValueError(f"Unexpected content type: {response.headers.get('content-type')}")
                
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(0.5 * (attempt + 1))  # Exponential backoff
                continue
            else:
                raise Exception(f"Failed to download tile ({tile_x}, {tile_y}) after {retries} attempts: {e}")
    
    raise Exception(f"Failed to download tile ({tile_x}, {tile_y})")


def download_satellite_image(
    lat: float,
    lon: float,
    zoom: int = 18,
    tiles_wide: int = 3,
    tiles_high: int = 3,
    output_path: str = "satellite_image.png",
    maptype: str = "satellite",
    provider: str = "esri"
) -> None:
    """
    Download satellite imagery by downloading and stitching multiple tiles.
    
    Args:
        lat: Latitude of the center location
        lon: Longitude of the center location
        zoom: Zoom level (1-20, higher = more zoomed in)
        tiles_wide: Number of tiles horizontally
        tiles_high: Number of tiles vertically
        output_path: Path to save the final image
        maptype: Map type - 'satellite' or 'hybrid'
        provider: Provider to use - 'esri' (default, free) or 'google'
    """
    
    # Get center tile coordinates
    center_tile_x, center_tile_y = deg2num(lat, lon, zoom)
    
    # Calculate tile range
    start_x = center_tile_x - tiles_wide // 2
    end_x = center_tile_x + tiles_wide // 2 + (tiles_wide % 2)
    start_y = center_tile_y - tiles_high // 2
    end_y = center_tile_y + tiles_high // 2 + (tiles_high % 2)
    
    total_tiles = tiles_wide * tiles_high
    print(f"Downloading satellite imagery...")
    print(f"  Location: {lat}, {lon}")
    print(f"  Zoom level: {zoom}")
    print(f"  Tile grid: {tiles_wide}x{tiles_high} ({total_tiles} tiles)")
    print(f"  Map type: {maptype}")
    print(f"  Provider: {provider}")
    print(f"  Center tile: ({center_tile_x}, {center_tile_y})")
    
    # Download all tiles
    tiles = []
    tile_count = 0
    
    for y in range(start_y, end_y):
        tile_row = []
        for x in range(start_x, end_x):
            tile_count += 1
            print(f"  Downloading tile {tile_count}/{total_tiles} ({x}, {y})...", end="\r")
            try:
                tile = download_tile(x, y, zoom, provider, maptype)
                tile_row.append(tile)
            except Exception as e:
                print(f"\n  Error downloading tile ({x}, {y}): {e}")
                # Create a blank tile as placeholder
                tile_row.append(Image.new('RGB', (256, 256), color='gray'))
            time.sleep(0.1)  # Be respectful with requests
        tiles.append(tile_row)
        print()  # New line after each row
    
    # Stitch tiles together
    print("Stitching tiles together...")
    tile_width = tiles[0][0].width if tiles and tiles[0] else 256
    tile_height = tiles[0][0].height if tiles and tiles[0] else 256
    
    final_width = tile_width * tiles_wide
    final_height = tile_height * tiles_high
    final_image = Image.new("RGB", (final_width, final_height))
    
    for row_idx, tile_row in enumerate(tiles):
        for col_idx, tile in enumerate(tile_row):
            x = col_idx * tile_width
            y = row_idx * tile_height
            final_image.paste(tile, (x, y))
    
    # Save the image
    final_image.save(output_path, "PNG")
    file_size = os.path.getsize(output_path)
    print(f"✓ Image saved to: {output_path}")
    print(f"  Resolution: {final_width}x{final_height} pixels")
    print(f"  File size: {file_size:,} bytes")
    
    # Calculate approximate area covered
    top_left_lat, top_left_lon = num2deg(start_x, start_y, zoom)
    bottom_right_lat, bottom_right_lon = num2deg(end_x, end_y, zoom)
    print(f"  Area covered: ~{abs(top_left_lat - bottom_right_lat):.6f}° lat × ~{abs(top_left_lon - bottom_right_lon):.6f}° lon")


def main():
    parser = argparse.ArgumentParser(
        description="Download high-resolution Google Maps satellite imagery (free, no API key required)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using coordinates
  python get_maps.py --lat 32.882449 --lon -117.2348557 --zoom 20 --output location.png
  
  # Using coordinates string
  python get_maps.py --coordinates "32.882449,-117.2348557" --zoom 18 --tiles 5x5
  
  # High-resolution with more tiles
  python get_maps.py --lat 32.882449 --lon -117.2348557 --zoom 20 --tiles 10x10 --output highres.png
  
  # Hybrid view (satellite with labels)
  python get_maps.py --lat 32.882449 --lon -117.2348557 --zoom 18 --maptype hybrid --output hybrid.png

Note: This script accesses publicly available map tiles. Use responsibly and in
accordance with Google's Terms of Service. For production use, consider using
the official Google Maps API.
        """
    )
    
    # Location input
    location_group = parser.add_mutually_exclusive_group(required=False)
    location_group.add_argument(
        "--coordinates", "-c",
        type=str,
        metavar="LAT,LON",
        help="Coordinates as 'latitude,longitude' (e.g., '32.882449,-117.2348557')"
    )
    
    parser.add_argument(
        "--lat",
        type=float,
        help="Latitude (use with --lon)"
    )
    parser.add_argument(
        "--lon",
        type=float,
        help="Longitude (use with --lat)"
    )
    
    parser.add_argument(
        "--zoom", "-z",
        type=int,
        default=18,
        help="Zoom level (1-20, default: 18). Higher = more zoomed in. Max 20 for satellite."
    )
    
    parser.add_argument(
        "--tiles",
        type=str,
        default="3x3",
        metavar="WxH",
        help="Number of tiles to download (format: WIDTHxHEIGHT, default: 3x3). More tiles = larger area."
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="satellite_image.png",
        help="Output file path (default: satellite_image.png)"
    )
    
    parser.add_argument(
        "--maptype", "-m",
        type=str,
        choices=["satellite", "hybrid"],
        default="satellite",
        help="Map type: 'satellite' (pure satellite) or 'hybrid' (satellite with labels, default: satellite)"
    )
    
    parser.add_argument(
        "--provider", "-p",
        type=str,
        choices=["esri", "google"],
        default="esri",
        help="Imagery provider: 'esri' (Esri World Imagery, free, default) or 'google' (may have restrictions)"
    )
    
    args = parser.parse_args()
    
    # Get coordinates
    if args.coordinates:
        try:
            lat, lon = map(float, args.coordinates.split(","))
        except ValueError:
            print("Error: Invalid coordinates format. Use 'latitude,longitude'")
            sys.exit(1)
    elif args.lat is not None and args.lon is not None:
        lat, lon = args.lat, args.lon
    else:
        # Default to the provided coordinates
        lat, lon = 32.882449, -117.2348557
        print(f"No coordinates provided, using default: {lat}, {lon}")
    
    # Parse tile dimensions
    try:
        tiles_wide, tiles_high = map(int, args.tiles.split("x"))
        if tiles_wide < 1 or tiles_high < 1:
            raise ValueError("Tile dimensions must be positive")
    except (ValueError, AttributeError):
        print(f"Error: Invalid tile format '{args.tiles}'. Use format 'WIDTHxHEIGHT' (e.g., '3x3')")
        sys.exit(1)
    
    # Validate zoom level
    if not (1 <= args.zoom <= 20):
        print("Error: Zoom level must be between 1 and 20")
        sys.exit(1)
    
    # Download image
    try:
        download_satellite_image(
            lat, lon,
            zoom=args.zoom,
            tiles_wide=tiles_wide,
            tiles_high=tiles_high,
            output_path=args.output,
            maptype=args.maptype,
            provider=args.provider
        )
    except KeyboardInterrupt:
        print("\n\nDownload cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
