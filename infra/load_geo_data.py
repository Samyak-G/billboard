

# infra/load_geo_data.py
import os
import csv
import json
import psycopg2
from psycopg2.extras import Json
from shapely.geometry import shape
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def get_db_connection():
    """Establishes a new database connection."""
    conn = psycopg2.connect(DATABASE_URL)
    return conn

def clear_existing_data(conn):
    """Clears existing data from zones and permits tables."""
    with conn.cursor() as cur:
        print("Clearing existing geospatial data...")
        cur.execute("DELETE FROM permits;")
        cur.execute("DELETE FROM zones;")
        print("Done.")

def load_zones(conn, geojson_path):
    """Loads zone polygons from a GeoJSON file into the database."""
    print(f"Loading zones from {geojson_path}...")
    with conn.cursor() as cur:
        with open(geojson_path, 'r') as f:
            data = json.load(f)
            for feature in data['features']:
                geom = shape(feature['geometry'])
                properties = feature['properties']
                name = properties.get('name')
                rules = properties.get('rules')
                
                # Insert data, converting Shapely geometry to WKT for PostGIS
                cur.execute(
                    "INSERT INTO zones (name, rules, geom) VALUES (%s, %s, ST_GeomFromText(%s, 4326))",
                    (name, Json(rules), geom.wkt)
                )
    print(f"Successfully loaded {len(data['features'])} zones.")

def load_permits(conn, csv_path):
    """Loads permits from a CSV file into the database."""
    print(f"Loading permits from {csv_path}...")
    count = 0
    with conn.cursor() as cur:
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                lat = float(row['lat'])
                lon = float(row['lon'])
                allowed_dims = {
                    "w": float(row['allowed_dims_w_m']),
                    "h": float(row['allowed_dims_h_m'])
                }
                
                # Create a PostGIS point geometry from lat/lon
                point_wkt = f"POINT({lon} {lat})"
                
                cur.execute(
                    "INSERT INTO permits (id, owner, valid_from, valid_to, allowed_dims, geom) VALUES (%s, %s, %s, %s, %s, ST_SetSRID(ST_GeomFromText(%s), 4326))",
                    (row['permit_id'], row['owner'], row['valid_from'], row['valid_to'], Json(allowed_dims), point_wkt)
                )
                count += 1
    print(f"Successfully loaded {count} permits.")

def main():
    """Main function to run the data loading process."""
    if not DATABASE_URL:
        print("Error: DATABASE_URL environment variable not set.")
        return

    conn = None
    try:
        conn = get_db_connection()
        clear_existing_data(conn)
        load_zones(conn, 'infra/jabalpur_zones.geojson')
        load_permits(conn, 'infra/jabalpur_permits.csv')
        conn.commit()
        print("\nGeospatial data loading complete.")
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"An error occurred: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    main()

