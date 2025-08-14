#!/usr/bin/env python3
"""
Database data insertion test for Billboard schema
Run this when you have internet access
"""

import asyncio
import asyncpg
import os
from dotenv import load_dotenv
import uuid
import json
from datetime import datetime

async def insert_sample_users(conn):
    """Insert sample users first, or get existing ones"""
    sample_users = [
        {
            "display_name": "Test User 1",
            "email": "user1@example.com",
            "role": "user"
        },
        {
            "display_name": "Test User 2", 
            "email": "user2@example.com",
            "role": "user"
        },
        {
            "display_name": "Admin User",
            "email": "admin@example.com", 
            "role": "admin"
        }
    ]
    
    print("Getting/inserting sample users...")
    user_ids = []
    
    for user in sample_users:
        # Try to get existing user first
        existing = await conn.fetchrow(
            "SELECT id FROM users WHERE email = $1", user["email"]
        )
        
        if existing:
            user_ids.append(existing["id"])
            print(f"  ✅ Found existing user: {user['display_name']} (ID: {existing['id']})")
        else:
            # Insert new user
            rec = await conn.fetchrow(
                """INSERT INTO users(display_name, email, role) 
                   VALUES($1, $2, $3) RETURNING id""",
                user["display_name"], user["email"], user["role"]
            )
            user_ids.append(rec["id"])
            print(f"  ✅ Inserted new user: {user['display_name']} (ID: {rec['id']})")
    
    return user_ids

async def insert_sample_reports(conn, user_ids):
    """Insert sample reports with proper schema structure"""
    sample_reports = [
        {
            "user_id": user_ids[0],  # Use actual user ID
            "storage_key": "reports/billboard_delhi_001.jpg", 
            "lat": 28.6139,
            "lon": 77.2090,
            "status": "pending"
        },
        {
            "user_id": user_ids[1],
            "storage_key": "reports/billboard_mumbai_002.jpg",
            "lat": 19.0760,
            "lon": 72.8777,
            "status": "processed"
        },
        {
            "user_id": user_ids[0],
            "storage_key": "reports/billboard_chennai_003.jpg", 
            "lat": 13.0827,
            "lon": 80.2707,
            "status": "pending"
        },
        {
            "user_id": None,  # Anonymous report
            "storage_key": "reports/billboard_jabalpur_004.jpg",
            "lat": 23.1815,
            "lon": 79.9864, 
            "status": "needs_review"
        },
        {
            "user_id": user_ids[2],  # Admin user
            "storage_key": "reports/billboard_bangalore_005.jpg",
            "lat": 12.9716,
            "lon": 77.5946,
            "status": "processed"
        }
    ]
    
    print("Inserting sample reports...")
    inserted_report_ids = []
    
    for report in sample_reports:
        rec = await conn.fetchrow(
            """INSERT INTO reports(user_id, storage_key, lat, lon, status) 
               VALUES($1, $2, $3, $4, $5) RETURNING id""",
            report["user_id"], report["storage_key"], 
            report["lat"], report["lon"], report["status"]
        )
        inserted_report_ids.append(rec["id"])
        print(f"  ✅ Inserted report ID {rec['id']}: {report['storage_key']}")
    
    return inserted_report_ids

async def insert_sample_detections(conn, report_ids):
    """Insert sample detection data"""
    sample_detections = [
        {
            "report_id": report_ids[0],
            "bbox": [120, 80, 200, 150],  # [x, y, width, height]
            "class": "billboard",
            "conf": 0.95,
            "size_m": {"w": 12.5, "h": 8.0, "conf": 0.85}
        },
        {
            "report_id": report_ids[1], 
            "bbox": [50, 60, 180, 120],
            "class": "billboard",
            "conf": 0.88,
            "size_m": {"w": 10.0, "h": 6.0, "conf": 0.75}
        },
        {
            "report_id": report_ids[2],
            "bbox": [200, 150, 250, 180],
            "class": "billboard", 
            "conf": 0.92,
            "size_m": {"w": 15.0, "h": 10.0, "conf": 0.90}
        }
    ]
    
    print("Inserting sample detections...")
    
    for detection in sample_detections:
        rec = await conn.fetchrow(
            """INSERT INTO detections(report_id, bbox, class, conf, size_m) 
               VALUES($1, $2, $3, $4, $5) RETURNING id""",
            detection["report_id"], json.dumps(detection["bbox"]), 
            detection["class"], detection["conf"], json.dumps(detection["size_m"])
        )
        print(f"  ✅ Inserted detection ID {rec['id']} for report {detection['report_id']}")

async def test_db_and_insert_data():
    load_dotenv()
    DATABASE_URL = os.getenv("DATABASE_URL")
    
    if not DATABASE_URL:
        print("❌ DATABASE_URL not found in .env file")
        return False
    
    print(f"Testing connection to: {DATABASE_URL[:50]}...")
    print("=" * 60)
    
    try:
        # Test connection
        conn = await asyncpg.connect(DATABASE_URL)
        print("✅ Database connection successful!")
        
        # Test PostGIS extension
        postgis_version = await conn.fetchval("SELECT PostGIS_version()")
        print(f"✅ PostGIS version: {postgis_version}")
        
        # Check existing data
        user_count = await conn.fetchval("SELECT COUNT(*) FROM users")
        report_count = await conn.fetchval("SELECT COUNT(*) FROM reports") 
        detection_count = await conn.fetchval("SELECT COUNT(*) FROM detections")
        
        print(f"📊 Current data: {user_count} users, {report_count} reports, {detection_count} detections")
        
        # Insert sample data
        print("\n🔄 Inserting sample data...")
        
        # Insert users first
        user_ids = await insert_sample_users(conn)
        
        # Insert reports
        report_ids = await insert_sample_reports(conn, user_ids)
        
        # Insert detections for some reports
        await insert_sample_detections(conn, report_ids[:3])
        
        # Verify final counts
        final_user_count = await conn.fetchval("SELECT COUNT(*) FROM users")
        final_report_count = await conn.fetchval("SELECT COUNT(*) FROM reports")
        final_detection_count = await conn.fetchval("SELECT COUNT(*) FROM detections")
        
        print(f"\n� Final data: {final_user_count} users, {final_report_count} reports, {final_detection_count} detections")
        
        # Show sample of inserted data
        reports = await conn.fetch("""
            SELECT r.id, r.storage_key, r.lat, r.lon, r.status, u.display_name 
            FROM reports r 
            LEFT JOIN users u ON r.user_id = u.id 
            ORDER BY r.created_at DESC 
            LIMIT 5
        """)
        
        print("\n📋 Sample reports:")
        for report in reports:
            user_name = report['display_name'] or 'Anonymous'
            print(f"  {report['storage_key']} - {user_name} - ({report['lat']}, {report['lon']}) - {report['status']}")
        
        await conn.close()
        print("\n🎉 Database test and data insertion completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Database operation failed: {e}")
        return False

if __name__ == "__main__":
    print("🗄️  Billboard Database Test & Data Insertion")
    print("="*60)
    
    success = asyncio.run(test_db_and_insert_data())
    
    if success:
        print("\n🔧 Next Steps:")
        print("1. Start your server:")
        print("   python3 -m uvicorn src.app:app --reload --port 8000")
        print("\n2. Test the API:")
        print("   curl http://localhost:8000/")
        print("   curl http://localhost:8000/reports")
        print("\n3. Check your Supabase dashboard to see the data!")
    else:
        print("\n💡 Check internet connectivity and DATABASE_URL configuration")