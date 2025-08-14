# src/app.py
import os
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncpg
from dotenv import load_dotenv
import uuid

load_dotenv()  # expects DATABASE_URL in .env

DATABASE_URL = os.getenv("DATABASE_URL")
app = FastAPI(title="Billboard Backend (MVP)")

class ReportCreate(BaseModel):
    user_id: str | None = None
    storage_key: str
    lat: float | None = None
    lon: float | None = None
    timestamp: str | None = None

@app.get("/")
async def root():
    return {
        "message": "Billboard API with Database", 
        "database_url": DATABASE_URL[:50] + "..." if DATABASE_URL else "Not configured",
        "status": "connected" if hasattr(app.state, 'db') else "connecting"
    }

@app.get("/health")
async def health():
    try:
        async with app.state.db.acquire() as conn:
            result = await conn.fetchval("SELECT 1")
            return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "database": f"error: {str(e)}"}

@app.get("/reports")
async def get_reports():
    async with app.state.db.acquire() as conn:
        records = await conn.fetch("""
            SELECT r.id, r.storage_key, r.lat, r.lon, r.status, r.created_at,
                   u.display_name as user_name
            FROM reports r 
            LEFT JOIN users u ON r.user_id = u.id 
            ORDER BY r.created_at DESC
        """)
        reports = [dict(record) for record in records]
        # Convert UUID to string for JSON serialization
        for report in reports:
            report['id'] = str(report['id'])
        return {"reports": reports, "count": len(reports)}

@app.on_event("startup")
async def startup():
    app.state.db = await asyncpg.create_pool(DATABASE_URL)

@app.on_event("shutdown")
async def shutdown():
    await app.state.db.close()

@app.post("/reports", status_code=202)
async def create_report(r: ReportCreate):
    async with app.state.db.acquire() as conn:
        rec = await conn.fetchrow(
            """INSERT INTO reports(storage_key, lat, lon, status) 
               VALUES($1, $2, $3, $4) RETURNING id""",
            r.storage_key, r.lat, r.lon, 'pending'
        )
    return {"report_id": str(rec["id"])}

@app.get("/reports/{report_id}")
async def get_report(report_id: str):
    try:
        report_uuid = uuid.UUID(report_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid UUID format")
    
    async with app.state.db.acquire() as conn:
        rec = await conn.fetchrow("""
            SELECT r.*, u.display_name as user_name,
                   ST_X(r.geom) as longitude, ST_Y(r.geom) as latitude
            FROM reports r 
            LEFT JOIN users u ON r.user_id = u.id 
            WHERE r.id = $1
        """, report_uuid)
        if not rec:
            raise HTTPException(status_code=404, detail="Report not found")
        
        result = dict(rec)
        result['id'] = str(result['id'])
        if result.get('user_id'):
            result['user_id'] = str(result['user_id'])
        return result
