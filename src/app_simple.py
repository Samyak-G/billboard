# src/app_simple.py - Version without database dependency
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime

app = FastAPI(title="Billboard Backend - Simple Mode")

# Simple in-memory storage
reports = []

class ReportCreate(BaseModel):
    user_id: str | None = None
    storage_key: str
    lat: float | None = None
    lon: float | None = None
    timestamp: str | None = None

@app.get("/")
async def root():
    return {
        "message": "Billboard API is running!",
        "mode": "simple",
        "reports_count": len(reports),
        "status": "healthy"
    }

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

@app.post("/reports", status_code=202)
async def create_report(r: ReportCreate):
    report_id = len(reports) + 1
    report = {
        "id": report_id,
        "user_id": r.user_id,
        "storage_key": r.storage_key,
        "lat": r.lat,
        "lon": r.lon,
        "timestamp": r.timestamp or datetime.now().isoformat(),
        "created_at": datetime.now().isoformat()
    }
    reports.append(report)
    return {"report_id": report_id}

@app.get("/reports")
async def get_reports():
    return {"reports": reports, "count": len(reports)}

@app.get("/reports/{report_id}")
async def get_report(report_id: int):
    for report in reports:
        if report["id"] == report_id:
            return report
    raise HTTPException(status_code=404, detail="Report not found")