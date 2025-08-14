# src/app.py
import os
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncpg
from dotenv import load_dotenv

load_dotenv()  # expects DATABASE_URL in .env

DATABASE_URL = os.getenv("DATABASE_URL")
app = FastAPI(title="Billboard Backend (MVP)")

class ReportCreate(BaseModel):
    user_id: str | None = None
    storage_key: str
    lat: float | None = None
    lon: float | None = None
    timestamp: str | None = None

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
            "insert into reports(user_id, storage_key, lat, lon, timestamp) values($1,$2,$3,$4,$5) returning id",
            r.user_id, r.storage_key, r.lat, r.lon, r.timestamp
        )
    return {"report_id": rec["id"]}
