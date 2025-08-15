# src/app.py - FastAPI Billboard Detection API
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import os
import uuid
import json
from datetime import datetime, timezone
from typing import Optional, List
from contextlib import asynccontextmanager

import asyncpg
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from supabase import create_client

# Import background processing from tasks
from .tasks import process_report

# Environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

if not all([SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, DATABASE_URL]):
    raise ValueError("Missing required environment variables")

# Initialize Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# --- Pydantic Models ---
class ReportResponse(BaseModel):
    id: str
    lat: float
    lon: float
    storage_key: str
    status: str
    created_at: str
    verdict: Optional[dict] = None
    user_id: Optional[str] = None
    notes: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    database: str
    storage: str

# --- Database Connection ---
class DatabasePool:
    def __init__(self):
        self.pool = None
    
    async def connect(self):
        self.pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
    
    async def disconnect(self):
        if self.pool:
            await self.pool.close()

db_pool = DatabasePool()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await db_pool.connect()
    
    # Ensure Supabase storage bucket exists
    try:
        buckets = supabase.storage.list_buckets()
        bucket_names = [bucket.name for bucket in buckets]
        
        if 'reports' not in bucket_names:
            supabase.storage.create_bucket('reports', {
                'public': False,
                'allowedMimeTypes': ['image/jpeg', 'image/png', 'image/webp'],
                'fileSizeLimit': 10485760  # 10MB
            })
            print("✅ Created 'reports' storage bucket")
        else:
            print("✅ 'reports' storage bucket already exists")
            
    except Exception as e:
        print(f"⚠️ Storage bucket setup warning: {e}")
    
    yield
    # Shutdown
    await db_pool.disconnect()

# --- FastAPI App ---
app = FastAPI(
    title="Billboard Detection API",
    description="AI-powered billboard violation detection system for smart cities",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def get_db():
    async with db_pool.pool.acquire() as conn:
        yield conn

# --- API Endpoints ---

@app.get("/health", response_model=HealthResponse)
async def health_check(db: asyncpg.Connection = Depends(get_db)):
    """Health check endpoint for monitoring"""
    try:
        await db.fetchval("SELECT 1")
        db_status = "healthy"
    except Exception:
        db_status = "unhealthy"
    
    try:
        supabase.storage.list_buckets()
        storage_status = "healthy"
    except Exception:
        storage_status = "unhealthy"
    
    return HealthResponse(
        status="healthy" if db_status == "healthy" and storage_status == "healthy" else "degraded",
        timestamp=datetime.now(timezone.utc).isoformat(),
        database=db_status,
        storage=storage_status
    )

@app.post("/reports", response_model=ReportResponse)
async def create_report(
    background_tasks: BackgroundTasks,
    db: asyncpg.Connection = Depends(get_db),
    lat: float = Form(...),
    lon: float = Form(...),
    notes: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None), # Must be a valid UUID if provided
    file: UploadFile = File(...)
):
    """Create a new report by uploading an image and its metadata directly."""
    try:
        # 1. Upload file to Supabase Storage
        file_ext = file.filename.split('.')[-1] if '.' in file.filename else 'jpg'
        unique_filename = f"{uuid.uuid4()}.{file_ext}"
        storage_key = f"reports/{unique_filename}"
        
        contents = await file.read()
        
        supabase.storage.from_("reports").upload(
            path=unique_filename,
            file=contents,
            file_options={'content-type': file.content_type}
        )

        # 2. Insert report into database
        report_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc)
        
        query = """
            INSERT INTO reports (id, lat, lon, storage_key, status, created_at, user_id, notes)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING id, lat, lon, storage_key, status, created_at, user_id, notes
        """
        
        # Handle optional UUID for user_id
        db_user_id = uuid.UUID(user_id) if user_id else None

        row = await db.fetchrow(
            query,
            report_id, lat, lon, storage_key, "pending", 
            created_at, db_user_id, notes
        )
        
        # 3. Add audit log
        audit_query = """
            INSERT INTO audit_logs (report_id, event_type, payload, created_at)
            VALUES ($1, $2, $3, $4)
        """
        await db.execute(
            audit_query, report_id, "report_created",
            json.dumps({"source": "api_upload", "coordinates": [lat, lon]}),
            created_at
        )
        
        # 4. Trigger background processing
        background_tasks.add_task(process_report, report_id, storage_key)
        
        return ReportResponse(
            id=str(row['id']),
            lat=row['lat'],
            lon=row['lon'],
            storage_key=row['storage_key'],
            status=row['status'],
            created_at=row['created_at'].isoformat(),
            user_id=str(row['user_id']) if row['user_id'] else None,
            notes=row['notes']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create report: {str(e)}")

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Billboard Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }