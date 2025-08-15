# src/app.py - FastAPI Billboard Detection API
import os
import uuid
import json
from datetime import datetime, timezone
from typing import Optional, List
from contextlib import asynccontextmanager

import asyncpg
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
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

# Pydantic Models
class ReportCreate(BaseModel):
    lat: float = Field(..., ge=-90, le=90, description="Latitude")
    lon: float = Field(..., ge=-180, le=180, description="Longitude") 
    storage_key: str = Field(..., description="Supabase storage key (e.g., reports/image.jpg)")
    user_id: Optional[str] = Field(None, description="User ID (optional for citizen reports)")
    notes: Optional[str] = Field(None, description="Additional notes from reporter")

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

class UploadUrlRequest(BaseModel):
    filename: str = Field(..., description="Original filename with extension")
    content_type: str = Field(default="image/jpeg", description="MIME type")

class UploadUrlResponse(BaseModel):
    upload_url: str
    storage_key: str
    expires_in: int = 3600  # 1 hour

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    database: str
    storage: str

# Database connection pool
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
    yield
    # Shutdown
    await db_pool.disconnect()

# FastAPI app
app = FastAPI(
    title="Billboard Detection API",
    description="AI-powered billboard violation detection system for smart cities",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get database connection
async def get_db():
    async with db_pool.pool.acquire() as conn:
        yield conn

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check(db: asyncpg.Connection = Depends(get_db)):
    """Health check endpoint for monitoring"""
    try:
        # Test database
        await db.fetchval("SELECT 1")
        db_status = "healthy"
    except Exception:
        db_status = "unhealthy"
    
    try:
        # Test Supabase storage
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

# Generate signed upload URL
@app.post("/upload-url", response_model=UploadUrlResponse)
async def generate_upload_url(request: UploadUrlRequest):
    """Generate signed URL for mobile app to upload images directly to Supabase Storage"""
    try:
        # Generate unique filename
        file_ext = request.filename.split('.')[-1] if '.' in request.filename else 'jpg'
        unique_filename = f"{uuid.uuid4()}.{file_ext}"
        storage_key = f"reports/{unique_filename}"
        
        # Generate signed upload URL
        # Note: Supabase Python client might not have direct signed URL generation
        # Using REST API approach
        response = supabase.storage.from_("reports").create_signed_upload_url(storage_key)
        
        if not response.get('signed_url'):
            # Fallback: create signed URL using REST API
            bucket = "reports"
            url = f"{SUPABASE_URL}/storage/v1/object/{bucket}/{unique_filename}"
            
            return UploadUrlResponse(
                upload_url=url,
                storage_key=storage_key,
                expires_in=3600
            )
        
        return UploadUrlResponse(
            upload_url=response['signed_url'],
            storage_key=storage_key,
            expires_in=3600
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate upload URL: {str(e)}")

# Create report with metadata
@app.post("/reports", response_model=ReportResponse)
async def create_report(
    report: ReportCreate, 
    background_tasks: BackgroundTasks,
    db: asyncpg.Connection = Depends(get_db)
):
    """Create a new billboard report after image upload"""
    try:
        # Generate report ID
        report_id = str(uuid.uuid4())
        
        # Insert report into database
        query = """
            INSERT INTO reports (id, lat, lon, storage_key, status, created_at, user_id, notes)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING id, lat, lon, storage_key, status, created_at, user_id, notes
        """
        
        created_at = datetime.now(timezone.utc)
        
        row = await db.fetchrow(
            query,
            report_id,
            report.lat,
            report.lon, 
            report.storage_key,
            "pending",
            created_at,
            report.user_id,
            report.notes
        )
        
        # Add audit log
        audit_query = """
            INSERT INTO audit_logs (report_id, event_type, payload, created_at)
            VALUES ($1, $2, $3, $4)
        """
        await db.execute(
            audit_query,
            report_id,
            "report_created",
            json.dumps({"source": "mobile_api", "coordinates": [report.lat, report.lon]}),
            created_at
        )
        
        # Trigger background processing
        background_tasks.add_task(process_report, report_id, report.storage_key)
        
        return ReportResponse(
            id=str(row['id']),
            lat=row['lat'],
            lon=row['lon'],
            storage_key=row['storage_key'],
            status=row['status'],
            created_at=row['created_at'].isoformat(),
            user_id=row['user_id'],
            notes=row['notes']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create report: {str(e)}")

# Get report by ID
@app.get("/reports/{report_id}", response_model=ReportResponse)
async def get_report(report_id: str, db: asyncpg.Connection = Depends(get_db)):
    """Get report details by ID"""
    try:
        query = """
            SELECT id, lat, lon, storage_key, status, created_at, verdict, user_id, notes
            FROM reports 
            WHERE id = $1
        """
        
        row = await db.fetchrow(query, report_id)
        
        if not row:
            raise HTTPException(status_code=404, detail="Report not found")
        
        return ReportResponse(
            id=str(row['id']),
            lat=row['lat'],
            lon=row['lon'],
            storage_key=row['storage_key'],
            status=row['status'],
            created_at=row['created_at'].isoformat(),
            verdict=row['verdict'],
            user_id=row['user_id'],
            notes=row['notes']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get report: {str(e)}")

# List all reports
@app.get("/reports", response_model=List[ReportResponse])
async def list_reports(
    limit: int = 50,
    offset: int = 0,
    status: Optional[str] = None,
    db: asyncpg.Connection = Depends(get_db)
):
    """List reports with pagination and optional status filter"""
    try:
        # Build query with optional status filter
        base_query = """
            SELECT id, lat, lon, storage_key, status, created_at, verdict, user_id, notes
            FROM reports
        """
        
        conditions = []
        params = []
        param_count = 0
        
        if status:
            param_count += 1
            conditions.append(f"status = ${param_count}")
            params.append(status)
        
        if conditions:
            base_query += " WHERE " + " AND ".join(conditions)
        
        base_query += f" ORDER BY created_at DESC LIMIT ${param_count + 1} OFFSET ${param_count + 2}"
        params.extend([limit, offset])
        
        rows = await db.fetch(base_query, *params)
        
        return [
            ReportResponse(
                id=str(row['id']),
                lat=row['lat'],
                lon=row['lon'],
                storage_key=row['storage_key'],
                status=row['status'],
                created_at=row['created_at'].isoformat(),
                verdict=row['verdict'],
                user_id=row['user_id'],
                notes=row['notes']
            )
            for row in rows
        ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list reports: {str(e)}")

# Root endpoint
@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Billboard Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }
