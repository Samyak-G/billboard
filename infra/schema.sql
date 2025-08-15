-- Extensions (run once)
create extension if not exists postgis;
create extension if not exists pgcrypto;

-- Users (minimal)
create table if not exists users (
  id uuid primary key default gen_random_uuid(),
  display_name text,
  email text unique,
  role text default 'user', -- user | reviewer | admin
  created_at timestamptz default now()
);

-- Reports
create table if not exists reports (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references users(id) on delete set null,
  storage_key text not null, -- e.g. reports/<uuid>.jpg
  lat double precision,
  lon double precision,
  geom geometry(Point,4326), -- created from lat/lon on insert
  timestamp timestamptz default now(),
  status text default 'pending', -- pending, processing, processed, needs_review, auto_flagged
  verdict jsonb, -- final verdict + reason codes
  notes text, -- additional notes from reporter
  created_at timestamptz default now()
);

-- Use trigger/function to set geom from lon/lat
create or replace function set_report_geom() returns trigger language plpgsql as $$
begin
  if (new.lon is not null and new.lat is not null) then
    new.geom := ST_SetSRID(ST_MakePoint(new.lon, new.lat), 4326);
  end if;
  return new;
end;
$$;

create trigger trg_set_report_geom before insert or update on reports
for each row execute function set_report_geom();

-- Detections
create table if not exists detections (
  id uuid primary key default gen_random_uuid(),
  report_id uuid references reports(id) on delete cascade,
  bbox jsonb, -- [x,y,w,h] (image coord space)
  class text,
  conf double precision,
  size_m jsonb, -- {"w":..,"h":..,"conf":..}
  created_at timestamptz default now()
);

-- Zones (polygons) with rules json
create table if not exists zones (
  id serial primary key,
  name text,
  geom geometry(Polygon,4326),
  rules jsonb, -- rules like max_area_m2, proximity restrictions
  created_at timestamptz default now()
);

-- Permits table (pilot)
create table if not exists permits (
  id uuid primary key default gen_random_uuid(),
  owner text,
  geom geometry(Point,4326),
  valid_from date,
  valid_to date,
  allowed_dims jsonb,
  created_at timestamptz default now()
);

-- Audit logs
create table if not exists audit_logs (
  id bigserial primary key,
  report_id uuid references reports(id) on delete cascade,
  event_type text,
  payload jsonb,
  created_at timestamptz default now()
);

-- Indexes
create index if not exists idx_reports_geom on reports using gist(geom);
create index if not exists idx_zones_geom on zones using gist(geom);
create index if not exists idx_permits_geom on permits using gist(geom);
create index if not exists idx_detections_report_id on detections(report_id);
create index if not exists idx_reports_created_at on reports(created_at);
