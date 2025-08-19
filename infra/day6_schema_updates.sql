-- Day 6: OCR, Content Moderation, and PII Handling Tables
-- Add these tables to the existing schema

-- OCR text per detection
CREATE TABLE IF NOT EXISTS ocr_texts (
  id SERIAL PRIMARY KEY,
  detection_id UUID NOT NULL REFERENCES detections(id) ON DELETE CASCADE,
  text TEXT,
  confidence FLOAT,
  ocr_engine TEXT DEFAULT 'tesseract', -- tesseract, easyocr
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Content flags for moderation
CREATE TABLE IF NOT EXISTS content_flags (
  id SERIAL PRIMARY KEY,
  detection_id UUID NOT NULL REFERENCES detections(id) ON DELETE CASCADE,
  flag_type TEXT NOT NULL, -- profanity, political, misinformation, sexual, expired-permit-claim
  score FLOAT NOT NULL, -- 0..1 severity / confidence
  details JSONB,
  flagged_by TEXT DEFAULT 'automated', -- automated, reviewer_id
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- PII redaction artifacts
CREATE TABLE IF NOT EXISTS pii_artifacts (
  id SERIAL PRIMARY KEY,
  report_id UUID NOT NULL REFERENCES reports(id) ON DELETE CASCADE,
  artifact_type TEXT NOT NULL, -- face, license_plate, person
  bbox JSONB NOT NULL, -- {x,y,w,h} in image coordinates
  redacted_url TEXT, -- path to blurred/obfuscated image in storage
  confidence FLOAT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Review audit for manual overrides
CREATE TABLE IF NOT EXISTS review_audits (
  id SERIAL PRIMARY KEY,
  report_id UUID NOT NULL REFERENCES reports(id) ON DELETE CASCADE,
  detection_id UUID REFERENCES detections(id) ON DELETE CASCADE,
  reviewer_id UUID REFERENCES users(id) ON DELETE SET NULL,
  action TEXT NOT NULL, -- accept, reject, flag, unflag
  reason TEXT,
  previous_status TEXT,
  new_status TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_ocr_texts_detection_id ON ocr_texts(detection_id);
CREATE INDEX IF NOT EXISTS idx_content_flags_detection_id ON content_flags(detection_id);
CREATE INDEX IF NOT EXISTS idx_content_flags_type ON content_flags(flag_type);
CREATE INDEX IF NOT EXISTS idx_pii_artifacts_report_id ON pii_artifacts(report_id);
CREATE INDEX IF NOT EXISTS idx_review_audits_report_id ON review_audits(report_id);
CREATE INDEX IF NOT EXISTS idx_review_audits_reviewer_id ON review_audits(reviewer_id);
