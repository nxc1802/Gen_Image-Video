-- ============================================================
-- FLUX.1 Image Generation Queue Schema for Supabase
-- ============================================================

CREATE TABLE IF NOT EXISTS public.image_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    
    -- Request parameters
    prompt TEXT NOT NULL,
    model TEXT NOT NULL DEFAULT 'flux-1-dev',
    size TEXT NOT NULL DEFAULT '1024x1024',
    seed BIGINT,
    steps INT,
    guidance FLOAT,
    response_format TEXT DEFAULT 'b64_json',
    
    -- Queue status: 'pending', 'processing', 'completed', 'failed'
    status TEXT NOT NULL DEFAULT 'pending',
    
    -- Output results
    result_b64 TEXT,
    image_url TEXT,
    error_message TEXT,
    inference_time_sec FLOAT,
    device_name TEXT
);

-- Index for fast queue polling by status and timestamp
CREATE INDEX IF NOT EXISTS idx_image_jobs_queue 
ON public.image_jobs (status, created_at);

-- Enable Row Level Security (RLS)
ALTER TABLE public.image_jobs ENABLE ROW LEVEL SECURITY;

-- Allow read & write access for authenticated & anon with valid API key
CREATE POLICY "Allow public select on image_jobs" 
ON public.image_jobs FOR SELECT USING (true);

CREATE POLICY "Allow public insert on image_jobs" 
ON public.image_jobs FOR INSERT WITH CHECK (true);

CREATE POLICY "Allow public update on image_jobs" 
ON public.image_jobs FOR UPDATE USING (true);

-- Enable Realtime replication for instant push notifications
ALTER PUBLICATION supabase_realtime ADD TABLE public.image_jobs;
