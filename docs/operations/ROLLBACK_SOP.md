# AI Factory v4 Production Rollback Standard Operating Procedure (SOP)

This runbook documents the exact procedures for safely rolling back application binaries, container deployments, and database schema migrations during release regressions or incidents.

---

## 1. Rollback Decision Triggers

Initiate a rollback immediately if any of the following occur during or after a deployment:
1. **Health Check Failure**: Control API `/health` fails or returns 5xx for > 60 seconds.
2. **Error Rate Spike**: API 5xx error rate exceeds 0.5% over a 3-minute rolling window.
3. **Queue Latency Degradation**: Job submission P95 latency exceeds 1500ms.
4. **Worker Lease Failure**: Worker heartbeats fail or worker leases cannot be claimed.
5. **Security Contract Violation**: Unauthorized token rejection or authentication regression.

---

## 2. Canary Deployment Rollback

When traffic is routed via progressive canary weights (e.g., 90% stable, 10% canary):
1. **Immediately shift 100% traffic to stable version**:
   ```bash
   # Via edge routing / traffic manager
   ngrok api traffic-policy update --traffic 100%stable,0%canary
   ```
2. **Drain in-flight jobs on canary instances**:
   ```bash
   # Send SIGTERM to allow graceful completion of admitted jobs
   pkill -15 -f "factory_control_api"
   ```
3. **De-register canary instances** from load balancer.

---

## 3. Database Schema Rollback (Alembic)

Database schema migrations are designed with deterministic `downgrade()` steps in `factory/apps/control-api/alembic/versions/`.

### 3.1 Verify Current Migration State
```bash
cd factory/apps/control-api
alembic current
```

### 3.2 Step-by-Step Downgrade Execution
1. **Single Revision Rollback**:
   ```bash
   alembic downgrade -1
   ```
2. **Rollback to Specific Target Revision**:
   ```bash
   # Example: downgrade back to 0002
   alembic downgrade 0002
   ```
3. **Verify Resulting Schema**:
   ```bash
   alembic current
   python3 -c "
   from factory.apps.control_api.src.queue.repository import get_job_repository
   import asyncio
   repo = get_job_repository()
   asyncio.run(repo.initialize())
   print('Schema verification successful')
   "
   ```

---

## 4. Frontend (Studio Web) Rollback

Studio Web uses static SSG with Next.js 15:
1. Revert edge CDN pointer to previous release immutable build hash:
   ```bash
   # If hosted via Vercel / container / static hosting
   npm run deploy:rollback -- --target=<PREVIOUS_IMMUTABLE_HASH>
   ```
2. Invalidate CDN cache:
   ```bash
   curl -X POST https://api.cloudflare.com/client/v4/zones/${ZONE_ID}/purge_cache \
     -H "Authorization: Bearer ${CF_TOKEN}" \
     -d '{"purge_everything":true}'
   ```

---

## 5. Post-Rollback Smoke Test & Certification

Execute the following automated health checks to verify restoration:
1. **Public Health Check**:
   ```bash
   curl -f http://localhost:8000/health
   ```
2. **Authenticated Model Registry Verification**:
   ```bash
   curl -f -H "Authorization: Bearer ${FACTORY_API_KEY}" http://localhost:8000/v1/models
   ```
3. **Queue Submission Verification**:
   ```bash
   curl -f -X POST http://localhost:8000/v1/jobs \
     -H "Authorization: Bearer ${FACTORY_API_KEY}" \
     -H "Content-Type: application/json" \
     -d '{"service":"text","model":"qwen-text","task":"chat_completion","inputs":{"messages":[{"role":"user","content":"test"}]}}'
   ```
4. **Log Analysis**:
   Inspect structured JSON logs for error codes:
   ```bash
   tail -n 100 /var/log/factory/control-api.log | jq 'select(.level == "ERROR")'
   ```
