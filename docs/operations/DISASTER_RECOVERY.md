# AI Factory v4 Disaster Recovery Runbook

This Standard Operating Procedure (SOP) defines the disaster recovery (DR) protocols, recovery point/time objectives, and verified procedures for restoring PostgreSQL, Redis, and S3-compatible storage tiers.

---

## 1. Disaster Recovery Objectives

| Metric | Target | Description |
|---|---|---|
| **RPO (Recovery Point Objective)** | < 5 minutes | Maximum acceptable data loss window during a disaster scenario. |
| **RTO (Recovery Time Objective)** | < 30 minutes | Maximum acceptable downtime before full operational restoration. |

---

## 2. PostgreSQL Database Recovery

PostgreSQL serves as the canonical, ACID-compliant source of truth for:
- Jobs (`jobs` table)
- Distributed Worker Leases & Registrations (`worker_registrations`)
- Lifecycle Events (`job_events`)
- Media Asset Metadata (`assets`)

### 2.1 Automated Backup Cadence
- **Full Logical Snapshots**: Daily at 02:00 UTC using `infrastructure/postgres/backup.sh`.
- **WAL Archiving**: Continuous archiving to cold object storage (`pg_wal` shipping every 60s).
- **Snapshot Retention**: 30 days locally, 90 days in off-site encrypted bucket.

### 2.2 Point-In-Time Recovery (PITR) Execution
To recover to a specific timestamp `target_time`:
1. Stop the application control plane:
   ```bash
   systemctl stop factory-control-api
   ```
2. Restore the latest clean base backup:
   ```bash
   ./infrastructure/postgres/restore.sh ./backups/postgres/factory_v4_20261102T000000Z.dump
   ```
3. Configure `recovery.signal` and `postgresql.conf`:
   ```ini
   restore_command = 'cp /mnt/wal_archive/%f %p'
   recovery_target_time = '2026-11-02 14:30:00 UTC'
   recovery_target_action = 'promote'
   ```
4. Restart PostgreSQL and monitor recovery progress in logs:
   ```bash
   tail -f /var/log/postgresql/postgresql.log
   ```
5. Run Alembic schema check to confirm version alignment:
   ```bash
   alembic current
   ```

---

## 3. Redis Tier Recovery

Redis provides streaming acceleration, worker heartbeat leases, priority job queues, and pub/sub event multiplexing.

### 3.1 Snapshot & AOF Protocol
- **RDB Snapshots**: Captured hourly via `infrastructure/redis/backup.sh`.
- **Append-Only File (AOF)**: Configured with `appendfsync everysec`.

### 3.2 Recovery from RDB Snapshot
1. Terminate or isolate current Redis daemon:
   ```bash
   redis-cli -h 127.0.0.1 -p 6379 shutdown nosave || true
   ```
2. Replace corrupt `dump.rdb` with verified snapshot:
   ```bash
   ./infrastructure/redis/restore.sh ./backups/redis/dump_latest.rdb
   ```
3. Restart Redis:
   ```bash
   systemctl start redis-server
   ```
4. Validate connectivity and key integrity:
   ```bash
   redis-cli ping
   redis-cli dbsize
   ```
5. Reconnect Control Plane:
   - Control API dynamically re-synchronizes running state from PostgreSQL upon reconnect.

---

## 4. S3 / MinIO Object Storage Recovery

The S3 layer stores binary media artifacts (images, video, audio, checkpoints, tensor models).

### 4.1 Replication Policy
- Cross-region bucket replication configured from primary MinIO to secondary disaster recovery cluster.
- Bucket versioning enabled on `factory-artifacts` bucket to prevent accidental deletion or ransomware corruption.

### 4.2 Restoration Procedure
1. Verify target endpoint connectivity:
   ```bash
   curl -I "${S3_ENDPOINT_URL}/minio/health/live"
   ```
2. Execute automated bucket synchronization from backup directory:
   ```bash
   ./infrastructure/s3/restore.sh ./backups/s3/factory-artifacts_20261102T000000Z
   ```
3. Run asset verification check against PostgreSQL records:
   ```bash
   python3 -m factory.apps.control_api.src.assets.verifier
   ```

---

## 5. Post-Recovery Verification Checklist

- [ ] Control Plane responds HTTP 200 on `/health`.
- [ ] Database migrations validated via `alembic current`.
- [ ] Worker registrations visible in `GET /v1/inspector/workers`.
- [ ] Presigned asset download URLs resolve cleanly.
- [ ] Test job submitted and processed end-to-end (`POST /v1/jobs`).
