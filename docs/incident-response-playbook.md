# Incident Response Playbook

## Table of Contents
1. [Overview](#overview)
2. [Severity Level Definitions](#severity-level-definitions)
3. [Escalation Procedures](#escalation-procedures)
4. [Emergency Contact Information](#emergency-contact-information)
5. [Recovery Time Objectives](#recovery-time-objectives)
6. [Common Issue Resolutions](#common-issue-resolutions)
7. [Incident Response Runbooks](#incident-response-runbooks)
   - [Service Outage Runbook](#service-outage-runbook)
   - [Data Corruption Runbook](#data-corruption-runbook)
   - [Security Breach Runbook](#security-breach-runbook)
   - [Performance Degradation Runbook](#performance-degradation-runbook)
8. [Post-Mortem Template](#post-mortem-template)
9. [Quick Reference Guide](#quick-reference-guide)

## Overview

This playbook provides structured guidance for responding to incidents affecting the AI Trading Bot system. Follow these procedures to ensure rapid, effective incident response while minimizing financial impact and maintaining system integrity.

### Key Principles
- **Safety First**: Immediately halt trading on any critical incident
- **Preserve Evidence**: Document everything for post-mortem analysis
- **Communicate Early**: Notify stakeholders at the first sign of issues
- **Follow the Process**: Use established procedures, don't improvise during incidents

## Severity Level Definitions

### SEV-1: Critical (System Down)
**Definition**: Complete trading system failure or potential for significant financial loss
- Trading completely halted
- Unable to close open positions
- Data loss or corruption affecting positions
- Security breach with active exploitation
- Exchange API authentication failure

**Response Time**: < 5 minutes
**Resolution Target**: < 1 hour

### SEV-2: Major (Degraded Trading)
**Definition**: Significant functionality impaired but system partially operational
- Intermittent trading failures (>20% failure rate)
- Performance degradation affecting trade execution
- Risk management systems offline
- Memory/learning system unavailable
- Partial data corruption

**Response Time**: < 15 minutes
**Resolution Target**: < 4 hours

### SEV-3: Moderate (Limited Impact)
**Definition**: Non-critical functionality affected, trading continues
- Individual indicator failures
- Logging system issues
- Minor performance degradation (<20% impact)
- UI/monitoring dashboard issues
- Non-critical service failures

**Response Time**: < 30 minutes
**Resolution Target**: < 8 hours

### SEV-4: Minor (Minimal Impact)
**Definition**: Cosmetic or non-functional issues
- Documentation errors
- Non-critical alerts
- Development environment issues
- Minor configuration problems

**Response Time**: < 2 hours
**Resolution Target**: < 24 hours

## Escalation Procedures

### Initial Response Flow
```
1. Detect Issue (Manual/Automated)
   ↓
2. Assess Severity Level
   ↓
3. Execute Immediate Actions
   ↓
4. Notify On-Call Engineer
   ↓
5. Create Incident Channel
   ↓
6. Begin Resolution Process
```

### Escalation Matrix

| Severity | Primary Contact | Escalation (15 min) | Executive (30 min) |
|----------|----------------|---------------------|-------------------|
| SEV-1    | On-Call Eng    | Lead Engineer       | CTO/Risk Officer  |
| SEV-2    | On-Call Eng    | Lead Engineer       | Tech Manager      |
| SEV-3    | Primary Eng    | Senior Engineer     | Team Lead         |
| SEV-4    | Any Engineer   | Team Lead           | N/A               |

### Communication Templates

#### Initial Alert
```
🚨 INCIDENT ALERT - SEV-[X]
Time: [TIMESTAMP]
System: AI Trading Bot
Impact: [Brief description]
Status: Investigating
IC: [Name]
Channel: #incident-[YYYYMMDD-XXX]
```

#### Status Update (Every 30 min for SEV-1/2)
```
📊 INCIDENT UPDATE - SEV-[X]
Time: [TIMESTAMP]
Current Status: [Status]
Actions Taken: [List]
Next Steps: [List]
ETA: [Time estimate]
```

#### Resolution Notice
```
✅ INCIDENT RESOLVED - SEV-[X]
Time: [TIMESTAMP]
Duration: [X hours Y minutes]
Root Cause: [Brief summary]
Impact: [Financial/operational impact]
Post-Mortem: [Scheduled date/time]
```

## Emergency Contact Information

### Internal Contacts
```yaml
on_call_rotation:
  primary: "+1-XXX-XXX-XXXX"
  secondary: "+1-XXX-XXX-XXXX"
  pagerduty: "trading-bot-oncall"

engineering_leads:
  technical_lead: "lead@company.com | +1-XXX-XXX-XXXX"
  infrastructure: "infra@company.com | +1-XXX-XXX-XXXX"
  security: "security@company.com | +1-XXX-XXX-XXXX"

management:
  engineering_manager: "em@company.com | +1-XXX-XXX-XXXX"
  cto: "cto@company.com | +1-XXX-XXX-XXXX"
  risk_officer: "risk@company.com | +1-XXX-XXX-XXXX"
```

### External Contacts
```yaml
exchanges:
  coinbase_support: "support@coinbase.com | 1-888-908-7930"
  bluefin_support: "support@bluefin.io"

vendors:
  aws_support: "AWS Support Center | Business Support"
  openai_support: "support@openai.com"
  monitoring_vendor: "support@datadog.com"

legal_compliance:
  legal_counsel: "legal@lawfirm.com | +1-XXX-XXX-XXXX"
  compliance_officer: "compliance@company.com"
```

## Recovery Time Objectives

### System Components RTO/RPO

| Component | RTO | RPO | Backup Frequency |
|-----------|-----|-----|------------------|
| Trading Engine | 5 min | 0 min | Real-time replication |
| Position Database | 10 min | 1 min | Continuous backup |
| Configuration | 15 min | 5 min | Every 5 minutes |
| Historical Data | 1 hour | 1 hour | Hourly snapshots |
| Memory/Learning System | 30 min | 15 min | Every 15 minutes |
| Monitoring/Logs | 2 hours | 1 hour | Hourly backup |

### Recovery Priority Order
1. **Stop Loss System** - Ensure all positions have stop losses
2. **Position Tracking** - Verify all open positions are accounted for
3. **Trading Engine** - Restore core trading functionality
4. **Risk Management** - Enable position sizing and leverage controls
5. **Market Data Feed** - Restore real-time price feeds
6. **Decision Engine** - Bring LLM and indicators online
7. **Memory System** - Restore learning capabilities
8. **Monitoring** - Re-enable observability

## Common Issue Resolutions

### Quick Fix Procedures

#### 1. Exchange API Authentication Failure
```bash
# Check API key expiration
poetry run python -m bot.utils.check_auth

# Rotate API keys
export EXCHANGE__CDP_API_KEY_NAME="new-key-name"
export EXCHANGE__CDP_PRIVATE_KEY="$(cat new-private-key.pem)"

# Test connection
poetry run python -m bot.exchange.test_connection
```

#### 2. Memory/Database Issues
```bash
# Check memory usage
docker stats ai-trading-bot

# Clear cache and restart
docker-compose exec ai-trading-bot rm -rf /tmp/cache/*
docker-compose restart ai-trading-bot

# Rebuild memory indices
docker-compose exec mcp-memory python rebuild_indices.py
```

#### 3. WebSocket Disconnection
```bash
# Check connection status
poetry run python -m bot.utils.websocket_health

# Force reconnection
docker-compose exec ai-trading-bot kill -USR1 1

# Increase timeout settings
export EXCHANGE__WEBSOCKET_TIMEOUT=60
export EXCHANGE__WEBSOCKET_PING_INTERVAL=20
```

#### 4. High CPU/Memory Usage
```bash
# Identify resource-heavy processes
docker-compose exec ai-trading-bot top

# Adjust resource limits
docker-compose down
# Edit docker-compose.yml resource limits
docker-compose up -d

# Emergency resource cleanup
docker system prune -a --volumes
```

## Incident Response Runbooks

### Service Outage Runbook

#### Detection Indicators
- Monitoring alerts for service unavailability
- Trading engine health check failures
- User reports of access issues
- Exchange connection timeouts

#### Immediate Actions (< 5 minutes)
1. **HALT ALL TRADING**
   ```bash
   # Emergency stop
   docker-compose exec ai-trading-bot touch /tmp/EMERGENCY_STOP

   # Verify positions are closed or protected
   poetry run python -m bot.utils.position_safety_check
   ```

2. **Assess Scope**
   ```bash
   # Check system health
   docker-compose ps
   docker-compose logs --tail=100 ai-trading-bot

   # Verify exchange connectivity
   curl -X GET https://api.exchange.com/api/v3/accounts
   ```

3. **Notify Stakeholders**
   - Send initial alert via established channels
   - Create incident Slack channel
   - Update status page

#### Investigation Steps
1. **Check Infrastructure**
   ```bash
   # System resources
   df -h
   free -m
   iostat -x 1 5

   # Network connectivity
   ping -c 5 api.exchange.com
   traceroute api.exchange.com

   # DNS resolution
   nslookup api.exchange.com
   ```

2. **Review Recent Changes**
   ```bash
   # Git history
   git log --oneline -10

   # Deployment history
   docker-compose logs | grep -i deploy

   # Configuration changes
   git diff HEAD~1 .env config/
   ```

3. **Analyze Logs**
   ```bash
   # Error patterns
   docker-compose logs | grep -E "ERROR|CRITICAL|FATAL"

   # Trading failures
   grep -r "trade.*failed" logs/

   # System logs
   journalctl -u docker --since "1 hour ago"
   ```

#### Recovery Procedures
1. **Service Restart**
   ```bash
   # Graceful restart
   docker-compose stop ai-trading-bot
   docker-compose start ai-trading-bot

   # Force restart if needed
   docker-compose kill ai-trading-bot
   docker-compose up -d ai-trading-bot
   ```

2. **Rollback if Needed**
   ```bash
   # Identify last known good version
   git log --oneline | grep -i "stable"

   # Rollback
   git checkout [COMMIT_HASH]
   docker-compose build
   docker-compose up -d
   ```

3. **Verify Recovery**
   ```bash
   # Health checks
   poetry run python -m bot.utils.health_check

   # Test trade execution (paper mode)
   SYSTEM__DRY_RUN=true poetry run python -m bot.utils.test_trade

   # Monitor for 15 minutes
   watch -n 30 'docker-compose logs --tail=20'
   ```

### Data Corruption Runbook

#### Detection Indicators
- Inconsistent position calculations
- Database integrity check failures
- Unexpected null values or data types
- Checksum mismatches

#### Immediate Actions (< 10 minutes)
1. **Isolate Affected Data**
   ```bash
   # Stop writes to affected tables
   docker-compose exec ai-trading-bot touch /tmp/READONLY_MODE

   # Create immediate backup
   docker-compose exec postgres pg_dump trading_bot > backup_$(date +%Y%m%d_%H%M%S).sql
   ```

2. **Assess Impact**
   ```sql
   -- Check data integrity
   SELECT COUNT(*) FROM positions WHERE entry_price IS NULL OR current_price IS NULL;
   SELECT COUNT(*) FROM trades WHERE status NOT IN ('open', 'closed', 'cancelled');

   -- Verify position totals
   SELECT symbol,
          SUM(quantity) as total_quantity,
          SUM(quantity * entry_price) as total_value
   FROM positions
   WHERE status = 'open'
   GROUP BY symbol;
   ```

#### Recovery Procedures
1. **Data Validation**
   ```python
   # Run integrity checks
   poetry run python -m bot.utils.data_integrity_check

   # Export suspicious records
   poetry run python -m bot.utils.export_corrupted_data --output=/tmp/corrupted_data.json
   ```

2. **Restore from Backup**
   ```bash
   # List available backups
   ls -la /backups/postgres/

   # Restore specific tables
   docker-compose exec postgres psql trading_bot < /backups/positions_backup.sql

   # Verify restoration
   poetry run python -m bot.utils.verify_restoration
   ```

3. **Reconciliation**
   ```python
   # Compare with exchange data
   poetry run python -m bot.utils.reconcile_positions --exchange=coinbase

   # Fix discrepancies
   poetry run python -m bot.utils.apply_reconciliation --dry-run=false
   ```

### Security Breach Runbook

#### Detection Indicators
- Unauthorized API access attempts
- Unusual trading patterns
- Configuration file modifications
- Suspicious network connections
- Anomalous log entries

#### Immediate Actions (< 5 minutes)
1. **EMERGENCY SHUTDOWN**
   ```bash
   # Kill all trading processes
   docker-compose down

   # Revoke API access
   # Coinbase: Dashboard > API > Revoke All Keys
   # Bluefin: Account Settings > API Management > Disable All

   # Preserve evidence
   tar -czf incident_evidence_$(date +%Y%m%d_%H%M%S).tar.gz logs/ data/ .env
   ```

2. **Isolate System**
   ```bash
   # Block outbound connections (except critical)
   sudo iptables -A OUTPUT -j DROP
   sudo iptables -A OUTPUT -d 127.0.0.1 -j ACCEPT

   # Document network state
   netstat -tulpn > /tmp/network_state.txt
   ps aux > /tmp/process_state.txt
   ```

#### Investigation Steps
1. **Access Log Analysis**
   ```bash
   # Check for unauthorized access
   grep -E "401|403|Invalid|Unauthorized" logs/

   # Analyze API usage patterns
   poetry run python -m bot.utils.api_usage_analysis --days=7

   # Review authentication logs
   docker-compose logs | grep -i "auth"
   ```

2. **System Forensics**
   ```bash
   # Check for modified files
   find . -type f -mtime -1 -ls

   # Verify file integrity
   sha256sum -c checksums.txt

   # Review user activities
   last -f /var/log/wtmp
   history | tail -100
   ```

3. **Trading Analysis**
   ```sql
   -- Identify suspicious trades
   SELECT * FROM trades
   WHERE created_at > NOW() - INTERVAL '24 hours'
   AND (
     size > (SELECT AVG(size) * 3 FROM trades) OR
     leverage > max_allowed_leverage OR
     ip_address NOT IN (SELECT ip FROM whitelist)
   );
   ```

#### Recovery Procedures
1. **Security Hardening**
   ```bash
   # Generate new API keys
   openssl rand -hex 32 > new_api_key.txt

   # Update credentials
   poetry run python -m bot.utils.rotate_credentials

   # Enable additional security
   export SECURITY__ENABLE_2FA=true
   export SECURITY__IP_WHITELIST="x.x.x.x,y.y.y.y"
   ```

2. **System Restoration**
   ```bash
   # Clean reinstall
   git clean -fdx
   git checkout main
   poetry install

   # Restore from secure backup
   ./scripts/restore_from_secure_backup.sh

   # Verify system integrity
   poetry run python -m bot.utils.security_audit
   ```

### Performance Degradation Runbook

#### Detection Indicators
- Trade execution latency > 500ms
- CPU usage consistently > 80%
- Memory usage > 90%
- Websocket message queue backlog
- Increased error rates

#### Immediate Actions (< 15 minutes)
1. **Reduce Load**
   ```bash
   # Lower trading frequency
   export TRADING__MIN_TIME_BETWEEN_TRADES=300  # 5 minutes

   # Disable non-critical features
   export MCP_ENABLED=false
   export MONITORING__DETAILED_METRICS=false

   # Reduce position size temporarily
   export RISK__MAX_POSITION_SIZE=0.1  # 10% max
   ```

2. **Performance Diagnostics**
   ```bash
   # CPU profiling
   docker-compose exec ai-trading-bot py-spy top --pid 1

   # Memory profiling
   docker-compose exec ai-trading-bot python -m memory_profiler bot.main

   # I/O analysis
   iotop -b -n 5
   ```

#### Investigation Steps
1. **Identify Bottlenecks**
   ```python
   # Run performance profiler
   poetry run python -m cProfile -o profile.stats bot.main
   poetry run python -m pstats profile.stats

   # Analyze slow queries
   poetry run python -m bot.utils.slow_query_log --threshold=100ms

   # Check cache hit rates
   poetry run python -m bot.utils.cache_analysis
   ```

2. **Resource Analysis**
   ```bash
   # Database connections
   docker-compose exec postgres psql -c "SELECT count(*) FROM pg_stat_activity;"

   # Thread analysis
   docker-compose exec ai-trading-bot python -m bot.utils.thread_dump

   # Network latency
   ping -c 10 api.exchange.com | grep avg
   ```

#### Optimization Procedures
1. **Quick Wins**
   ```bash
   # Clear caches
   docker-compose exec ai-trading-bot redis-cli FLUSHALL

   # Optimize database
   docker-compose exec postgres vacuumdb -z -d trading_bot

   # Restart with increased resources
   docker-compose down
   export DOCKER_MEMORY=4g
   export DOCKER_CPUS=2
   docker-compose up -d
   ```

2. **Configuration Tuning**
   ```python
   # Adjust performance settings
   {
     "performance": {
       "websocket_buffer_size": 10000,
       "indicator_cache_ttl": 300,
       "batch_size": 100,
       "parallel_workers": 4,
       "db_connection_pool": 20
     }
   }
   ```

## Post-Mortem Template

### Incident Summary
```markdown
**Incident ID**: INC-YYYYMMDD-XXX
**Date**: YYYY-MM-DD
**Duration**: X hours Y minutes
**Severity**: SEV-X
**Lead**: [Name]

### Timeline
- **HH:MM** - Initial detection
- **HH:MM** - Incident declared
- **HH:MM** - Root cause identified
- **HH:MM** - Fix implemented
- **HH:MM** - Service restored
- **HH:MM** - Incident closed

### Impact
- **Financial Impact**: $X lost/at risk
- **Trades Affected**: X trades failed/delayed
- **Users Affected**: X users
- **SLA Breach**: Yes/No
```

### Root Cause Analysis
```markdown
### What Happened
[Detailed description of the incident]

### Root Cause
[Technical explanation of the underlying cause]

### Contributing Factors
1. [Factor 1]
2. [Factor 2]
3. [Factor 3]

### Why It Wasn't Caught Earlier
[Explanation of detection gaps]
```

### Lessons Learned
```markdown
### What Went Well
- [Positive aspect 1]
- [Positive aspect 2]

### What Went Poorly
- [Improvement area 1]
- [Improvement area 2]

### Where We Got Lucky
- [Near miss 1]
- [Near miss 2]
```

### Action Items
```markdown
| Action | Owner | Due Date | Priority |
|--------|-------|----------|----------|
| [Action 1] | [Name] | YYYY-MM-DD | P0 |
| [Action 2] | [Name] | YYYY-MM-DD | P1 |
| [Action 3] | [Name] | YYYY-MM-DD | P2 |

### Follow-up
- **Review Date**: YYYY-MM-DD
- **Success Metrics**: [How we'll measure improvement]
```

## Quick Reference Guide

### Emergency Commands Cheatsheet
```bash
# STOP TRADING IMMEDIATELY
docker-compose exec ai-trading-bot touch /tmp/EMERGENCY_STOP

# Check system status
docker-compose ps && docker-compose logs --tail=50

# Backup current state
./scripts/emergency_backup.sh

# View open positions
poetry run python -m bot.utils.show_positions

# Close all positions (USE WITH CAUTION)
CONFIRM=yes poetry run python -m bot.utils.close_all_positions

# Rollback to last stable
git checkout stable-latest && docker-compose up -d

# Contact on-call
./scripts/page_oncall.sh "SEV-1: Trading system down"
```

### Critical File Locations
```
/logs/trading_bot.log          - Main application logs
/data/positions.db             - Position database
/tmp/EMERGENCY_STOP           - Emergency stop flag
/config/production.json        - Production config
/backups/                      - Automated backups
.env                          - Environment configuration
/var/log/incidents/           - Incident logs
```

### Monitoring URLs
- **Grafana Dashboard**: http://monitoring.internal/d/trading-bot
- **Alerts**: http://monitoring.internal/alerts
- **Status Page**: https://status.trading-bot.com
- **Exchange Status**:
  - Coinbase: https://status.coinbase.com
  - Bluefin: https://status.bluefin.io

### Recovery Validation Checklist
- [ ] All services running (`docker-compose ps`)
- [ ] No error logs in last 5 minutes
- [ ] Successful test trade in paper mode
- [ ] All positions accounted for
- [ ] Risk limits enforced
- [ ] Monitoring alerts cleared
- [ ] Stakeholders notified
- [ ] Post-mortem scheduled

---

**Remember**: In any incident, protecting capital is the highest priority. When in doubt, halt trading and escalate.
