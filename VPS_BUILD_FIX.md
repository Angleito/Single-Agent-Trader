# VPS Build Fix

The simplified entrypoint scripts are already in the correct locations:
- `/scripts/docker-entrypoint-simple.sh` - For ai-trading-bot
- `/services/docker-entrypoint-bluefin-simple.sh` - For bluefin-service

If you're getting "file not found" errors, please run:
```bash
git pull origin main
```

Then rebuild:
```bash
docker-compose build --no-cache ai-trading-bot bluefin-service
docker-compose up -d ai-trading-bot bluefin-service
```

The scripts were created in a previous commit but may not have been pulled to your VPS yet.
