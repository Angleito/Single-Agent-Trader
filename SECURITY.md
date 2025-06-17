# Security Policy

## Reporting Security Vulnerabilities

If you discover a security vulnerability in this project, please report it privately to maintain the security of users. Do not create public issues for security vulnerabilities.

## Security Best Practices

### Environment Variables

1. **Never commit `.env` files** with real credentials
2. Use `example.env` as a template with placeholder values
3. Store sensitive credentials securely using:
   - Environment variables on your host system
   - Secret management services (AWS Secrets Manager, HashiCorp Vault)
   - Encrypted configuration files (not in repository)

### API Keys and Secrets

- All API keys should be stored as environment variables
- Use `pydantic.SecretStr` for handling sensitive strings in code
- Never log sensitive information
- Rotate API keys regularly

### Docker Security

1. **Container Security**:
   - Containers run as non-root user
   - Read-only root filesystem where possible
   - No new privileges flag enabled
   - Resource limits enforced

2. **Network Security**:
   - Use internal Docker networks
   - Expose only necessary ports
   - Enable TLS for external communications

### Code Security

1. **Input Validation**:
   - All user inputs are validated using Pydantic models
   - SQL injection prevention through parameterized queries
   - Command injection prevention

2. **Error Handling**:
   - No stack traces exposed in production
   - Generic error messages for users
   - Detailed errors only in logs

### Deployment Security

1. **HTTPS Only**:
   - Use TLS certificates for all external endpoints
   - Enable HSTS headers
   - Redirect HTTP to HTTPS

2. **Rate Limiting**:
   - API endpoints are rate-limited
   - WebSocket connections are limited
   - DDoS protection enabled

3. **CORS Configuration**:
   - Restrictive CORS policies
   - Environment-based origin configuration
   - Credentials only when necessary

## Security Checklist

Before deploying to production:

- [ ] All `.env` files are excluded from version control
- [ ] API keys are stored securely
- [ ] Docker containers use security best practices
- [ ] HTTPS is configured with valid certificates
- [ ] Rate limiting is enabled
- [ ] Error messages don't leak sensitive information
- [ ] Logs are sanitized of sensitive data
- [ ] Dependencies are up-to-date
- [ ] Security headers are configured
- [ ] Debug mode is disabled

## Dependency Management

- Regular dependency updates: `poetry update`
- Security scanning: `poetry run safety check`
- License compliance: `poetry run pip-licenses`

## Secure Configuration Example

```bash
# Production deployment
export ENVIRONMENT=production
export SYSTEM__DRY_RUN=true  # Start with paper trading
export CORS_ORIGINS=https://yourdomain.com
export LOG_LEVEL=warning

# Use secure Docker compose
docker-compose -f docker-compose.secure.yml up -d
```