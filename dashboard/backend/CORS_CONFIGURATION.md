# CORS Configuration Documentation

## Overview
This document describes the CORS (Cross-Origin Resource Sharing) configuration for the AI Trading Bot Dashboard backend.

## Configuration Details

### Allowed Origins
- `http://localhost:3000` - Frontend development server (React/Next.js)
- `http://127.0.0.1:3000` - Alternative localhost address
- `http://localhost:8000` - Backend self-requests
- `*` - All origins (for development, should be restricted in production)

### Allowed Methods
- GET
- POST
- PUT
- DELETE
- OPTIONS

### Allowed Headers
- Accept
- Accept-Language
- Content-Language
- Content-Type
- Authorization
- X-Requested-With
- Origin
- Access-Control-Request-Method
- Access-Control-Request-Headers
- Cache-Control
- Pragma

### Exposed Headers
- Content-Type
- Content-Length
- Access-Control-Allow-Origin
- Access-Control-Allow-Headers

## Additional Features

### JSON Response Middleware
- Automatically sets `Content-Type: application/json` for API and UDF endpoints
- Adds security headers (X-Content-Type-Options, X-Frame-Options, X-XSS-Protection)
- Handles WebSocket upgrade CORS headers

### OPTIONS Handler
- Explicit OPTIONS endpoint handler for CORS preflight requests
- Responds with proper CORS headers and 200 status
- Sets Access-Control-Max-Age for caching preflight responses

### WebSocket CORS Support
- WebSocket connections accept any origin in development
- Logs origin information for debugging
- Includes CORS headers for WebSocket upgrade requests

## Security Considerations

### Development vs Production
- Current configuration allows all origins (`*`) for development ease
- In production, remove `*` and only allow specific trusted origins
- Consider implementing origin validation for WebSocket connections

### Headers Security
- X-Content-Type-Options: nosniff - Prevents MIME type sniffing
- X-Frame-Options: DENY - Prevents clickjacking attacks
- X-XSS-Protection: 1; mode=block - Enables XSS filtering

## Testing
Use the provided `test_cors.py` script to verify CORS configuration:

```bash
python test_cors.py
```

## Troubleshooting

### Common Issues
1. **CORS Error in Browser**: Check browser console for specific error message
2. **Preflight Failures**: Verify OPTIONS handler is working correctly
3. **WebSocket Connection Issues**: Check origin header and connection upgrade headers

### Browser Console Errors
- `Access to fetch at 'http://localhost:8000/api/status' from origin 'http://localhost:3000' has been blocked by CORS policy`
  - Solution: Ensure localhost:3000 is in allowed origins list

### Debugging
- Check server logs for origin information in WebSocket connections
- Use browser developer tools Network tab to inspect CORS headers
- Test with curl to verify server responses:

```bash
curl -H "Origin: http://localhost:3000" -v http://localhost:8000/api/status
```

## Production Deployment

### Required Changes for Production
1. Remove `*` from allowed origins
2. Add specific production domain to allowed origins
3. Consider implementing API key authentication
4. Enable HTTPS and update origins to use https://
5. Implement proper WebSocket origin validation

### Example Production Configuration
```python
allow_origins=[
    "https://your-dashboard-domain.com",
    "https://api.your-domain.com"
],
```
