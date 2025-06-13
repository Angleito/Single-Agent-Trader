# AI Trading Bot Dashboard Frontend

A modern, real-time dashboard for monitoring AI trading bot performance built with Vite, TypeScript, and TradingView Charting Library.

## Features

- **Real-time Market Data**: Live price updates and market statistics
- **TradingView Charts**: Professional trading charts with technical indicators
- **Bot Status Monitoring**: Real-time bot status and trading actions
- **Position Tracking**: Monitor open positions and P&L
- **Risk Management**: Display risk metrics and portfolio performance
- **Activity Logs**: Real-time activity and error logging
- **Responsive Design**: Works on desktop, tablet, and mobile devices

## Quick Start

### Prerequisites

- Node.js 18.0.0 or higher
- npm or yarn package manager

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

### Development Scripts

```bash
# Development server with hot reload
npm run dev

# Type checking
npm run type-check

# Linting
npm run lint

# Build for production
npm run build
```

## Project Structure

```
src/
├── main.ts           # Application entry point
├── types.ts          # TypeScript type definitions
├── websocket.ts      # WebSocket client for real-time data
├── tradingview.ts    # TradingView chart integration
├── ui.ts             # Dashboard UI management
└── style.css         # Dashboard styles
```

## Configuration

### Environment Variables

The dashboard uses environment variables for configuration:

- `VITE_API_BASE_URL`: Backend API base URL
- `VITE_WS_URL`: WebSocket connection URL
- `VITE_DEFAULT_SYMBOL`: Default trading symbol
- `VITE_REFRESH_INTERVAL`: Data refresh interval (ms)
- `VITE_LOG_LEVEL`: Logging level (debug, info, warn, error)

### TradingView Integration

The dashboard integrates with TradingView Charting Library for professional trading charts:

- Charts load via CDN for easy setup
- Supports real-time data updates
- Custom indicators and studies
- Responsive chart sizing
- Dark theme optimized for trading

## Backend Integration

The frontend connects to the AI trading bot backend via:

### WebSocket Connection

Real-time data streaming for:
- Bot status updates
- Market data (price, volume, changes)
- Trade actions and decisions
- VuManChu indicator values
- Position updates
- Risk metrics

### REST API (optional)

For historical data and configuration:
- `/api/status` - Bot status
- `/api/positions` - Current positions
- `/api/metrics` - Risk metrics
- `/api/history` - Trade history

## Development

### Code Structure

The application is built with a modular architecture:

1. **DashboardApp**: Main application class
2. **DashboardUI**: UI management and updates
3. **DashboardWebSocket**: Real-time data connection
4. **TradingViewChart**: Chart integration
5. **Types**: TypeScript definitions

### Adding Features

1. Define new data types in `types.ts`
2. Update WebSocket message handling in `main.ts`
3. Add UI components in `ui.ts`
4. Update styles in `style.css`

### Styling

The dashboard uses CSS custom properties for theming:

```css
:root {
  --bg-primary: #0d1421;
  --text-primary: #ffffff;
  --accent-primary: #3b82f6;
  /* ... */
}
```

## Deployment

### Production Build

```bash
npm run build
```

The build output in `dist/` can be served by any static file server.

### Docker Deployment

The dashboard can be containerized with a simple Dockerfile:

```dockerfile
FROM nginx:alpine
COPY dist/ /usr/share/nginx/html/
```

### Backend Proxy

For production, configure your web server to proxy API calls:

```nginx
location /api {
    proxy_pass http://backend:8000;
}

location /ws {
    proxy_pass http://backend:8000;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
}
```

## Browser Support

- Chrome/Chromium 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Performance

- **Bundle Size**: Optimized chunks for fast loading
- **Hot Module Replacement**: Instant development updates
- **Tree Shaking**: Dead code elimination
- **Code Splitting**: Separate chunks for TradingView and WebSocket
- **Asset Optimization**: Compressed and minified assets

## Troubleshooting

### Common Issues

1. **TradingView not loading**: Check CDN connection and browser console
2. **WebSocket connection failed**: Verify backend is running on correct port
3. **Charts not updating**: Check WebSocket message format and data flow
4. **Build errors**: Ensure all TypeScript types are properly defined

### Debug Mode

Enable debug logging by setting `VITE_LOG_LEVEL=debug` in your environment.

## License

This project is part of the AI Trading Bot system. See the main project LICENSE file for details.