import { defineConfig } from 'vite'
import { resolve } from 'path'
import { VitePWA } from 'vite-plugin-pwa'

export default defineConfig({
  plugins: [
    VitePWA({
      registerType: 'autoUpdate',
      workbox: {
        globPatterns: ['**/*.{js,css,html,ico,png,svg}'],
        runtimeCaching: [
          {
            urlPattern: /^https:\/\/api\./,
            handler: 'NetworkFirst',
            options: {
              cacheName: 'api-cache',
              cacheableResponse: {
                statuses: [0, 200]
              }
            }
          }
        ]
      },
      manifest: {
        name: 'AI Trading Bot Dashboard',
        short_name: 'TradingBot',
        description: 'Professional AI-powered cryptocurrency trading bot dashboard',
        theme_color: '#1a1a1a',
        background_color: '#000000',
        display: 'standalone',
        icons: [
          {
            src: '/icons/icon-192x192.png',
            sizes: '192x192',
            type: 'image/png',
            purpose: 'any maskable'
          },
          {
            src: '/icons/icon-512x512.png',
            sizes: '512x512',
            type: 'image/png',
            purpose: 'any maskable'
          }
        ]
      }
    })
  ],
  server: {
    port: 3000,
    host: '0.0.0.0',
    cors: true,
    hmr: {
      port: 3000
    }
  },
  clearScreen: false,
  logLevel: 'info',
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src')
    }
  },
  build: {
    target: 'es2020',
    outDir: 'dist',
    // Enable source maps for debugging in production
    sourcemap: false,
    // Minimize bundle size
    minify: 'esbuild',
    // Enable compression
    reportCompressedSize: true,
    // Chunk size warning limit
    chunkSizeWarningLimit: 1000,
    // Advanced optimization options
    rollupOptions: {
      // Code splitting configuration
      output: {
        // Automatic chunk splitting
        manualChunks: undefined,
        // Optimized chunk file names
        chunkFileNames: (chunkInfo) => {
          const facadeModuleId = chunkInfo.facadeModuleId ?
            chunkInfo.facadeModuleId.split('/').pop()?.replace('.ts', '') : 'chunk';
          return `js/${facadeModuleId}-[hash].js`;
        },
        entryFileNames: 'js/[name]-[hash].js',
        assetFileNames: (assetInfo) => {
          const info = assetInfo.name?.split('.') ?? [];
          const ext = info[info.length - 1];
          if (/png|jpe?g|svg|gif|tiff|bmp|ico/i.test(ext ?? '')) {
            return `img/[name]-[hash].${ext}`;
          }
          if (/css/i.test(ext ?? '')) {
            return `css/[name]-[hash].${ext}`;
          }
          return `assets/[name]-[hash].${ext}`;
        }
      },
      // External dependencies to exclude from bundle
      external: (id) => {
        // Don't externalize any dependencies for this dashboard
        return false;
      },
      // Tree-shaking optimization
      treeshake: {
        moduleSideEffects: true
      }
    },
    // CSS optimization
    cssCodeSplit: true,
    cssMinify: true,
    // Asset optimization
    assetsInlineLimit: 4096, // 4kb
    // Enable gzip compression
    assetsDir: 'assets'
  },
  // Optimization options
  esbuild: {
    // Remove console.log in production
    drop: process.env.NODE_ENV === 'production' ? ['console', 'debugger'] : [],
    // Minify identifiers
    minifyIdentifiers: true,
    // Minify syntax
    minifySyntax: true,
    // Minify whitespace
    minifyWhitespace: true,
    // Target modern browsers for smaller bundle
    target: 'es2020',
    // Enable tree-shaking
    treeShaking: true
  },
  // Performance optimizations
  optimizeDeps: {
    // No external dependencies to optimize
    include: [],
    force: true
  },
  // Define environment variables for optimization
  define: {
    // Remove development-only code in production
    __DEV__: process.env.NODE_ENV !== 'production',
    __PROD__: process.env.NODE_ENV === 'production'
  }
})
