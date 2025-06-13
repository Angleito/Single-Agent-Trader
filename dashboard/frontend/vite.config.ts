import { defineConfig } from 'vite'

export default defineConfig({
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
  build: {
    target: 'es2020',
    outDir: 'dist'
  }
})