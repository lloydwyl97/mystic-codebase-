import react from '@vitejs/plugin-react';
import dns from 'node:dns';
import path from 'path';
import { defineConfig } from 'vite';

// Fix DNS resolution issues
dns.setDefaultResultOrder('verbatim');

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    host: 'localhost',
    open: false, // Don't auto-open browser
    strictPort: false,
    hmr: {
      overlay: false, // Disable error overlay
    },
    proxy: {
      '/api': {
        target: process.env.VITE_API_URL || 'http://127.0.0.1:8000',
        changeOrigin: true,
        secure: false,
        ws: true, // Enable WebSocket proxying for API routes
      },
      '/health': {
        target: process.env.VITE_API_URL || 'http://127.0.0.1:8000',
        changeOrigin: true,
        secure: false,
      },
      '/ws': {
        target: process.env.VITE_API_URL || 'http://127.0.0.1:8000',
        changeOrigin: true,
        secure: false,
        ws: true,
      }
    },
    fs: {
      strict: false
    },
    cors: {
      origin: ['http://localhost:3000', 'http://127.0.0.1:3000', 'http://localhost:5173'],
      credentials: true
    },
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization'
    }
  },
  optimizeDeps: {
    include: [
      'react',
      'react-dom'
    ],
    exclude: [
      '@mui/icons-material',
      '@mui/material',
      '@emotion/react',
      '@emotion/styled'
    ],
    force: false,
    esbuildOptions: {
      target: 'es2020',
      minify: false
    }
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
    minify: false, // Disable minification for faster processing
    terserOptions: {
      compress: {
        drop_console: false,
        drop_debugger: false
      }
    },
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom']
        },
        chunkFileNames: 'assets/[name]-[hash].js',
        entryFileNames: 'assets/[name]-[hash].js',
        assetFileNames: 'assets/[name]-[hash].[ext]'
      },
      external: []
    },
    chunkSizeWarningLimit: 2000
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    }
  },
  define: {
    'process.env.NODE_ENV': '"development"',
    'process.env.VITE_API_URL': JSON.stringify(process.env.VITE_API_URL || 'http://127.0.0.1:8000')
  },
  esbuild: {
    target: 'es2020',
    minify: false
  },
  // Minimize file watching and processing
  clearScreen: false,
  logLevel: 'error', // Only show errors
  // Disable aggressive file watching
  watch: {
    ignored: ['**/node_modules/**', '**/dist/**', '**/.git/**']
  }
});
