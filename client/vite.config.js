import { defineConfig } from 'vite'

export default defineConfig({
  base: '/frontend/',
  build: {
    outDir: 'dist',
    assetsDir: 'assets'
  }
}) 