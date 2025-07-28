import react from "@vitejs/plugin-react-swc";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  base: "/frontend/",
  build: {
    outDir: "dist",
    assetsDir: "assets",
  },
});
