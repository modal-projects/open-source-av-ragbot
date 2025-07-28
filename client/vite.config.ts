import react from "@vitejs/plugin-react-swc";
import { defineConfig, loadEnv } from "vite";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");

  return {
    plugins: [react()],
    base: "/frontend/",
    build: {
      outDir: "dist",
      assetsDir: "assets",
    },
    server: {
      allowedHosts: true,
      proxy: {
        "/offer": {
          target: env.VITE_API_URL,
          changeOrigin: true,
        },
      },
    },
  };
});
