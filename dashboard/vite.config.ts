/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // The coordinator API. Dev proxy so the browser talks same-origin.
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ""),
        ws: true,
      },
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["tests/setup.ts"],
    css: false,
    // Explicit allowlist: vitest collects only its own tests. Playwright
    // directories (e2e/, e2e-live/, any future e2e-*/) can never be collected,
    // so adding one is no longer a landmine — the exclude-list approach broke
    // twice (dashboard merge, then e2e-live/).
    include: ["tests/**/*.{test,spec}.{ts,tsx}", "src/**/*.{test,spec}.{ts,tsx}"],
  },
});
