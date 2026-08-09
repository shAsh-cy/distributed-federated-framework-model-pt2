/**
 * AUDIT config: drives the dashboard against a LIVE coordinator API
 * (expects `vite preview` on 4173 proxying /api to 127.0.0.1:8000, and the
 * API serving real runs). No webServer block and no VITE_MOCK — this is the
 * un-mocked path the packaged e2e never exercises.
 */
import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "e2e-live",
  timeout: 420_000,
  expect: { timeout: 30_000 },
  use: {
    baseURL: "http://127.0.0.1:4173",
    trace: "retain-on-failure",
  },
});
