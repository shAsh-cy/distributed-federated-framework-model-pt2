import { defineConfig } from "@playwright/test";

/**
 * The e2e runs the BUILT app in mock mode: MSW in the browser serves the
 * recorded fixtures and streams the scripted live-demo events (real
 * protocol through the real store, values from recorded measurements).
 * No backend, no training — per the machine constraint documented in the PR.
 */
export default defineConfig({
  testDir: "e2e",
  timeout: 60_000,
  use: { baseURL: "http://127.0.0.1:4173" },
  webServer: {
    command: "npm run e2e:serve",
    port: 4173,
    env: { VITE_MOCK: "1" },
    reuseExistingServer: false,
    timeout: 120_000,
  },
});
