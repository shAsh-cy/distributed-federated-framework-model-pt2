/**
 * RELEASE config: records story mode against the BUILT app for the README GIF.
 *
 * Separate from playwright.config.ts so `npm run e2e` never picks the recorder
 * up — it is tooling that produces an artefact, not a test that asserts
 * anything. Separate from playwright.live.config.ts because this one needs no
 * coordinator: story mode replays a committed fixture.
 */
import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "e2e-record",
  timeout: 180_000,
  outputDir: "test-results/record",
  use: { baseURL: "http://127.0.0.1:4173" },
  webServer: {
    command: "npm run e2e:serve",
    port: 4173,
    env: { VITE_MOCK: "1" },
    reuseExistingServer: false,
    timeout: 120_000,
  },
});
