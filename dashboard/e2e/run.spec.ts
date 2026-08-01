/**
 * End to end: configure a two-round-visible run, start it, and assert the
 * topology and curves update from streamed events. The stream is the
 * scripted live-demo fixture (recorded measurements through the real event
 * store) served by MSW in the browser — the full frontend path exercised
 * without a training backend.
 */
import { expect, test } from "@playwright/test";

test("configure, start, and watch topology + curves update from events", async ({ page }) => {
  await page.goto("/");

  // Configure a run. Options are served by the (mocked) capabilities API.
  await page.getByRole("button", { name: "Configure" }).click();
  await expect(page.getByLabel("Dataset")).toBeVisible();
  await expect(page.getByText(/parameters/)).toBeVisible();
  await page.getByRole("button", { name: "Start run" }).click();

  // The console takes over on the started run.
  await expect(page.getByText(/stream open|connecting/)).toBeVisible();

  // run_started arrives: the topology draws all ten clients.
  const topology = page.getByRole("group", { name: /Client topology: 10 clients/ });
  await expect(topology).toBeVisible({ timeout: 15_000 });

  // Round counter passes 002 — at least two full protocol rounds observed.
  await expect(page.getByText("002", { exact: true })).toBeVisible({ timeout: 20_000 });

  // The accuracy curve is drawing: recharts renders line paths.
  await expect(page.locator("path.recharts-curve").first()).toBeVisible();

  // Client activity reaches the event log with real per-client lines.
  await expect(page.getByText(/client-\d+ reported/).first()).toBeVisible();

  // The deadline miss in round 3 is visible as a drop line.
  await expect(page.getByText(/dropped · deadline/).first()).toBeVisible({ timeout: 20_000 });

  // And the run completes with the recorded final accuracy.
  await expect(page.getByText(/run completed · 6 rounds/)).toBeVisible({ timeout: 30_000 });
  await expect(page.getByText(/stream closed/)).toBeVisible();
});
