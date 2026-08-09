/**
 * AUDIT: the live path. Configure a real DP run (5 clients, 3 rounds) through
 * the form, watch the console render from a REAL WebSocket fed by REAL
 * training, and assert every element of the audit checklist that the mocked
 * e2e cannot prove. Charts need two aggregated rounds before recharts draws
 * a path, so curve assertions run at completion, not mid-stream.
 */
import { expect, test } from "@playwright/test";

test("live: configure, train, watch topology/curves/meter, complete", async ({ page }) => {
  await page.goto("/");

  await page.getByRole("button", { name: "Configure" }).click();
  await expect(page.getByLabel("Dataset")).toBeVisible();

  await page.getByLabel("Client population").fill("5");
  await page.getByLabel("Rounds").fill("3");
  await page.getByLabel("Algorithm").selectOption("dp-fedavg");

  await page.getByRole("button", { name: "Start run" }).click();

  // The console takes over; the stream status is explicit.
  await expect(page.getByText(/stream open|connecting/)).toBeVisible({ timeout: 20_000 });

  // run_started arrived over the real WS: topology draws all five clients.
  await expect(
    page.getByRole("group", { name: /Client topology: 5 clients/ }),
  ).toBeVisible({ timeout: 60_000 });

  // Real rounds advance (TFF DP setup makes round 1 slow; be generous).
  await expect(page.getByText("003", { exact: true })).toBeVisible({ timeout: 300_000 });

  // run_completed closes the stream cleanly and the console says so.
  await expect(page.getByText("status completed")).toBeVisible({ timeout: 180_000 });
  await expect(page.getByText(/stream closed/)).toBeVisible();

  // Curves drew from real round_aggregated events (2+ points => paths).
  // .last() dodges a degenerate 1px-wide first path (see audit notes).
  await expect(page.locator("path.recharts-curve").last()).toBeVisible();
  expect(await page.locator("path.recharts-curve").count()).toBeGreaterThanOrEqual(2);

  // The event log followed the stream: per-round aggregation lines with ε.
  await expect(page.getByText(/round 3 aggregated/).first()).toBeVisible();

  // The privacy meter moved on a DP run: a nonzero ε readout is on screen.
  await expect(page.getByText("Privacy budget ε")).toBeVisible();
  await expect(page.getByText(/ε [0-9]+\.[0-9]+/).first()).toBeVisible();
});
