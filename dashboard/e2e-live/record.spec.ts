/**
 * RELEASE tooling, not a test: drives one real DP run against the live API
 * with Playwright's recorder on, producing the webm behind the README's
 * dashboard GIF. Run manually:
 *   npx playwright test --config playwright.live.config.ts e2e-live/record.spec.ts
 */
import { expect, test } from "@playwright/test";

test.use({
  video: { mode: "on", size: { width: 1280, height: 800 } },
  viewport: { width: 1280, height: 800 },
});

test("record: one live DP run for the README", async ({ page }) => {
  await page.goto("/");
  await page.waitForTimeout(1200); // let the shell settle on camera

  await page.getByRole("button", { name: "Configure" }).click();
  await expect(page.getByLabel("Dataset")).toBeVisible();
  await page.getByLabel("Client population").fill("5");
  await page.getByLabel("Rounds").fill("3");
  await page.getByLabel("Algorithm").selectOption("dp-fedavg");
  await page.waitForTimeout(800);

  await page.getByRole("button", { name: "Start run" }).click();
  await expect(
    page.getByRole("group", { name: /Client topology: 5 clients/ }),
  ).toBeVisible({ timeout: 60_000 });

  // Let the whole run play out on camera: rounds, curves, meter, completion.
  await expect(page.getByText("status completed")).toBeVisible({ timeout: 300_000 });
  await page.waitForTimeout(2500); // hold the completed state
});
