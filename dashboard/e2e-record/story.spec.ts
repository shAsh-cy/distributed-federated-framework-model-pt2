/**
 * RELEASE tooling, not a test: drives story mode with Playwright's recorder on
 * to produce the webm behind docs/story_mode.gif.
 *
 * It needs no backend and starts no run — story mode replays the recorded
 * fixture — so unlike e2e-live/record.spec.ts this one is reproducible on any
 * machine. Run it, then turn the webm into the GIF:
 *
 *   npm run record:story
 *   npm run gif:story          # needs Playwright's bundled ffmpeg
 *
 * The pacing below is deliberate. Each dwell is long enough for the stage's
 * own animation to complete at least once: the notebook rebound is 2.6 s on a
 * 0.7 s repeat, the recorded round window is about 5.4 s, the ink budget fills
 * over 2.4 s and the accuracy curves draw over 2.2 s.
 */
import { expect, test } from "@playwright/test";

test.use({
  video: { mode: "on", size: { width: 1280, height: 800 } },
  viewport: { width: 1280, height: 800 },
});

/** Total runtime lands in the 30–45 s the README GIF wants. */
test("record: story mode for the README", async ({ page }) => {
  await page.goto("/story");
  await expect(page.locator("[data-stage-index]")).toHaveText("01 / 07");
  await page.waitForTimeout(800); // let the shell settle on camera

  // 1. The problem: one full throw at the boundary, plus the rebound.
  await page.waitForTimeout(4200);

  // 2. One round, played out of the recorded stream.
  await page.keyboard.press("ArrowRight");
  await expect(page.locator("[data-stage-index]")).toHaveText("02 / 07");
  await page.waitForTimeout(7000);

  // 3. Heterogeneity: drag the slider both ways, slowly, because this one is
  // the thing a reader is meant to want to touch.
  await page.keyboard.press("ArrowRight");
  await expect(page.locator("[data-stage-index]")).toHaveText("03 / 07");
  await page.waitForTimeout(900);
  const slider = page.getByRole("slider", { name: /How lopsided/i });
  const box = (await slider.boundingBox())!;
  const y = box.y + box.height / 2;
  await page.mouse.move(box.x + box.width * 0.5, y);
  await page.mouse.down();
  for (let step = 0; step <= 24; step += 1) {
    await page.mouse.move(box.x + box.width * (0.5 + 0.48 * (step / 24)), y);
    await page.waitForTimeout(55);
  }
  await page.waitForTimeout(600);
  for (let step = 0; step <= 32; step += 1) {
    await page.mouse.move(box.x + box.width * (0.98 - 0.96 * (step / 32)), y);
    await page.waitForTimeout(55);
  }
  await page.mouse.up();
  // Hand the arrow keys back to the stage rail: while the slider has focus
  // they belong to it, which is the point of the guard in StoryMode.
  await slider.evaluate((node: HTMLElement) => node.blur());
  await page.waitForTimeout(700);

  // 4. The attack.
  await page.keyboard.press("ArrowRight");
  await expect(page.locator("[data-stage-index]")).toHaveText("04 / 07");
  await page.waitForTimeout(3000);

  // 5. The defence: trim, ink, the ochre budget, the attack again.
  await page.keyboard.press("ArrowRight");
  await expect(page.locator("[data-stage-index]")).toHaveText("05 / 07");
  await page.waitForTimeout(4000);
  await page.mouse.wheel(0, 520);
  await page.waitForTimeout(2600);
  await page.mouse.wheel(0, 520);
  await page.waitForTimeout(2000);
  await page.mouse.wheel(0, -1040);

  // 6. What it costs.
  await page.keyboard.press("ArrowRight");
  await expect(page.locator("[data-stage-index]")).toHaveText("06 / 07");
  await page.waitForTimeout(4500);

  // 7. Closing card.
  await page.keyboard.press("ArrowRight");
  await expect(page.locator("[data-stage-index]")).toHaveText("07 / 07");
  await page.waitForTimeout(2600);
});
