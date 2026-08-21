/**
 * Story mode, end to end in a real browser.
 *
 * Two things the unit tests cannot reach. First, that stepping through all
 * seven panels with the keyboard actually renders each one's caption and its
 * key figure — recharts draws nothing under jsdom, so stage 6's chart is only
 * ever really exercised here. Second, that prefers-reduced-motion degrades
 * every stage to a captioned still with no information lost, which is a claim
 * about a media query and therefore about a browser.
 *
 * No backend and no training: story mode replays the recorded fixture.
 */
import { expect, test, type Page } from "@playwright/test";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

// Read rather than import: a JSON import needs an attribute under node's ESM
// loader, and the point is to compare against the committed file anyway.
const figures = JSON.parse(
  readFileSync(fileURLToPath(new URL("../fixtures/story_figures.json", import.meta.url)), "utf-8"),
) as { figures: Record<string, { display: string }> };

const display = (name: string) => figures.figures[name]!.display;

/** Per stage: the words the caption must carry, and the figures it must show. */
const STAGES: { title: RegExp; caption: RegExp; figures: string[] }[] = [
  { title: /The problem/i, caption: /notebooks never leave/i, figures: [] },
  { title: /One round/i, caption: /teacher|student|corrections/i, figures: [] },
  { title: /Not everyone knows everything/i, caption: /a few chapters/i, figures: [] },
  { title: /The nosy teacher/i, caption: /rebuilt a page/i, figures: [] },
  {
    title: /The defence/i,
    caption: /nothing left to read/i,
    figures: ["clipNorm", "noiseZ", "epsilon", "delta", "clippedFinal"],
  },
  {
    title: /What it costs, honestly/i,
    caption: /measured it rather than hiding it/i,
    figures: ["nodpFinal", "dpFinal", "dpCost", "pooled", "longRoundsAcc"],
  },
  { title: /In one paragraph/i, caption: null as unknown as RegExp, figures: ["fashionDense"] },
];

async function stageIndex(page: Page, index: number) {
  await expect(page.locator("[data-stage-index]")).toHaveText(
    new RegExp(`^0${index + 1} / 0${STAGES.length}$`),
  );
}

/**
 * Open the walkthrough and wait for it to be listening. The keyboard shortcuts
 * are attached in an effect, so a keypress sent between navigation and mount
 * is simply dropped — every test therefore steps through this.
 */
async function openStory(page: Page) {
  await page.goto("/story");
  await stageIndex(page, 0);
}

/** Step forward to a zero-based stage from wherever we are, confirming each step. */
async function goToStage(page: Page, target: number) {
  const label = await page.locator("[data-stage-index]").innerText();
  for (let index = Number(label.slice(0, 2)) - 1; index < target; index += 1) {
    await page.keyboard.press("ArrowRight");
    await stageIndex(page, index + 1);
  }
}

async function assertStage(page: Page, index: number) {
  const stage = STAGES[index]!;
  await expect(page.getByRole("heading", { name: stage.title })).toBeVisible();
  await stageIndex(page, index);
  if (stage.caption) {
    const caption = page.locator("[data-story-caption]").first();
    await expect(caption).toBeVisible();
    await expect(caption).toHaveText(stage.caption);
  }
  for (const name of stage.figures) {
    await expect(page.locator(`[data-figure="${name}"]`).first()).toHaveText(display(name));
  }
}

test.describe("story mode", () => {
  test("steps through every stage, each with its caption and its figures", async ({ page }) => {
    await openStory(page);
    for (let index = 0; index < STAGES.length; index += 1) {
      await assertStage(page, index);
      if (index < STAGES.length - 1) await page.keyboard.press("ArrowRight");
    }
    // And back, so the rail is not one-way.
    for (let index = STAGES.length - 1; index > 0; index -= 1) {
      await page.keyboard.press("ArrowLeft");
      await assertStage(page, index - 1);
    }
  });

  test("stage 2 plays the recorded round: cohort, arrivals, the deadline miss", async ({
    page,
  }) => {
    await openStory(page);
    await goToStage(page, 1);
    const topology = page.getByRole("group", { name: /Client topology: 10 clients/ });
    await expect(topology).toBeVisible();
    // The caption narrates; by the end of the window it has reached the merge.
    await expect(page.locator("[data-story-caption]")).toHaveText(/merges|textbook/i, {
      timeout: 20_000,
    });
    // The round the recorded run lost a client in is the round on screen.
    await expect(page.getByText(/is late/i)).toBeVisible({ timeout: 20_000 });
  });

  test("stage 3's slider morphs the notebooks", async ({ page }) => {
    await openStory(page);
    await goToStage(page, 2);
    const slider = page.getByRole("slider", { name: /How lopsided/i });
    await expect(slider).toBeVisible();
    const evenest = await page
      .getByRole("img", { name: /Student 1 has notes on/ })
      .getAttribute("aria-label");
    // Home rather than fill(): a range input snaps to its step grid, and Home
    // is how a keyboard user would get to the extreme anyway.
    await slider.press("Home");
    await expect(page.getByRole("img", { name: /Student 1 has notes on/ })).not.toHaveAttribute(
      "aria-label",
      evenest ?? "",
    );
    // Arrow keys belong to the slider once it has focus, not to the stage rail.
    await slider.focus();
    await page.keyboard.press("ArrowRight");
    await expect(page.getByRole("heading", { name: /Not everyone knows everything/i })).toBeVisible();
  });

  test("stage 4 and 5 render the manifest's panels, and admit to placeholders", async ({
    page,
  }) => {
    await openStory(page);
    await goToStage(page, 3);
    await expect(page.locator('[data-attack-arm="undefended"] img')).toHaveCount(2);
    await expect(page.locator("[data-attack-condition]")).toHaveText(/batch \d+ · noise ×/);
    await goToStage(page, 4);
    await expect(page.locator('[data-attack-arm="defended"] img')).toHaveCount(2);
    // Every panel actually decoded — a broken data URI is a silent 0×0 image.
    for (const img of await page.locator("[data-attack-arm] img").all()) {
      expect(await img.evaluate((node: HTMLImageElement) => node.naturalWidth)).toBeGreaterThan(0);
    }
  });

  test("space pauses, escape leaves, and the console has a way in", async ({ page }) => {
    await page.goto("/");
    await page.getByRole("button", { name: /guided walkthrough/i }).click();
    await expect(page.getByRole("heading", { name: /How federated learning works/i })).toBeVisible();
    expect(new URL(page.url()).pathname).toMatch(/\/story$/);

    await stageIndex(page, 0);
    await goToStage(page, 1);
    const pause = page.getByRole("button", { name: /Pause this stage/ });
    await expect(pause).toBeVisible();
    await page.keyboard.press(" ");
    await expect(page.getByRole("button", { name: /Play this stage/ })).toBeVisible();

    await page.keyboard.press("Escape");
    await expect(page.getByRole("heading", { name: /Federated Learning Coordinator/i })).toBeVisible();
  });
});

test.describe("story mode with reduced motion", () => {
  // emulateMedia before the first navigation, so the components see the media
  // state on mount rather than after a repaint.
  test.beforeEach(async ({ page }) => {
    await page.emulateMedia({ reducedMotion: "reduce" });
  });

  test("every stage is a captioned still with nothing lost", async ({ page }) => {
    await openStory(page);
    await expect(page.locator('[data-story-reduced="true"]')).toBeVisible();
    // Nothing to play, so the control says so rather than doing nothing.
    await expect(page.getByRole("button", { name: /Play this stage|Pause this stage/ })).toBeDisabled();

    for (let index = 0; index < STAGES.length; index += 1) {
      await assertStage(page, index);
      if (index < STAGES.length - 1) await page.keyboard.press("ArrowRight");
    }
  });

  test("the stages that animate say what the still is showing instead", async ({ page }) => {
    await openStory(page);
    await expect(page.locator("[data-story-still]")).toBeVisible(); // stage 1
    await goToStage(page, 1);
    await expect(page.locator("[data-story-still]")).toContainText(/after the merge/i);
    await goToStage(page, 4);
    await expect(page.locator("[data-story-still]")).toContainText(/already trimmed/i);
    // The budget meter reads its measured total rather than "spending…".
    await expect(
      page.getByLabel(`Privacy ink budget: ε ${display("epsilon")}`),
    ).toBeVisible();
  });

  test("stage 6 still draws both curves and states both endpoints", async ({ page }) => {
    await openStory(page);
    await goToStage(page, 5);
    await expect(page.locator("path.recharts-curve")).toHaveCount(2);
    await expect(page.locator('[data-figure="nodpFinal"]').first()).toHaveText(
      display("nodpFinal"),
    );
    await expect(page.locator('[data-figure="dpFinal"]').first()).toHaveText(display("dpFinal"));
  });
});
