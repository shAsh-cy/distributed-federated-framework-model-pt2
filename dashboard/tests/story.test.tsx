/**
 * The story's honesty tests.
 *
 * Two claims are being defended here.
 *
 * ONE: every figure the explainer quotes still resolves, in the committed
 * results file, at the JSON pointer it says it came from. story_figures.json
 * is generated, but a generated file can be edited, and a pointer can rot when
 * an experiment is re-run. These tests re-read the real JSONs.
 *
 * TWO — the "no invented numbers" guard. The brief asked for a test that no
 * hardcoded numeric string in the story components fails to appear in the
 * committed results. Scanning source text for numeric literals turns out to be
 * the wrong shape: it cannot tell 72.8 from a stroke width, and it misses
 * anything assembled at runtime. So this does the stronger version — it scans
 * what the READER SEES. Each stage is rendered, every sanctioned display
 * string is struck out of its text, and whatever numbers are left must each be
 * backed by a value in a small, explicit pool: the sourced figures, a handful
 * of facts from the recorded fixture (how many rounds, how many students, who
 * was late), the inversion manifest, and the story's own control defaults.
 * Anything else fails, including a number typed straight into a caption.
 *
 * What it does not cover, stated rather than hidden: recharts renders nothing
 * under jsdom (ResponsiveContainer measures zero), so stage 6's axis ticks are
 * out of scope here. Their values come from the series asserted below, and
 * e2e/story.spec.ts checks the rendered figures in a real browser.
 */
import { act, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import batchB from "../../docs/_final_batch_b.json";
import femnistBaseline from "../../docs/_femnist_baseline.json";
import femnistBudgetE from "../../docs/_femnist_budget_e.json";
import femnistRCurve from "../../docs/_femnist_r_curve.json";
import manifest from "../../docs/inversion/manifest.json";
import fashionNoDp from "../../results/no_dp.json";
import figuresJson from "../fixtures/story_figures.json";
import { FIGURES, SERIES } from "../src/story/figures";
import { ClosingCard } from "../src/story/stages/ClosingCard";
import { Stage1Problem } from "../src/story/stages/Stage1Problem";
import { Stage2Round } from "../src/story/stages/Stage2Round";
import { Stage3Heterogeneity } from "../src/story/stages/Stage3Heterogeneity";
import { Stage4Attack } from "../src/story/stages/Stage4Attack";
import { Stage5Defence } from "../src/story/stages/Stage5Defence";
import { Stage6Cost } from "../src/story/stages/Stage6Cost";
import {
  COHORT_SIZE,
  DEADLINE_CLIENT,
  DEADLINE_ROUND,
  POPULATION,
  TOTAL_ROUNDS,
} from "../src/story/storyRun";

/**
 * The committed results, imported rather than read off disk, so this file
 * needs no node types and so a missing source is a build error rather than a
 * runtime one. Every file story_figures.json cites must appear here — the
 * first test below fails if one does not.
 */
const COMMITTED: Record<string, unknown> = {
  "docs/_final_batch_b.json": batchB,
  "docs/_femnist_baseline.json": femnistBaseline,
  "docs/_femnist_budget_e.json": femnistBudgetE,
  "docs/_femnist_r_curve.json": femnistRCurve,
  "results/no_dp.json": fashionNoDp,
};

function committed(relative: string): unknown {
  const document = COMMITTED[relative];
  if (document === undefined) {
    throw new Error(
      `story_figures.json cites ${relative}, which this test does not import. ` +
        "Add it to COMMITTED so the pointer is actually verified.",
    );
  }
  return document;
}

/** RFC 6901, the same subset scripts/build_story_figures.py resolves. */
function pointer(document: unknown, path: string): unknown {
  let node: unknown = document;
  for (const rawToken of path.split("/").slice(1)) {
    const token = rawToken.replace(/~1/g, "/").replace(/~0/g, "~");
    node = Array.isArray(node)
      ? (node as unknown[])[Number(token)]
      : (node as Record<string, unknown>)[token];
  }
  return node;
}

/* -- one: the figures are where they say they are -------------------------- */

describe("sourced figures", () => {
  it("resolves every pointer in the real committed results", () => {
    for (const [name, entry] of Object.entries(FIGURES)) {
      if ("series" in entry.source) continue; // derived; checked with its series
      const document = committed(entry.source.file);
      expect(pointer(document, entry.source.pointer), `${name}`).toBe(entry.value);
    }
  });

  it("recomputes both accuracy curves from the runs they name", () => {
    for (const [name, entry] of Object.entries(SERIES)) {
      const runs = pointer(committed(entry.source.file), entry.source.pointer) as {
        seed: number;
        history: Record<string, number>[];
      }[];
      expect(runs.map((r) => r.seed), `${name} seeds`).toEqual(entry.seeds);
      const recomputed = runs[0]!.history.map(
        (_, round) =>
          runs.reduce((sum, run) => sum + run.history[round]![entry.source.field]!, 0) /
          runs.length,
      );
      expect(recomputed.length, `${name} length`).toBe(entry.points.length);
      // Point-wise rather than deep-equal: Python's fmean and a JS reduce sum
      // in different orders, so the last bit of a float can differ.
      recomputed.forEach((point, round) => {
        expect(point, `${name} round ${round + 1}`).toBeCloseTo(entry.points[round]!, 12);
      });
    }
  });

  it("formats every display string from its own value", () => {
    // A display string that has drifted from its value is how a real number
    // becomes a wrong one on screen.
    const check: Record<string, (v: number) => string> = {
      pct1: (v) => `${(v * 100).toFixed(1)} %`,
      pct0: (v) => `${(v * 100).toFixed(0)} %`,
      points1: (v) => (v * 100).toFixed(1),
      fixed3: (v) => v.toFixed(3),
      fixed2: (v) => v.toFixed(2),
      fixed1: (v) => v.toFixed(1),
      integer: (v) => String(Math.trunc(v)),
      grouped: (v) => Math.trunc(v).toLocaleString("en-US"),
      exponent: (v) => `1e${Math.round(Math.log10(v))}`,
    };
    for (const [name, entry] of Object.entries(FIGURES)) {
      const format = check[entry.format];
      expect(format, `${name} uses an unknown format ${entry.format}`).toBeTruthy();
      expect(format!(entry.value), `${name}`).toBe(entry.display);
    }
  });

  it("keeps the headline arithmetic self-consistent", () => {
    expect(FIGURES.nodpFinal.value - FIGURES.dpFinal.value).toBeCloseTo(
      FIGURES.dpCost.value,
      9,
    );
    expect(SERIES.dpCurve.points.at(-1)).toBeCloseTo(FIGURES.dpFinal.value, 12);
    expect(SERIES.nodpCurve.points.at(-1)).toBeCloseTo(FIGURES.nodpFinal.value, 12);
  });
});

/* -- two: nothing on screen that nothing backs ----------------------------- */

/**
 * Small integers the recorded fixture puts on screen: how many rounds it ran,
 * how big the cohort was, how many students there are, which round lost one,
 * and the index in that student's id.
 */
const FIXTURE_FACTS = new Set<number>([
  TOTAL_ROUNDS,
  COHORT_SIZE,
  POPULATION,
  DEADLINE_ROUND,
  Number(DEADLINE_CLIENT.replace(/\D/g, "")),
  ...Array.from({ length: POPULATION }, (_, i) => i + 1), // student labels s1..s10
]);

/** Straight out of docs/inversion/manifest.json. */
const MANIFEST_FACTS = new Set<number>(
  manifest.entries.flatMap((entry) => [
    entry.batch_size,
    entry.noise_multiplier,
    ...(entry.epsilon === null ? [] : [entry.epsilon]),
  ]),
);

/**
 * The story's own controls. Not measurements, and the only entries in this
 * whole file that are hand-written numbers — kept to one line so that adding
 * to it is a visible act.
 */
const CONTROL_VALUES = new Set<number>([0.5]); // stage 3's default alpha

const BACKING_VALUES = [
  ...Object.values(FIGURES).map((f) => f.value),
  ...FIXTURE_FACTS,
  ...MANIFEST_FACTS,
  ...CONTROL_VALUES,
];

/** Display strings the reader is allowed to see, longest first. */
const SANCTIONED_STRINGS = [
  ...Object.values(FIGURES).map((f) => f.display),
  ...[...MANIFEST_FACTS].map((v) => v.toFixed(3)),
  ...[...CONTROL_VALUES].map((v) => v.toFixed(2)),
].sort((a, b) => b.length - a.length);

/** True if `token`, at its own precision, is a correct rounding of a backing value. */
function isBacked(token: string): boolean {
  const parsed = Number(token.replace(/,/g, ""));
  if (!Number.isFinite(parsed)) return false;
  const decimals = token.includes(".") ? token.split(".")[1]!.length : 0;
  const tolerance = 0.5 * 10 ** -decimals;
  return BACKING_VALUES.some((backing) => {
    if (Math.abs(parsed - backing) < tolerance) return true;
    // Percentages are the same measurement wearing a different unit.
    return Math.abs(parsed - backing * 100) < tolerance;
  });
}

function unbackedNumbers(text: string): string[] {
  let remaining = text;
  for (const sanctioned of SANCTIONED_STRINGS) {
    // Boundary-aware, so striking out "20" cannot turn 2024 into 24.
    const escaped = sanctioned.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    remaining = remaining.replace(new RegExp(`(?<![\\d.])${escaped}(?![\\d.])`, "g"), " ");
  }
  const tokens = remaining.match(/\d[\d,]*(?:\.\d+)?/g) ?? [];
  return [...new Set(tokens)].filter((token) => !isBacked(token));
}

afterEach(() => {
  vi.useRealTimers();
});

describe("no invented numbers", () => {
  it("the guard itself catches a number nothing backs", () => {
    // Without this, a bug in the tokeniser would make every test below pass.
    expect(unbackedNumbers("accuracy reached 91.4 % after the change")).toEqual(["91.4"]);
    expect(unbackedNumbers(`the cost was ${FIGURES.dpCost.display} points`)).toEqual([]);
  });

  it.each([
    ["stage 1", <Stage1Problem key="1" />],
    ["stage 3", <Stage3Heterogeneity key="3" />],
    ["stage 4", <Stage4Attack key="4" />],
    ["stage 5", <Stage5Defence key="5" active={false} still />],
    ["stage 6", <Stage6Cost key="6" active={false} still />],
    ["closing card", <ClosingCard key="c" />],
  ])("%s shows only numbers the committed results back", (_label, element) => {
    const { container } = render(element);
    expect(unbackedNumbers(container.textContent ?? "")).toEqual([]);
  });

  it("stage 2 shows only backed numbers at every beat of the round", () => {
    vi.useFakeTimers();
    const { container } = render(<Stage2Round active still={false} />);
    const offenders = new Set<string>();
    for (let step = 0; step < 40; step += 1) {
      unbackedNumbers(container.textContent ?? "").forEach((n) => offenders.add(n));
      act(() => {
        vi.advanceTimersByTime(400);
      });
    }
    expect([...offenders]).toEqual([]);
  });
});

/* -- the stages actually say their piece ----------------------------------- */

describe("stage content", () => {
  it("every stage carries a permanent caption", () => {
    for (const element of [
      <Stage1Problem key="1" />,
      <Stage2Round key="2" active={false} still />,
      <Stage3Heterogeneity key="3" />,
      <Stage4Attack key="4" />,
      <Stage5Defence key="5" active={false} still />,
      <Stage6Cost key="6" active={false} still />,
    ]) {
      const { container, unmount } = render(element);
      const caption = container.querySelector("[data-story-caption]");
      expect(caption?.textContent?.length ?? 0).toBeGreaterThan(20);
      unmount();
    }
  });

  it("stage 4 and stage 5 pick their arms out of the manifest, not by position", () => {
    const undefended = render(<Stage4Attack />);
    expect(
      undefended.container.querySelector("[data-attack-arm]")?.getAttribute("data-attack-arm"),
    ).toBe(manifest.entries.find((e) => e.noise_multiplier === 0)!.id);
    undefended.unmount();

    const defended = render(<Stage5Defence active={false} still />);
    const arms = [...defended.container.querySelectorAll("[data-attack-arm]")].map((n) =>
      n.getAttribute("data-attack-arm"),
    );
    expect(arms).toContain(manifest.entries.find((e) => e.noise_multiplier > 0)!.id);
  });

  it("says the reconstructions are placeholders while the manifest says they are", () => {
    const placeholders = manifest.entries.some((e) => e.placeholder);
    const { container } = render(<Stage4Attack />);
    expect(!!container.querySelector("[data-inversion-pending]")).toBe(placeholders);
  });

  it("stage 5 states the trim line, the noise and the budget", () => {
    render(<Stage5Defence active={false} still />);
    for (const name of ["clipNorm", "noiseZ", "epsilon", "delta", "clippedFinal"] as const) {
      expect(screen.getAllByText(FIGURES[name].display).length).toBeGreaterThan(0);
    }
  });

  it("stage 6 states both arms, the gap, and the pooled ceiling", () => {
    render(<Stage6Cost active={false} still />);
    for (const name of ["nodpFinal", "dpFinal", "dpCost", "pooled"] as const) {
      expect(screen.getAllByText(FIGURES[name].display).length).toBeGreaterThan(0);
    }
  });

  it("the committed figures file is the one the components read", () => {
    // Guards against a stray second copy drifting into the tree.
    expect(Object.keys(figuresJson.figures)).toEqual(Object.keys(FIGURES));
  });
});
