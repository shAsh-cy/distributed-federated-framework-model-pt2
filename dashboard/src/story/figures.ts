/**
 * The only way a number reaches a story screen.
 *
 * Every entry in story_figures.json carries the committed results file and the
 * JSON pointer it was read from (see scripts/build_story_figures.py), and the
 * display string the UI must use. Stage components import `figure()` and
 * `series()`; they never format a measurement themselves and never contain a
 * numeric literal that a reader will see. tests/story.test.tsx enforces both
 * halves of that: the pointers must still resolve, and nothing numeric may
 * appear on screen that these values do not back.
 */
import data from "../../fixtures/story_figures.json";

export type FigureName = keyof typeof data.figures;
export type SeriesName = keyof typeof data.series;

export type Figure = {
  value: number;
  display: string;
  format: string;
  meaning: string;
  source: { file: string; pointer: string };
};

export type Series = {
  points: number[];
  seeds: number[];
  meaning: string;
  source: { file: string; pointer: string; field: string; reduce: string };
};

export const FIGURES = data.figures as Record<FigureName, Figure>;
export const SERIES = data.series as Record<SeriesName, Series>;

/** The display string: "72.8 %", "6.228", "200". */
export function figure(name: FigureName): string {
  return FIGURES[name].display;
}

/** The underlying measurement, for drawing rather than printing. */
export function value(name: FigureName): number {
  return FIGURES[name].value;
}

export function series(name: SeriesName): Series {
  return SERIES[name];
}

/** "docs/_final_batch_b.json /achieved_epsilon" — shown as provenance. */
export function provenance(name: FigureName): string {
  const { file, pointer } = FIGURES[name].source;
  return `${file} ${pointer}`;
}
