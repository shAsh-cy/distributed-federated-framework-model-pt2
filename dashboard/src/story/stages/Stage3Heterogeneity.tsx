/**
 * Stage 3 — not everyone knows everything.
 *
 * The Dirichlet preview from the Configure view, turned into the thing the
 * reader drags. It uses the coupled sampler (see dirichlet.ts): the same
 * distribution fl/data.py deals from, but coupled across alpha through the
 * inverse-CDF trick so that every notebook MORPHS as the slider moves instead
 * of being redealt. That is the whole difference between a control that feels
 * like an instrument and one that feels like a dice roll.
 *
 * The bars here are a preview of the partition shape at this alpha, not a
 * measurement — the same honesty label Configure carries.
 */
import { useMemo, useState } from "react";
import { useReducedMotion } from "framer-motion";

import { previewPartitionCoupled } from "../../lib/dirichlet";
import { Slider } from "../../ui/primitives";
import { Caption } from "../ui";

const STUDENTS = 10;
const CHAPTERS = 10;
const PER_CHAPTER = 600;

/** Log-scaled slider: the interesting range is 0.05–1, not 5–10. */
const MIN_EXP = Math.log10(0.05);
const MAX_EXP = Math.log10(10);

function Notebook({
  counts,
  animate,
  index,
}: {
  counts: number[];
  animate: boolean;
  index: number;
}) {
  const width = 84;
  const height = 46;
  const max = Math.max(...counts, 1);
  const barWidth = width / counts.length;
  const chapters = counts.filter((c) => c > PER_CHAPTER * 0.02).length;
  return (
    <li className="flex min-w-0 flex-col items-center gap-1">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="w-full"
        preserveAspectRatio="none"
        role="img"
        aria-label={`Student ${index + 1} has notes on ${chapters} of ${CHAPTERS} chapters`}
      >
        <rect x={0} y={0} width={width} height={height} fill="var(--ground-raised)" />
        {counts.map((count, chapter) => {
          const h = Math.max(count > 0 ? 1 : 0, (count / max) * (height - 4));
          return (
            <rect
              key={chapter}
              x={chapter * barWidth + 0.5}
              y={height - h}
              width={barWidth - 1}
              height={h}
              fill="var(--client)"
              style={animate ? { transition: "y 110ms linear, height 110ms linear" } : undefined}
            />
          );
        })}
      </svg>
      <span className="readout text-xs text-slate">s{index + 1}</span>
    </li>
  );
}

export function Stage3Heterogeneity() {
  const reduced = useReducedMotion() ?? false;
  const [exponent, setExponent] = useState(Math.log10(0.5));
  const alpha = 10 ** exponent;

  const notebooks = useMemo(
    () =>
      previewPartitionCoupled({
        alpha,
        numClients: STUDENTS,
        numClasses: CHAPTERS,
        perClass: PER_CHAPTER,
      }),
    [alpha],
  );

  // A word, not a number: this is a property of the preview on screen, and the
  // explainer prints no number it cannot point at a results file for.
  const spread = useMemo(() => {
    const shares = notebooks.map((n) => {
      const total = n.reduce((s, x) => s + x, 0) || 1;
      return Math.max(...n) / total;
    });
    const mean = shares.reduce((s, x) => s + x, 0) / shares.length;
    if (mean > 0.6) return "one chapter each";
    if (mean > 0.32) return "lopsided";
    if (mean > 0.18) return "uneven";
    return "everyone has a bit of everything";
  }, [notebooks]);

  return (
    <div className="flex flex-col gap-4">
      <div className="border border-rule bg-ground-raised p-4">
        <Slider
          label="How lopsided are the notebooks?"
          value={exponent}
          min={MIN_EXP}
          max={MAX_EXP}
          step={0.005}
          onChange={setExponent}
          format={(v) => `α ${(10 ** v).toFixed(2)}`}
        />
        <div className="mt-1 flex justify-between font-prose text-xs text-slate">
          <span>everyone has a bit of everything</span>
          <span>{spread}</span>
          <span>one chapter each</span>
        </div>
      </div>

      <ul className="grid grid-cols-3 gap-3 sm:grid-cols-5">
        {notebooks.map((counts, index) => (
          <Notebook key={index} counts={counts} index={index} animate={!reduced} />
        ))}
      </ul>

      <Caption>
        Some students only have notes on a few chapters. That is the hard part of this whole field.
      </Caption>
      <p className="story-measure font-prose text-base">
        Drag the slider. Nobody chose who got which chapters — a hospital sees the patients near it,
        a phone sees the words its owner types. The teacher has to build one textbook that works for
        the whole class out of homework from students who each saw a different slice of the course.
      </p>
      <p className="font-prose text-xs text-slate">
        A preview of the partition shape at this α, drawn the same way{" "}
        <span className="readout">fl/data.py</span> deals the real one. Not a measurement.
      </p>
    </div>
  );
}
