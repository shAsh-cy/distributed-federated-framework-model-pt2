/**
 * Stage 5 — the defence. Two mechanisms, shown rather than named.
 *
 * Trimming is L2 clipping: every correction cut to the same maximum length.
 * The bars are twenty stand-ins for the two-hundred-client cohort, and the
 * share of them that crosses the trim line is the measured clipped fraction
 * from the last round of the run this story quotes — not a number chosen to
 * look convincing.
 *
 * Ink is the Gaussian noise added to the aggregate. Then the ochre meter, the
 * one colour in this whole design system reserved for the privacy budget, and
 * the attack from stage 4 re-run against what the mechanism actually produces.
 */
import { motion, useReducedMotion } from "framer-motion";
import { useEffect, useRef, useState } from "react";

import { Meter } from "../../ui/primitives";
import { AttackPanel, PendingNotice } from "../AttackPanel";
import { figure, value } from "../figures";
import { AWAITING_REAL_FIGURES, DEFENDED } from "../inversion";
import { Caption, Fig, StillNotice } from "../ui";

const BARS = 20;
const TRIM = value("clipNorm");
const CLIPPED_SHARE = value("clippedFinal");
const MEDIAN = value("medianNormFinal");

/**
 * Lengths for the stand-in cohort: exactly the measured share of them longer
 * than the trim line, spread around the measured median. Deterministic, so the
 * picture is the same every time the stage is opened.
 */
function barLengths(): number[] {
  const over = Math.round(BARS * CLIPPED_SHARE);
  return Array.from({ length: BARS }, (_, i) => {
    const rank = (i + 0.5) / BARS;
    const length =
      i < over
        ? TRIM + 0.15 + 1.5 * (1 - rank) // above the line, longest first
        : MEDIAN * (0.45 + 0.75 * (1 - rank)); // below it, tapering
    return Math.max(0.25, length);
  });
}

const LENGTHS = barLengths();
const LONGEST = Math.max(...LENGTHS, TRIM * 1.6);

function TrimAndInk({ still }: { still: boolean }) {
  const width = 460;
  const height = 150;
  const scale = (width - 90) / LONGEST;
  const trimX = 60 + TRIM * scale;
  const rowHeight = (height - 34) / BARS;

  return (
    <svg
      viewBox={"0 0 " + width + " " + height}
      className="w-full max-w-[460px]"
      role="img"
      aria-label={
        "Twenty corrections; " +
        figure("clippedFinal") +
        " of them are longer than the trim line at " +
        figure("clipNorm") +
        " and are cut back to it"
      }
    >
      <text x={0} y={11} className="readout" fontSize={9} fill="var(--slate)">
        corrections
      </text>
      <line x1={trimX} y1={16} x2={trimX} y2={height - 14} stroke="var(--ink)" strokeWidth={1.25} />
      <text x={trimX + 4} y={height - 3} className="readout" fontSize={9} fill="var(--ink)">
        trim line
      </text>
      {LENGTHS.map((length, i) => {
        const y = 22 + i * rowHeight;
        const full = length * scale;
        const cut = Math.min(length, TRIM) * scale;
        return (
          <g key={i}>
            {full > cut ? (
              <line
                x1={60 + cut}
                y1={y}
                x2={60 + full}
                y2={y}
                stroke="var(--slate)"
                strokeWidth={1}
                strokeDasharray="2 3"
              />
            ) : null}
            <motion.line
              x1={60}
              y1={y}
              y2={y}
              stroke="var(--client)"
              strokeWidth={2.5}
              initial={{ x2: 60 + full }}
              animate={
                still
                  ? { x2: 60 + cut }
                  : { x2: [60 + full, 60 + cut, 60 + cut + (i % 2 ? 5 : -5), 60 + cut] }
              }
              transition={
                still
                  ? { duration: 0 }
                  : {
                      duration: 3.4,
                      times: [0, 0.38, 0.62, 1],
                      delay: i * 0.02,
                      repeat: Infinity,
                      repeatDelay: 1.1,
                      ease: ["easeOut", "easeInOut", "easeOut"],
                    }
              }
            />
          </g>
        );
      })}
    </svg>
  );
}

export function Stage5Defence({ active, still }: { active: boolean; still: boolean }) {
  const reduced = useReducedMotion() ?? false;
  const quiet = still || reduced;
  const epsilon = value("epsilon");
  const [spent, setSpent] = useState(quiet ? epsilon : 0);
  // Pausing freezes the meter where it is; it does not snap it to the total.
  const spentRef = useRef(quiet ? epsilon : 0);
  spentRef.current = spent;

  useEffect(() => {
    if (quiet) {
      setSpent(epsilon);
      return;
    }
    if (!active) return;
    const from = spentRef.current >= epsilon ? 0 : spentRef.current;
    const started = performance.now();
    let frame = 0;
    const tick = (now: number) => {
      const progress = Math.min(1, (now - started) / 2400);
      setSpent(from + (epsilon - from) * progress);
      if (progress < 1) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [active, quiet, epsilon]);

  return (
    <div className="flex flex-col gap-5">
      <section className="flex flex-col gap-2">
        <h3 className="font-head text-sm uppercase tracking-head">First: trim every correction</h3>
        <TrimAndInk still={quiet} />
        <p className="story-measure font-prose text-base">
          No student is allowed to shout. Every correction is cut back to the same maximum length,
          so one student can never swing the textbook on their own. In the last round of the run
          this story quotes, <Fig name="clippedFinal" /> of corrections were long enough to be
          trimmed — the middle one was <Fig name="medianNormFinal" /> against a trim line of{" "}
          <Fig name="clipNorm" />.
        </p>
      </section>

      <section className="flex flex-col gap-2">
        <h3 className="font-head text-sm uppercase tracking-head">
          Then: smudge the ink before merging
        </h3>
        <p className="story-measure font-prose text-base">
          A little random ink is spilled on the pile of corrections before the teacher reads it.
          Not enough to ruin the lesson; enough that no single student&rsquo;s handwriting can be
          picked out of it. Here the smudge is <Fig name="noiseZ" /> times the trim line.
        </p>
      </section>

      <section className="flex flex-col gap-2 border border-rule bg-ground-raised p-4">
        <Meter
          label="Privacy ink budget"
          value={spent}
          max={epsilon}
          tone="budget"
          format={(v, max) => (v >= max ? "ε " + figure("epsilon") : "spending…")}
        />
        <p className="story-measure font-prose text-base">
          Every round spends some of the ink budget, and when it runs out, training stops. After{" "}
          <Fig name="rounds" /> rounds this run had spent <Fig name="epsilon" /> of it, at δ{" "}
          <Fig name="delta" />. That number is computed by the accountant, not chosen — you do not
          get to decide how private something was after the fact.
        </p>
        <p className="font-prose text-xs text-slate">
          The bar fills smoothly here for legibility; the budget does not actually accrue in equal
          steps. Only the total after <Fig name="rounds" /> rounds is measured.
        </p>
      </section>

      <section className="flex flex-col gap-2">
        <h3 className="font-head text-sm uppercase tracking-head">Now run the same attack again</h3>
        {DEFENDED ? (
          <AttackPanel
            entry={DEFENDED}
            originalTitle="The page"
            reconstructionTitle="What the attack rebuilt this time"
          />
        ) : (
          <p className="font-prose text-base text-client">
            docs/inversion/manifest.json has no entry with a noise multiplier above zero, so there
            is no defended arm to show. Add one; nothing in this stage needs to change.
          </p>
        )}
      </section>

      <Caption>
        Same attack, same student, same batch. Trimmed and smudged, there is nothing left to read.
      </Caption>
      {AWAITING_REAL_FIGURES ? <PendingNotice /> : null}
      {quiet ? (
        <StillNotice>
          Animation is off. The corrections are shown already trimmed to the line, and the ink
          budget already spent to its measured total.
        </StillNotice>
      ) : null}
    </div>
  );
}
