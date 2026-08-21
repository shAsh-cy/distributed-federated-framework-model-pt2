/**
 * Story mode: the guided walkthrough.
 *
 * Six stages and a closing card, driven by the recorded run — no backend, no
 * mock mode, no training started. Arrow keys step, space plays and pauses,
 * Escape leaves. Captions are permanent, not hover-only, and under
 * prefers-reduced-motion every stage degrades to a captioned still with the
 * same content: nothing is carried by animation alone.
 *
 * The frame is a classroom, carried consistently across all six stages: the
 * aggregator is the teacher, clients are students, local data is each
 * student's notebook, model updates are homework corrections, and the global
 * model is the class textbook.
 */
import { useReducedMotion } from "framer-motion";
import { useCallback, useEffect, useState } from "react";

import { Button } from "../ui/primitives";
import { ClosingCard } from "./stages/ClosingCard";
import { Stage1Problem } from "./stages/Stage1Problem";
import { Stage2Round } from "./stages/Stage2Round";
import { Stage3Heterogeneity } from "./stages/Stage3Heterogeneity";
import { Stage4Attack } from "./stages/Stage4Attack";
import { Stage5Defence } from "./stages/Stage5Defence";
import { Stage6Cost } from "./stages/Stage6Cost";

type StageProps = { active: boolean; still: boolean };

type Stage = {
  title: string;
  /** Whether space has anything to play here. */
  timed: boolean;
  render: (props: StageProps) => JSX.Element;
};

const STAGES: Stage[] = [
  { title: "The problem", timed: false, render: () => <Stage1Problem /> },
  { title: "One round", timed: true, render: (p) => <Stage2Round {...p} /> },
  { title: "Not everyone knows everything", timed: false, render: () => <Stage3Heterogeneity /> },
  { title: "The nosy teacher", timed: false, render: () => <Stage4Attack /> },
  { title: "The defence", timed: true, render: (p) => <Stage5Defence {...p} /> },
  { title: "What it costs, honestly", timed: true, render: (p) => <Stage6Cost {...p} /> },
  { title: "In one paragraph", timed: false, render: () => <ClosingCard /> },
];

export const STAGE_COUNT = STAGES.length;

/** Typing in a control must never be stolen by the stage shortcuts. */
function isTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  return tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA" || target.isContentEditable;
}

export function StoryMode({ onExit }: { onExit: () => void }) {
  const reduced = useReducedMotion() ?? false;
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(!reduced);

  const stage = STAGES[index]!;

  const go = useCallback((next: number) => {
    setIndex((current) => {
      const clamped = Math.min(STAGES.length - 1, Math.max(0, next));
      if (clamped !== current) setPlaying(true);
      return clamped;
    });
  }, []);

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if (event.key === "Escape") {
        event.preventDefault();
        onExit();
        return;
      }
      if (isTypingTarget(event.target)) return;
      if (event.key === "ArrowRight") {
        event.preventDefault();
        setIndex((i) => {
          const next = Math.min(STAGES.length - 1, i + 1);
          if (next !== i) setPlaying(true);
          return next;
        });
      } else if (event.key === "ArrowLeft") {
        event.preventDefault();
        setIndex((i) => {
          const next = Math.max(0, i - 1);
          if (next !== i) setPlaying(true);
          return next;
        });
      } else if (event.key === " " || event.key === "Spacebar") {
        if (event.target instanceof HTMLElement) {
          const tag = event.target.tagName;
          if (tag === "BUTTON" || tag === "A") return; // space activates those
        }
        event.preventDefault();
        setPlaying((p) => !p);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onExit]);

  // Reduced motion means every stage is a still; there is nothing to play.
  useEffect(() => {
    if (reduced) setPlaying(false);
  }, [reduced]);

  return (
    <div className="mx-auto flex min-h-screen max-w-4xl flex-col px-4" data-story-mode>
      <header className="flex flex-wrap items-baseline justify-between gap-2 border-b-2 border-ink py-3">
        <h1 className="font-head text-xl uppercase tracking-head">
          How federated learning works
        </h1>
        <Button onClick={onExit} aria-label="Leave the walkthrough (Escape)">
          Back to the dashboard
        </Button>
      </header>

      <nav aria-label="Stages" className="flex flex-wrap items-center gap-1 py-3">
        {STAGES.map((entry, i) => (
          <button
            key={entry.title}
            onClick={() => go(i)}
            aria-current={i === index ? "step" : undefined}
            aria-label={`Stage ${i + 1}: ${entry.title}`}
            className={`h-1.5 flex-1 min-w-[24px] ${
              i === index ? "bg-global" : i < index ? "bg-ink opacity-40" : "bg-rule"
            }`}
          />
        ))}
      </nav>

      <div className="flex items-baseline justify-between gap-3 pb-3">
        <div className="flex items-baseline gap-3">
          <span className="readout text-xs text-slate" data-stage-index>
            {String(index + 1).padStart(2, "0")} / {String(STAGES.length).padStart(2, "0")}
          </span>
          <h2 className="font-head text-lg uppercase tracking-head">{stage.title}</h2>
        </div>
        <p className="readout hidden text-xs text-slate sm:block">
          ← → stages · space play/pause · esc exit
        </p>
      </div>

      <main
        className="flex-1"
        data-story-stage={index + 1}
        data-story-reduced={reduced ? "true" : "false"}
      >
        {stage.render({ active: playing, still: reduced })}
      </main>

      <p aria-live="polite" className="sr-only">
        Stage {index + 1} of {STAGES.length}: {stage.title}
      </p>

      <footer className="mt-6 flex items-center justify-between gap-2 border-t border-rule py-3">
        <Button onClick={() => go(index - 1)} disabled={index === 0}>
          ← Back
        </Button>
        <Button
          onClick={() => setPlaying((p) => !p)}
          disabled={reduced || !stage.timed}
          aria-label={playing ? "Pause this stage (space)" : "Play this stage (space)"}
        >
          {playing ? "Pause" : "Play"}
        </Button>
        <Button
          tone="primary"
          onClick={() => go(index + 1)}
          disabled={index === STAGES.length - 1}
        >
          Next →
        </Button>
      </footer>

      {reduced ? (
        <p className="pb-4 font-prose text-xs text-slate">
          Your system asks for reduced motion, so every stage is shown as a still. Nothing is
          missing: the captions carry what the animation would have.
        </p>
      ) : null}
    </div>
  );
}
