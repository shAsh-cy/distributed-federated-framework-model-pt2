/**
 * Shared furniture for the story stages. Instrument primitives where they fit
 * (Meter, Slider); these are the two or three things the walkthrough needs
 * that the console never did.
 */
import type { ReactNode } from "react";

import { figure, provenance, type FigureName } from "./figures";

/**
 * The caption. Always rendered, never a hover affordance, never conditional on
 * motion — it is the stage's content in text, and with animation disabled it
 * is most of what a reader gets.
 */
export function Caption({ children }: { children: ReactNode }) {
  return (
    <p className="story-caption story-measure font-prose text-base text-ink" data-story-caption>
      {children}
    </p>
  );
}

/** A figure with its provenance in the title attribute, in the readout face. */
export function Fig({ name }: { name: FigureName }) {
  return (
    <span className="readout" title={provenance(name)} data-figure={name}>
      {figure(name)}
    </span>
  );
}

export function StageHeading({ index, title }: { index: number; title: string }) {
  return (
    <div className="flex items-baseline gap-3">
      <span className="readout text-xs text-slate">{String(index).padStart(2, "0")}</span>
      <h2 className="font-head text-lg uppercase tracking-head">{title}</h2>
    </div>
  );
}

/** The line under a stage that says where its numbers came from. */
export function Sourced({ names }: { names: FigureName[] }) {
  return (
    <p className="font-prose text-xs text-slate">
      Measured, not illustrative —{" "}
      {[...new Set(names.map((n) => provenance(n).split(" ")[0]))].join(", ")}
    </p>
  );
}

/** A stage's still-frame notice when animation is off. */
export function StillNotice({ children }: { children: ReactNode }) {
  return (
    <p className="font-prose text-xs text-slate" data-story-still>
      {children}
    </p>
  );
}
