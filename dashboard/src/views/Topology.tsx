/**
 * Live client topology — the signature element and the whole motion budget.
 *
 * Clients ring a central aggregator; each node carries its local label
 * histogram, so the heterogeneity that makes FL hard is visible before a
 * single number is read. The animation encodes the real protocol:
 * sampled nodes illuminate; local training pulses for a duration
 * proportional to the client's measured wall-clock; updates travel inward
 * staggered by real arrival order with edge weight ∝ bytes; the aggregator
 * pulses; the new model propagates outward to every node — including the
 * unsampled, because that is what FedAvg does; deadline-missers dim, their
 * edge dashes, and they stay dimmed one round.
 *
 * Frameworks are distinguished by GLYPH (□ tensorflow / ◇ torch), never by
 * colour — colour already carries meaning here.
 *
 * Above 60 clients the full population renders as ring density bands
 * (SVG paths, not DOM-per-client) with the sampled cohort still resolved
 * individually. Reduced motion collapses every animation to instant state
 * changes with no information loss.
 *
 * Two motion profiles share this one component. "instrument" is the console's:
 * terse, quick, out of the way. "story" is the guided walkthrough's: the same
 * protocol, the same geometry, the same data — animated with mass. Nodes
 * overshoot and settle when the teacher calls on them, updates arrive with
 * momentum instead of sliding, the merge lands with weight. No second topology
 * exists; the explainer animates this one with more intent.
 */
import { motion, useReducedMotion } from "framer-motion";
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";

import type { ClientState, RunState } from "../lib/events";

const RING_THRESHOLD = 60;

export type NodePlacement = { id: string; x: number; y: number; angle: number };

/** Geometry handed to an overlay so it can draw in the topology's own frame. */
export type TopologyGeometry = { size: number; radius: number; placements: NodePlacement[] };

export type MotionProfile = "instrument" | "story";

/** Mass, overshoot, settle. The one place the story's physics is tuned. */
const STORY_SPRING = { type: "spring", stiffness: 140, damping: 11, mass: 1.1 } as const;
const STORY_ARRIVAL = { type: "spring", stiffness: 60, damping: 14, mass: 1 } as const;

function placeNodes(ids: string[], radius: number): NodePlacement[] {
  return ids.map((id, i) => {
    const angle = (i / ids.length) * Math.PI * 2 - Math.PI / 2;
    return { id, x: Math.cos(angle) * radius, y: Math.sin(angle) * radius, angle };
  });
}

function phaseColour(phase: ClientState["phase"]): string {
  switch (phase) {
    case "sampled":
    case "reported":
      return "var(--client)";
    case "dropped":
      return "var(--slate)";
    default:
      return "var(--slate)";
  }
}

function NodeHistogram({
  counts,
  size,
  colour,
}: {
  counts: number[];
  size: number;
  colour: string;
}) {
  const max = Math.max(...counts, 1);
  const barWidth = size / counts.length;
  return (
    <g aria-hidden="true">
      {counts.map((count, i) => {
        const h = Math.max(count > 0 ? 1 : 0, (count / max) * (size * 0.55));
        return (
          <rect
            key={i}
            x={-size / 2 + i * barWidth}
            y={size / 2 - h}
            width={Math.max(0.5, barWidth - 0.6)}
            height={h}
            fill={colour}
            opacity={0.85}
          />
        );
      })}
    </g>
  );
}

function frameworkGlyph(framework: string | null): string {
  if (framework === "torch") return "◇";
  if (framework === "tensorflow") return "□";
  return "";
}

/** Media-query hook; SSR-safe default is the wide layout. */
function useNarrowViewport(): boolean {
  const [narrow, setNarrow] = useState(
    () => typeof window !== "undefined" && window.matchMedia("(max-width: 640px)").matches,
  );
  useEffect(() => {
    const query = window.matchMedia("(max-width: 640px)");
    const onChange = (e: MediaQueryListEvent) => setNarrow(e.matches);
    query.addEventListener("change", onChange);
    return () => query.removeEventListener("change", onChange);
  }, []);
  return narrow;
}

/**
 * Narrow-viewport topology: a vertical list of client rows with sparkline
 * histograms. The same information, genuinely usable at 375px — not a
 * shrunken ring.
 */
function TopologyList({
  run,
  onPin,
  pinned,
}: {
  run: RunState;
  onPin: (clientId: string | null) => void;
  pinned: string | null;
}) {
  return (
    <ol aria-label={`Client list: ${run.clientOrder.length} clients`} className="flex flex-col">
      {run.clientOrder.map((id) => {
        const c = run.clients.get(id);
        if (!c) return null;
        const active = c.phase === "sampled" || c.phase === "reported";
        const max = Math.max(...c.info.label_histogram, 1);
        return (
          <li key={id}>
            <button
              onClick={() => onPin(pinned === id ? null : id)}
              aria-pressed={pinned === id}
              className={`flex w-full items-center gap-2 border-b border-rule py-1.5 text-left ${
                c.phase === "dropped" ? "opacity-50" : ""
              }`}
            >
              <span
                className="h-2.5 w-2.5 shrink-0 rounded-full"
                style={{ background: active ? "var(--client)" : "var(--slate)" }}
                aria-hidden="true"
              />
              <span className="readout w-14 shrink-0 text-xs">
                {id.replace("client-", "c")}
                {frameworkGlyph(c.framework)}
              </span>
              <svg width={70} height={16} aria-hidden="true" className="shrink-0">
                {c.info.label_histogram.map((count, i) => {
                  const bw = 70 / c.info.label_histogram.length;
                  const h = Math.max(count > 0 ? 1 : 0, (count / max) * 14);
                  return (
                    <rect
                      key={i}
                      x={i * bw}
                      y={16 - h}
                      width={Math.max(0.5, bw - 0.8)}
                      height={h}
                      fill={active ? "var(--client)" : "var(--slate)"}
                    />
                  );
                })}
              </svg>
              <span className="readout flex-1 text-right text-xs text-slate">
                {c.phase === "reported"
                  ? `acc ${c.lastLocalAccuracy?.toFixed(3) ?? "—"}`
                  : c.phase}
              </span>
            </button>
          </li>
        );
      })}
    </ol>
  );
}

export function Topology({
  run,
  highlightRound,
  onPin,
  pinned,
  motionProfile = "instrument",
  overlay,
  showDetail = true,
  forceRing = false,
}: {
  run: RunState;
  highlightRound: number | null;
  onPin: (clientId: string | null) => void;
  pinned: string | null;
  /** "story" swaps eased tweens for springs; see STORY_SPRING. */
  motionProfile?: MotionProfile;
  /** Drawn inside the SVG, above the edges and below the nodes. */
  overlay?: (geometry: TopologyGeometry) => ReactNode;
  showDetail?: boolean;
  /** Story mode narrates the ring even on a phone; the list view loses it. */
  forceRing?: boolean;
}) {
  const reduced = useReducedMotion();
  const narrow = useNarrowViewport() && !forceRing;
  const story = motionProfile === "story";
  const [hovered, setHovered] = useState<string | null>(null);
  const [focusIndex, setFocusIndex] = useState(0);
  const containerRef = useRef<SVGSVGElement>(null);

  const size = 460;
  const radius = size / 2 - 64;
  const ids = run.clientOrder;
  const dense = ids.length > RING_THRESHOLD;

  const sampledIds = useMemo(
    () =>
      ids.filter((id) => {
        const c = run.clients.get(id);
        return c && c.phase !== "idle";
      }),
    [ids, run.clients],
  );

  const placements = useMemo(
    () => placeNodes(dense ? sampledIds : ids, radius),
    [dense, sampledIds, ids, radius],
  );

  const keyNav = useCallback(
    (e: React.KeyboardEvent) => {
      const list = placements;
      if (list.length === 0) return;
      if (e.key === "ArrowRight" || e.key === "ArrowDown") {
        e.preventDefault();
        setFocusIndex((i) => (i + 1) % list.length);
      } else if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
        e.preventDefault();
        setFocusIndex((i) => (i - 1 + list.length) % list.length);
      } else if (e.key === "Enter") {
        e.preventDefault();
        const id = list[focusIndex]?.id ?? null;
        onPin(pinned === id ? null : id);
      }
    },
    [placements, focusIndex, onPin, pinned],
  );

  useEffect(() => {
    setFocusIndex(0);
  }, [placements.length]);

  const detailId = pinned ?? hovered ?? placements[focusIndex]?.id ?? null;
  const detail = detailId ? run.clients.get(detailId) : undefined;

  // Density bands for >60 clients: 24 arc segments shaded by client count.
  const bands = useMemo(() => {
    if (!dense) return null;
    const segments = 24;
    const counts = new Array<number>(segments).fill(0);
    ids.forEach((_, i) => {
      const segment = Math.floor((i / ids.length) * segments) % segments;
      counts[segment] = (counts[segment] ?? 0) + 1;
    });
    const max = Math.max(...counts, 1);
    return counts.map((count, s) => {
      const a0 = (s / segments) * Math.PI * 2 - Math.PI / 2;
      const a1 = ((s + 1) / segments) * Math.PI * 2 - Math.PI / 2;
      const r0 = radius + 18;
      const r1 = radius + 30;
      const path = [
        `M ${Math.cos(a0) * r0} ${Math.sin(a0) * r0}`,
        `A ${r0} ${r0} 0 0 1 ${Math.cos(a1) * r0} ${Math.sin(a1) * r0}`,
        `L ${Math.cos(a1) * r1} ${Math.sin(a1) * r1}`,
        `A ${r1} ${r1} 0 0 0 ${Math.cos(a0) * r1} ${Math.sin(a0) * r1}`,
        "Z",
      ].join(" ");
      return { path, opacity: 0.15 + 0.6 * (count / max), key: s };
    });
  }, [dense, ids, radius]);

  const lastCurvePoint = run.curve[run.curve.length - 1];
  const aggregatedThisRound =
    lastCurvePoint !== undefined && lastCurvePoint.round === run.currentRound;

  if (narrow) {
    return <TopologyList run={run} onPin={onPin} pinned={pinned} />;
  }

  const geometry: TopologyGeometry = { size, radius, placements };

  return (
    <div className="flex flex-col gap-2 lg:flex-row lg:items-start">
      <svg
        ref={containerRef}
        viewBox={`${-size / 2} ${-size / 2} ${size} ${size}`}
        className="mx-auto w-full max-w-[460px]"
        role="group"
        aria-label={`Client topology: ${ids.length} clients, round ${run.currentRound}. Use arrow keys to move between clients, Enter to pin.`}
        tabIndex={0}
        onKeyDown={keyNav}
      >
        {/* density ring for large populations */}
        {bands?.map((b) => (
          <path key={b.key} d={b.path} fill="var(--slate)" opacity={b.opacity} />
        ))}

        {/* edges */}
        {placements.map(({ id, x, y }) => {
          const c = run.clients.get(id);
          if (!c || c.phase === "idle") return null;
          const dropped = c.phase === "dropped";
          const bytes = c.lastBytes ?? 900_136;
          const width = Math.min(4, 0.75 + bytes / 600_000);
          const highlight =
            highlightRound !== null && c.droppedInRound !== highlightRound && !dropped;
          return (
            <g key={`edge-${id}`}>
              <line
                x1={x}
                y1={y}
                x2={0}
                y2={0}
                stroke={dropped ? "var(--slate)" : "var(--client)"}
                strokeWidth={dropped ? 1 : width}
                strokeDasharray={dropped ? "4 4" : undefined}
                opacity={dropped ? 0.5 : highlight ? 0.9 : 0.55}
              />
              {/* the update travelling inward, staggered by real arrival order */}
              {c.phase === "reported" && !reduced ? (
                <motion.circle
                  r={story ? Math.min(6, 1.6 + bytes / 320_000) : 3.2}
                  fill="var(--client)"
                  initial={{ cx: x, cy: y, opacity: 0 }}
                  animate={{ cx: 0, cy: 0, opacity: [0, 1, 1, 0] }}
                  transition={
                    story
                      ? { ...STORY_ARRIVAL, delay: (c.arrivalOrder ?? 0) * 0.28 }
                      : { duration: 0.9, delay: (c.arrivalOrder ?? 0) * 0.18, ease: "easeIn" }
                  }
                />
              ) : null}
            </g>
          );
        })}

        {/* aggregator */}
        <motion.rect
          x={-14}
          y={-14}
          width={28}
          height={28}
          fill="var(--global)"
          style={story ? { transformOrigin: "center", transformBox: "fill-box" } : undefined}
          animate={
            reduced
              ? undefined
              : aggregatedThisRound
                ? { scale: story ? [1, 1.55, 1] : [1, 1.25, 1] }
                : { scale: 1 }
          }
          transition={story ? { ...STORY_SPRING, damping: 7 } : { duration: 0.5 }}
          aria-label="Aggregator"
        />
        {/* new model propagating outward to ALL nodes after aggregation */}
        {aggregatedThisRound && !reduced ? (
          <motion.circle
            r={radius}
            fill="none"
            stroke="var(--global)"
            strokeWidth={1.5}
            initial={{ r: 16, opacity: 0.8 }}
            animate={{ r: radius + (story ? 34 : 0), opacity: 0 }}
            transition={story ? { duration: 1.5, ease: "easeOut" } : { duration: 0.8, ease: "easeOut" }}
          />
        ) : null}

        {/* nodes */}
        {placements.map(({ id, x, y }, index) => {
          const c = run.clients.get(id);
          if (!c) return null;
          const active = c.phase === "sampled" || c.phase === "reported";
          const colour = phaseColour(c.phase);
          const isFocus = index === focusIndex;
          const isPinned = pinned === id;
          const pulseDuration = Math.min(6, Math.max(0.8, (c.lastWallClock ?? 2) / 2));
          return (
            <g
              key={id}
              transform={`translate(${x} ${y})`}
              onMouseEnter={() => setHovered(id)}
              onMouseLeave={() => setHovered((h) => (h === id ? null : h))}
              onClick={() => onPin(isPinned ? null : id)}
              role="button"
              aria-label={`${id}: ${c.info.num_examples} examples, ${c.phase}`}
              style={{ cursor: "pointer" }}
            >
              {story ? (
                // Two layers, because the story asks the node to say two
                // things at once: WEIGHT when the teacher calls on it (a
                // spring that overshoots and settles), and HOW LONG the
                // student studies (a breath whose period is that client's
                // measured wall-clock). One property cannot carry both.
                <motion.g
                  style={{ transformOrigin: "center", transformBox: "fill-box" }}
                  animate={{
                    scale: reduced ? 1 : active ? 1.18 : c.phase === "dropped" ? 0.88 : 1,
                  }}
                  transition={STORY_SPRING}
                >
                  <motion.circle
                    r={16}
                    fill="var(--ground-raised)"
                    stroke={isPinned || isFocus ? "var(--global)" : colour}
                    strokeWidth={isPinned ? 3 : isFocus ? 2.5 : active ? 2 : 1}
                    animate={
                      reduced
                        ? { opacity: c.phase === "dropped" ? 0.45 : 1 }
                        : c.phase === "sampled"
                          ? { opacity: [1, 0.55, 1] }
                          : { opacity: c.phase === "dropped" ? 0.45 : 1 }
                    }
                    transition={
                      !reduced && c.phase === "sampled"
                        ? { repeat: Infinity, duration: pulseDuration, ease: "easeInOut" }
                        : { duration: 0.25 }
                    }
                  />
                </motion.g>
              ) : (
                <motion.circle
                  r={16}
                  fill="var(--ground-raised)"
                  stroke={isPinned || isFocus ? "var(--global)" : colour}
                  strokeWidth={isPinned ? 3 : isFocus ? 2.5 : active ? 2 : 1}
                  opacity={c.phase === "dropped" ? 0.45 : 1}
                  animate={
                    reduced
                      ? undefined
                      : c.phase === "sampled"
                        ? { scale: [1, 1.12, 1] }
                        : { scale: 1 }
                  }
                  transition={
                    c.phase === "sampled"
                      ? { repeat: Infinity, duration: pulseDuration, ease: "easeInOut" }
                      : { duration: 0.2 }
                  }
                />
              )}
              <NodeHistogram counts={c.info.label_histogram} size={20} colour={colour} />
              <text
                y={26}
                textAnchor="middle"
                className="readout"
                fontSize={8}
                fill="var(--ink)"
                opacity={0.75}
              >
                {id.replace("client-", "c")}
                {frameworkGlyph(c.framework)}
              </text>
            </g>
          );
        })}
        {/* Drawn last: a story overlay is annotation, and annotation sits on
            top of the thing it annotates. */}
        {overlay ? overlay(geometry) : null}

        {dense ? (
          <text
            y={size / 2 - 8}
            textAnchor="middle"
            className="readout"
            fontSize={9}
            fill="var(--slate)"
          >
            {ids.length} clients · ring shows density · sampled cohort resolved
          </text>
        ) : null}
      </svg>

      {/* hover / pin detail panel */}
      {showDetail ? (
      <aside
        aria-live="polite"
        className="min-w-[180px] border border-rule bg-ground-raised p-3 lg:mt-6"
      >
        {detail ? (
          <div className="flex flex-col gap-1">
            <span className="font-head text-sm uppercase tracking-head">
              {detailId} {pinned === detailId ? "· pinned" : ""}
            </span>
            <span className="readout text-xs">
              {detail.info.num_examples.toLocaleString("en-US")} examples
            </span>
            <span className="readout text-xs">
              framework {detail.framework ?? "unknown"}{" "}
              {frameworkGlyph(detail.framework)}
            </span>
            <span className="readout text-xs">
              local acc{" "}
              {detail.lastLocalAccuracy !== null
                ? detail.lastLocalAccuracy.toFixed(4)
                : "—"}
            </span>
            <span className="readout text-xs">
              wall {detail.lastWallClock !== null ? `${detail.lastWallClock.toFixed(1)}s` : "—"}
            </span>
            <svg width={140} height={44} role="img" aria-label="Enlarged label histogram">
              {detail.info.label_histogram.map((count, i) => {
                const max = Math.max(...detail.info.label_histogram, 1);
                const barWidth = 140 / detail.info.label_histogram.length;
                const h = Math.max(count > 0 ? 1 : 0, (count / max) * 40);
                return (
                  <rect
                    key={i}
                    x={i * barWidth + 0.5}
                    y={44 - h}
                    width={barWidth - 1}
                    height={h}
                    fill="var(--client)"
                  />
                );
              })}
            </svg>
          </div>
        ) : (
          <p className="font-prose text-xs text-slate">
            Hover a node for its shard; click or press Enter to pin.
          </p>
        )}
      </aside>
      ) : null}
    </div>
  );
}
