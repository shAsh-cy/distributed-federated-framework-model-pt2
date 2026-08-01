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
 */
import { motion, useReducedMotion } from "framer-motion";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { ClientState, RunState } from "../lib/events";

const RING_THRESHOLD = 60;

type NodePlacement = { id: string; x: number; y: number; angle: number };

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

export function Topology({
  run,
  highlightRound,
  onPin,
  pinned,
}: {
  run: RunState;
  highlightRound: number | null;
  onPin: (clientId: string | null) => void;
  pinned: string | null;
}) {
  const reduced = useReducedMotion();
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
                  r={3.2}
                  fill="var(--client)"
                  initial={{ cx: x, cy: y, opacity: 0 }}
                  animate={{ cx: 0, cy: 0, opacity: [0, 1, 1, 0] }}
                  transition={{
                    duration: 0.9,
                    delay: (c.arrivalOrder ?? 0) * 0.18,
                    ease: "easeIn",
                  }}
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
          animate={
            reduced
              ? undefined
              : aggregatedThisRound
                ? { scale: [1, 1.25, 1] }
                : { scale: 1 }
          }
          transition={{ duration: 0.5 }}
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
            animate={{ r: radius, opacity: 0 }}
            transition={{ duration: 0.8, ease: "easeOut" }}
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
    </div>
  );
}
