/**
 * Run console: topology hero, plotter-style curves, ochre budget meter,
 * bytes-per-round readout, monospace event log. Views are linked: hovering a
 * curve point highlights that round in the topology and scrolls the log.
 * Everything except the topology is still.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { FlEvent } from "../lib/events";
import { useRunEvents } from "../lib/runData";
import { Meter } from "../ui/primitives";
import { Topology } from "./Topology";

function formatBytes(n: number | null): string {
  if (n === null) return "—";
  if (n >= 1 << 20) return `${(n / (1 << 20)).toFixed(1)} MiB`;
  if (n >= 1 << 10) return `${(n / (1 << 10)).toFixed(1)} KiB`;
  return `${n} B`;
}

function describeEvent(event: FlEvent): string {
  switch (event.type) {
    case "run_started":
      return `run started · ${event.clients.length} clients · ${event.num_classes} classes`;
    case "round_started":
      return `round ${event.round} open · model v${event.model_version}`;
    case "client_sampled":
      return `  ${event.client_id} sampled${event.framework ? ` (${event.framework})` : ""}`;
    case "client_reported":
      return `  ${event.client_id} reported · n=${event.num_examples} · acc ${
        event.local_accuracy?.toFixed(4) ?? "—"
      } · ${event.wall_clock_seconds?.toFixed(1) ?? "—"}s`;
    case "client_dropped":
      return `  ${event.client_id} dropped · ${event.reason}`;
    case "round_aggregated":
      return `round ${event.round} aggregated · acc ${
        event.global_accuracy?.toFixed(4) ?? "—"
      }${event.cumulative_epsilon !== null ? ` · ε ${event.cumulative_epsilon.toFixed(3)}` : ""}`;
    case "run_completed":
      return `run completed · ${event.rounds_completed} rounds${
        event.stopped_early ? " · stopped early" : ""
      } · final ${event.final_accuracy?.toFixed(4) ?? "—"}`;
    case "run_failed":
      return `run FAILED after ${event.rounds_completed} rounds · ${event.error}`;
  }
}

export function ConsoleView({ runId }: { runId: string | null }) {
  const { state, socket } = useRunEvents(runId);
  const [highlightRound, setHighlightRound] = useState<number | null>(null);
  const [pinned, setPinned] = useState<string | null>(null);
  const logRef = useRef<HTMLOListElement>(null);

  useEffect(() => {
    const el = logRef.current;
    if (el && highlightRound === null) el.scrollTop = el.scrollHeight;
  }, [state.log.length, highlightRound]);

  const curveData = useMemo(
    () =>
      state.curve.map((point) => ({
        round: point.round,
        accuracy: point.accuracy,
        loss: point.loss,
      })),
    [state.curve],
  );

  const lastPoint = state.curve[state.curve.length - 1] ?? null;
  const dpRun = state.curve.some((p) => p.epsilon !== null);
  const epsilonNow = lastPoint?.epsilon ?? null;
  const epsilonTarget =
    state.status === "completed" && epsilonNow !== null ? epsilonNow : null;

  if (!runId) {
    return (
      <p className="font-prose text-base">
        No run selected. Configure a new run, or open one from History — imported runs
        replay their full event record here.
      </p>
    );
  }

  return (
    <div className="flex flex-col gap-6">
      {/* status line: connection state is explicit, never a silent freeze */}
      <div className="flex flex-wrap items-baseline justify-between gap-2 border-b border-rule pb-2">
        <div className="flex items-baseline gap-4">
          <span className="font-head text-xs uppercase tracking-head text-slate">Round</span>
          <span className="readout text-2xl" aria-live="polite">
            {String(state.currentRound).padStart(3, "0")}
          </span>
          <span className="readout text-xs text-slate">model v{state.modelVersion}</span>
          <span className="readout text-xs text-slate">status {state.status}</span>
        </div>
        <span className="readout text-xs" role="status">
          {socket === null && "connecting"}
          {socket?.kind === "connecting" && "connecting…"}
          {socket?.kind === "open" && "● stream open"}
          {socket?.kind === "reconnecting" &&
            `reconnecting… retry ${socket.attempt} — stream will resume from last event`}
          {socket?.kind === "closed" && `stream closed · ${socket.reason}`}
        </span>
      </div>

      {state.status === "failed" && state.error ? (
        <p className="border border-client p-3 font-prose text-sm text-client">
          This run failed: {state.error}. The event record up to the failure is complete
          below; start a new run from Configure when ready.
        </p>
      ) : null}

      {state.clientOrder.length > 0 ? (
        <Topology
          run={state}
          highlightRound={highlightRound}
          onPin={setPinned}
          pinned={pinned}
        />
      ) : (
        <p className="font-prose text-sm text-slate">
          Waiting for run_started — the topology draws itself from the first event.
        </p>
      )}

      <div className="grid gap-6 lg:grid-cols-[2fr_1fr]">
        <section aria-label="Accuracy and loss curves" className="min-w-0">
          <h2 className="mb-1 font-head text-sm uppercase tracking-head">
            Global accuracy / loss
          </h2>
          <div className="h-56 w-full">
            <ResponsiveContainer>
              <LineChart
                data={curveData}
                margin={{ top: 4, right: 8, bottom: 0, left: -18 }}
                onMouseMove={(chart) => {
                  const label = chart?.activeLabel;
                  setHighlightRound(typeof label === "number" ? label : null);
                }}
                onMouseLeave={() => setHighlightRound(null)}
              >
                <CartesianGrid stroke="var(--rule)" strokeDasharray="2 4" />
                <XAxis
                  dataKey="round"
                  stroke="var(--ink)"
                  tick={{ fontFamily: "Consolas, monospace", fontSize: 10 }}
                />
                <YAxis
                  stroke="var(--ink)"
                  domain={[0, 1]}
                  tick={{ fontFamily: "Consolas, monospace", fontSize: 10 }}
                />
                <Tooltip
                  isAnimationActive={false}
                  contentStyle={{
                    background: "var(--ground-raised)",
                    border: "1px solid var(--rule)",
                    fontFamily: "Consolas, monospace",
                    fontSize: 11,
                  }}
                />
                <Line
                  type="linear"
                  dataKey="accuracy"
                  stroke="var(--global)"
                  strokeWidth={1.75}
                  dot={false}
                  isAnimationActive={false}
                />
                <Line
                  type="linear"
                  dataKey="loss"
                  stroke="var(--client)"
                  strokeWidth={1}
                  dot={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <p className="sr-only">
            {`Accuracy by round: ${curveData
              .map((p) => `round ${p.round}: ${p.accuracy?.toFixed(4) ?? "no value"}`)
              .join("; ")}`}
          </p>
        </section>

        <section className="flex flex-col gap-4">
          {dpRun ? (
            <Meter
              label="Privacy budget ε"
              value={epsilonNow ?? 0}
              max={epsilonTarget ?? Math.max(1, (epsilonNow ?? 0) * 1.2)}
              tone="budget"
              format={(v) =>
                epsilonTarget !== null
                  ? `${v.toFixed(3)} / ${epsilonTarget.toFixed(3)}`
                  : `${v.toFixed(3)} spent`
              }
            />
          ) : (
            <div className="font-prose text-xs text-slate">
              No differential privacy on this run — the budget meter stays dark.
            </div>
          )}
          <div className="flex flex-col gap-1">
            <span className="font-head text-xs uppercase tracking-head">
              Communication, last round
            </span>
            <span className="readout text-sm">
              → clients {formatBytes(lastPoint?.bytesSent ?? null)} · ← server{" "}
              {formatBytes(lastPoint?.bytesReceived ?? null)}
            </span>
            <span className="readout text-xs text-slate">
              total{" "}
              {formatBytes(
                state.curve.reduce(
                  (sum, p) => sum + (p.bytesSent ?? 0) + (p.bytesReceived ?? 0),
                  0,
                ) || null,
              )}
            </span>
          </div>
        </section>
      </div>

      <section aria-label="Event log">
        <h2 className="mb-1 font-head text-sm uppercase tracking-head">Event log</h2>
        <ol
          ref={logRef}
          className="readout max-h-56 overflow-y-auto border border-rule bg-ground-raised p-2 text-xs leading-5"
        >
          {state.log.map((event) => {
            const eventRound = "round" in event ? event.round : null;
            const highlighted = highlightRound !== null && eventRound === highlightRound;
            return (
              <li
                key={event.seq}
                className={highlighted ? "bg-rule/60" : undefined}
                data-round={eventRound ?? undefined}
              >
                <span className="text-slate">{String(event.seq).padStart(4, "0")}</span>{" "}
                {describeEvent(event)}
              </li>
            );
          })}
        </ol>
      </section>
    </div>
  );
}
