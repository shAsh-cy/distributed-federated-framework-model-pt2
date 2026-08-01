/**
 * Run history: every run the coordinator knows, live and imported. Honesty
 * markers are the point — an imported run without a per-round record is
 * visibly different from a streamed one, and nothing interpolates a curve
 * that was never measured. Select two event-bearing runs to overlay their
 * accuracy curves on shared axes.
 */
import { useQuery } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { api, type RunSummary } from "../lib/api";
import { fetchRunState } from "../lib/runData";
import { Button, Skeleton } from "../ui/primitives";

function runKind(run: RunSummary): "aggregate" | "imported" | "live" {
  if (run.is_aggregate) return "aggregate";
  if (run.source === "imported") return "imported";
  return "live";
}

export function HistoryView({ onOpenRun }: { onOpenRun: (runId: string) => void }) {
  const runs = useQuery({ queryKey: ["runs"], queryFn: api.runs });
  const [selected, setSelected] = useState<string[]>([]);
  const [overlay, setOverlay] = useState<
    { label: string; points: { round: number; accuracy: number | null }[] }[] | null
  >(null);
  const [overlayNote, setOverlayNote] = useState<string | null>(null);
  const [loadingOverlay, setLoadingOverlay] = useState(false);

  const rows = useMemo(() => runs.data ?? [], [runs.data]);

  function toggle(id: string) {
    setSelected((current) =>
      current.includes(id)
        ? current.filter((x) => x !== id)
        : [...current.slice(-1), id],
    );
  }

  async function compare() {
    if (selected.length !== 2) return;
    setLoadingOverlay(true);
    setOverlayNote(null);
    try {
      const states = await Promise.all(selected.map((id) => fetchRunState(id)));
      const series = states.map((state, i) => {
        const run = rows.find((r) => r.id === selected[i]);
        return {
          label: run?.label || selected[i]!.slice(0, 8),
          points: state.curve.map((p) => ({ round: p.round, accuracy: p.accuracy })),
        };
      });
      const empty = series.filter((s) => s.points.length === 0);
      if (empty.length > 0) {
        setOverlayNote(
          `${empty.map((s) => s.label).join(" and ")} recorded final metrics only — there is no per-round curve to draw, so it is omitted rather than invented.`,
        );
      }
      setOverlay(series.filter((s) => s.points.length > 0));
    } finally {
      setLoadingOverlay(false);
    }
  }

  if (runs.isLoading) return <Skeleton lines={8} label="Run history" />;
  if (runs.isError)
    return (
      <p className="font-prose text-base">
        The run list did not load. Check the coordinator API, then reload this view.
      </p>
    );
  if (rows.length === 0)
    return (
      <p className="font-prose text-base">
        No runs yet. Configure one — or import the repo's history with
        <span className="readout"> coordinator.importer.import_history</span> to fill this
        view with the 97 recorded experiments.
      </p>
    );

  const colours = ["var(--global)", "var(--client)"];

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-baseline justify-between">
        <h2 className="font-head text-lg uppercase tracking-head">
          {rows.length} runs
        </h2>
        <div className="flex items-center gap-3">
          <span className="readout text-xs text-slate">
            {selected.length}/2 selected for overlay
          </span>
          <Button
            tone="primary"
            disabled={selected.length !== 2 || loadingOverlay}
            onClick={() => void compare()}
          >
            {loadingOverlay ? "Loading curves" : "Compare"}
          </Button>
        </div>
      </div>

      {overlay ? (
        <section aria-label="Accuracy overlay" className="border border-rule p-3">
          <div className="h-56 w-full">
            <ResponsiveContainer>
              <LineChart margin={{ top: 4, right: 8, bottom: 0, left: -18 }}>
                <CartesianGrid stroke="var(--rule)" strokeDasharray="2 4" />
                <XAxis
                  dataKey="round"
                  type="number"
                  domain={["dataMin", "dataMax"]}
                  stroke="var(--ink)"
                  tick={{ fontFamily: "Consolas, monospace", fontSize: 10 }}
                />
                <YAxis
                  domain={[0, 1]}
                  stroke="var(--ink)"
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
                <Legend wrapperStyle={{ fontFamily: "Consolas, monospace", fontSize: 11 }} />
                {overlay.map((series, i) => (
                  <Line
                    key={series.label}
                    data={series.points}
                    dataKey="accuracy"
                    name={series.label}
                    stroke={colours[i % colours.length]}
                    strokeWidth={1.75}
                    dot={false}
                    isAnimationActive={false}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
          {overlayNote ? (
            <p className="mt-2 font-prose text-xs text-slate">{overlayNote}</p>
          ) : null}
        </section>
      ) : null}

      <table className="w-full border-collapse text-left">
        <thead>
          <tr className="border-b border-ink font-head text-xs uppercase tracking-head">
            <th className="py-1 pr-2">Sel</th>
            <th className="py-1 pr-2">Run</th>
            <th className="py-1 pr-2">Status</th>
            <th className="py-1 pr-2">Kind</th>
            <th className="py-1 pr-2">Seed</th>
            <th className="py-1 pr-2">Final acc</th>
            <th className="py-1" />
          </tr>
        </thead>
        <tbody className="readout text-xs">
          {rows.map((run) => {
            const kind = runKind(run);
            return (
              <tr key={run.id} className="border-b border-rule">
                <td className="py-1 pr-2">
                  <input
                    type="checkbox"
                    aria-label={`Select ${run.label || run.id} for comparison`}
                    checked={selected.includes(run.id)}
                    onChange={() => toggle(run.id)}
                  />
                </td>
                <td className="max-w-[26ch] truncate py-1 pr-2" title={run.label}>
                  {run.label || run.id.slice(0, 8)}
                </td>
                <td className="py-1 pr-2">{run.status}</td>
                <td className="py-1 pr-2">
                  {kind === "aggregate" ? (
                    <span title="Multi-seed summary carrying mean and range">
                      Σ mean±range
                    </span>
                  ) : kind === "imported" ? (
                    <span title="Imported from committed results; not live-streamed">
                      ⬡ imported
                    </span>
                  ) : (
                    "live"
                  )}
                </td>
                <td className="py-1 pr-2">{run.seed ?? "—"}</td>
                <td className="py-1 pr-2">—</td>
                <td className="py-1">
                  <button
                    className="text-global underline-offset-2 hover:underline"
                    onClick={() => onOpenRun(run.id)}
                  >
                    open
                  </button>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
