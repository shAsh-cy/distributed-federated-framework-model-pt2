/**
 * Results: the permanent artefacts. Epsilon–accuracy tradeoff and the
 * cohort-size curves, drawn from the imported multi-seed aggregates. Seed
 * ranges render as bands (min..max of the recorded seeds) — measured spread,
 * shown, never hidden, never collapsed to a point.
 */
import { useQuery } from "@tanstack/react-query";
import { useMemo } from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { api, type RunSummary } from "../lib/api";
import { Skeleton } from "../ui/primitives";

type BandPoint = {
  x: number;
  mean: number;
  band: [number, number];
  label: string;
};

/** Parse "…/m=50" style keys into their numeric axis value. */
function parseAxisValue(groupKey: string, prefix: string): number | null {
  const match = groupKey.match(new RegExp(`${prefix}=([0-9.]+)`));
  return match ? Number(match[1]) : null;
}

function useAggregates() {
  const runs = useQuery({ queryKey: ["runs"], queryFn: api.runs });
  const details = useQuery({
    queryKey: ["aggregate-details", runs.data?.length],
    enabled: !!runs.data,
    queryFn: async () => {
      const aggregates = (runs.data ?? []).filter((r) => r.is_aggregate);
      const loaded = await Promise.all(aggregates.map((r) => api.run(r.id)));
      return loaded;
    },
  });
  return { runs, details };
}

function bandPoints(
  aggregates: { group_key?: string | null; final_metrics?: Record<string, unknown> | null }[],
  keyPrefix: string,
  groupFilter: string,
): BandPoint[] {
  const points: BandPoint[] = [];
  for (const run of aggregates) {
    const groupKey = run.group_key ?? "";
    if (!groupKey.includes(groupFilter)) continue;
    const x = parseAxisValue(groupKey, keyPrefix);
    const metrics = run.final_metrics as
      | { mean_final?: number; final_per_seed?: number[] }
      | null
      | undefined;
    if (x === null || !metrics?.mean_final || !metrics.final_per_seed?.length) continue;
    points.push({
      x,
      mean: metrics.mean_final,
      band: [Math.min(...metrics.final_per_seed), Math.max(...metrics.final_per_seed)],
      label: groupKey,
    });
  }
  return points.sort((a, b) => a.x - b.x);
}

function BandChart({
  title,
  points,
  xLabel,
  xScale,
}: {
  title: string;
  points: BandPoint[];
  xLabel: string;
  xScale?: "log" | "linear";
}) {
  const data = points.map((p) => ({ ...p, bandLow: p.band[0], bandHigh: p.band[1] }));
  return (
    <section aria-label={title} className="min-w-0">
      <h2 className="mb-1 font-head text-sm uppercase tracking-head">{title}</h2>
      {points.length === 0 ? (
        <p className="font-prose text-xs text-slate">
          No multi-seed aggregates matched — import the repo history to populate this chart.
        </p>
      ) : (
        <div className="h-64 w-full">
          <ResponsiveContainer>
            <ComposedChart data={data} margin={{ top: 4, right: 8, bottom: 16, left: -12 }}>
              <CartesianGrid stroke="var(--rule)" strokeDasharray="2 4" />
              <XAxis
                dataKey="x"
                type="number"
                scale={xScale ?? "linear"}
                domain={["dataMin", "dataMax"]}
                stroke="var(--ink)"
                tick={{ fontFamily: "Consolas, monospace", fontSize: 10 }}
                label={{
                  value: xLabel,
                  position: "insideBottom",
                  offset: -8,
                  style: { fontFamily: "Consolas, monospace", fontSize: 10 },
                }}
              />
              <YAxis
                domain={[0, 1]}
                stroke="var(--ink)"
                tick={{ fontFamily: "Consolas, monospace", fontSize: 10 }}
              />
              <Tooltip
                isAnimationActive={false}
                formatter={(value: number | [number, number], name: string) =>
                  Array.isArray(value)
                    ? [`${value[0].toFixed(4)}–${value[1].toFixed(4)}`, "seed range"]
                    : [value.toFixed(4), name]
                }
                contentStyle={{
                  background: "var(--ground-raised)",
                  border: "1px solid var(--rule)",
                  fontFamily: "Consolas, monospace",
                  fontSize: 11,
                }}
              />
              <Area
                dataKey="band"
                stroke="none"
                fill="var(--global)"
                fillOpacity={0.16}
                isAnimationActive={false}
              />
              <Line
                dataKey="mean"
                stroke="var(--global)"
                strokeWidth={1.75}
                dot={{ r: 2.5, fill: "var(--global)" }}
                isAnimationActive={false}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      )}
      <p className="sr-only">
        {points
          .map(
            (p) =>
              `${xLabel} ${p.x}: mean ${p.mean.toFixed(4)}, seed range ${p.band[0].toFixed(4)} to ${p.band[1].toFixed(4)}`,
          )
          .join("; ")}
      </p>
    </section>
  );
}

/** The three shipped gRPC configurations: epsilon vs final accuracy. */
function tradeoffPoints(runs: RunSummary[]): { epsilon: number | null; label: string; id: string }[] {
  return runs
    .filter((r) => r.label.startsWith("grpc/") && !r.label.startsWith("grpc/pure")
      && !r.label.startsWith("grpc/mixed"))
    .map((r) => ({ epsilon: null, label: r.label, id: r.id }));
}

export function ResultsView() {
  const { runs, details } = useAggregates();

  const grpcDetails = useQuery({
    queryKey: ["grpc-details", runs.data?.length],
    enabled: !!runs.data,
    queryFn: async () => {
      const targets = tradeoffPoints(runs.data ?? []);
      return Promise.all(targets.map((t) => api.run(t.id)));
    },
  });

  const aggregates = useMemo(() => details.data ?? [], [details.data]);

  const cohort = useMemo(
    () => bandPoints(aggregates, "m", "_femnist_sweep"),
    [aggregates],
  );
  const clipBracket = useMemo(
    () => bandPoints(aggregates, "clip", "_femnist_bracket"),
    [aggregates],
  );

  const tradeoff = useMemo(() => {
    const rows = (grpcDetails.data ?? [])
      .map((run) => {
        const metrics = run.final_metrics as
          | { final_accuracy?: number; epsilon?: number | null }
          | null;
        if (!metrics?.final_accuracy) return null;
        return {
          label: run.label,
          accuracy: metrics.final_accuracy,
          epsilon: metrics.epsilon ?? null,
        };
      })
      .filter((x): x is NonNullable<typeof x> => x !== null);
    return rows;
  }, [grpcDetails.data]);

  if (runs.isLoading || details.isLoading) return <Skeleton lines={8} label="Results" />;

  return (
    <div className="flex flex-col gap-8">
      <section aria-label="Epsilon accuracy tradeoff">
        <h2 className="mb-1 font-head text-sm uppercase tracking-head">
          Privacy–accuracy, the three shipped configurations
        </h2>
        <p className="mb-2 font-prose text-xs text-slate">
          Single recorded gRPC runs (the shipped configs, un-tuned clip) — labelled as such;
          the multi-seed story lives in the banded charts below.
        </p>
        <div className="h-56 w-full">
          <ResponsiveContainer>
            <ComposedChart
              data={tradeoff.map((t) => ({
                ...t,
                x: t.epsilon ?? 1000, // no-DP plotted at the right edge, labelled ∞
              }))}
              margin={{ top: 4, right: 16, bottom: 16, left: -12 }}
            >
              <CartesianGrid stroke="var(--rule)" strokeDasharray="2 4" />
              <XAxis
                dataKey="x"
                type="number"
                scale="log"
                domain={[1, 1200]}
                stroke="var(--ink)"
                tick={{ fontFamily: "Consolas, monospace", fontSize: 10 }}
                tickFormatter={(v: number) => (v >= 1000 ? "∞ (no DP)" : String(v))}
                label={{
                  value: "epsilon (log)",
                  position: "insideBottom",
                  offset: -8,
                  style: { fontFamily: "Consolas, monospace", fontSize: 10 },
                }}
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
              <Scatter dataKey="accuracy" fill="var(--budget)" isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        <p className="sr-only">
          {tradeoff
            .map(
              (t) =>
                `${t.label}: epsilon ${t.epsilon ?? "none (no DP)"}, final accuracy ${t.accuracy.toFixed(4)}`,
            )
            .join("; ")}
        </p>
      </section>

      <BandChart
        title="Cohort size at fixed ε = 6.228 (FEMNIST, fixed shards, 3 seeds)"
        points={cohort}
        xLabel="clients per round"
        xScale="log"
      />
      <BandChart
        title="Clipping norm at m = 200 (FEMNIST, 3 seeds)"
        points={clipBracket}
        xLabel="clipping norm S"
        xScale="log"
      />
    </div>
  );
}
