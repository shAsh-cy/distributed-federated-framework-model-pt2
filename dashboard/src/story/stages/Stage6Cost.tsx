/**
 * Stage 6 — what it costs, honestly.
 *
 * Two real curves, twenty rounds each, both means over the recorded seeds:
 * the same federation with and without the privacy mechanism. The pooled
 * baseline sits above both as a rule, because the honest framing of this whole
 * project is that federation costs more than privacy does, and hiding that
 * would make the privacy number look worse than it is.
 *
 * Every point in both series comes from story_figures.json, which carries the
 * results file and JSON pointer behind it.
 */
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useMemo } from "react";

import { figure, series, value } from "../figures";
import { Caption, Fig, Sourced } from "../ui";

export function Stage6Cost({ active, still }: { active: boolean; still: boolean }) {
  const dp = series("dpCurve");
  const nodp = series("nodpCurve");

  const data = useMemo(
    () =>
      nodp.points.map((withoutPrivacy, index) => ({
        round: index + 1,
        withoutPrivacy: withoutPrivacy * 100,
        withPrivacy: (dp.points[index] ?? 0) * 100,
      })),
    [dp.points, nodp.points],
  );

  return (
    <div className="flex flex-col gap-4">
      <div className="h-[320px] w-full border border-rule bg-ground-raised p-2">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            key={active ? "playing" : "still"}
            data={data}
            margin={{ top: 12, right: 16, bottom: 8, left: 0 }}
          >
            <CartesianGrid stroke="var(--rule)" strokeDasharray="2 4" vertical={false} />
            <XAxis
              dataKey="round"
              stroke="var(--ink)"
              tick={{ fill: "var(--slate)", fontSize: 11 }}
              label={{
                value: "rounds",
                position: "insideBottomRight",
                offset: -4,
                fill: "var(--slate)",
                fontSize: 11,
              }}
            />
            <YAxis
              domain={[0, 100]}
              stroke="var(--ink)"
              tick={{ fill: "var(--slate)", fontSize: 11 }}
              width={44}
              unit="%"
            />
            <Tooltip
              contentStyle={{
                background: "var(--ground-raised)",
                border: "1px solid var(--rule)",
                fontSize: 12,
              }}
              formatter={(v: number) => v.toFixed(1) + " %"}
              labelFormatter={(round) => "round " + round}
            />
            <ReferenceLine
              y={value("pooled") * 100}
              stroke="var(--ink)"
              strokeDasharray="4 4"
              label={{
                value: "all the data in one place " + figure("pooled"),
                position: "insideTopLeft",
                fill: "var(--ink)",
                fontSize: 11,
              }}
            />
            <Line
              type="monotone"
              dataKey="withoutPrivacy"
              name="no privacy"
              stroke="var(--global)"
              strokeWidth={2}
              dot={false}
              isAnimationActive={!still}
              animationDuration={2200}
            />
            <Line
              type="monotone"
              dataKey="withPrivacy"
              name="with privacy"
              stroke="var(--budget)"
              strokeWidth={2}
              dot={false}
              isAnimationActive={!still}
              animationDuration={2200}
              animationBegin={200}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <ul className="flex flex-wrap gap-x-6 gap-y-1 font-prose text-sm">
        <li className="flex items-center gap-2">
          <span className="inline-block h-0.5 w-6" style={{ background: "var(--global)" }} />
          no privacy, ends at <Fig name="nodpFinal" />
        </li>
        <li className="flex items-center gap-2">
          <span className="inline-block h-0.5 w-6" style={{ background: "var(--budget)" }} />
          with privacy, ends at <Fig name="dpFinal" />
        </li>
      </ul>

      <Caption>
        Privacy cost <Fig name="dpCost" /> points of accuracy here. We measured it rather than
        hiding it.
      </Caption>

      <p className="story-measure font-prose text-base">
        And the wider frame, in one line: federation itself costs more than privacy does —{" "}
        <Fig name="pooled" /> if all the data sat in one place, <Fig name="nodpFinal" /> federated
        without privacy, <Fig name="dpFinal" /> with it.
      </p>
      <p className="story-measure font-prose text-base">
        The gap is not fixed. Let the same federation run for <Fig name="longRounds" /> rounds
        instead of <Fig name="rounds" /> and the no-privacy line reaches{" "}
        <Fig name="longRoundsAcc" />. Time buys back some of what distribution costs.
      </p>
      <Sourced names={["nodpFinal", "dpFinal", "pooled", "longRoundsAcc"]} />
    </div>
  );
}
