/**
 * The classroom labels drawn over the topology.
 *
 * The analogy has to be legible from the picture, not only from the prose
 * beneath it, or the reader is holding two diagrams in their head instead of
 * one. Both topology stages name the same three things in the same places.
 */
import type { TopologyGeometry } from "../views/Topology";

export function ClassroomLabels({ geometry }: { geometry: TopologyGeometry }) {
  const { size } = geometry;
  return (
    <g aria-hidden="true">
      <text
        y={30}
        textAnchor="middle"
        className="readout"
        fontSize={10}
        fill="var(--global)"
        letterSpacing="0.06em"
      >
        THE TEACHER
      </text>
      <text
        x={-size / 2 + 4}
        y={-size / 2 + 14}
        className="readout"
        fontSize={9}
        fill="var(--slate)"
      >
        each circle is a student · the bars are their notebook
      </text>
    </g>
  );
}
