/**
 * Stage 1 — the problem. Ten students, ten notebooks, and a boundary the
 * notebooks cannot cross.
 *
 * The notebooks are the clients' real label histograms from the recorded
 * fixture, drawn by the topology itself. The overlay adds the boundary and one
 * attempted transfer: a notebook launched toward the teacher, stopped dead at
 * the line, and thrown back past its own desk before it settles. Spring
 * physics on a drawn boundary, not a metaphor drawn in prose.
 */
import { motion, useReducedMotion } from "framer-motion";

import { Topology, type TopologyGeometry } from "../../views/Topology";
import { ClassroomLabels } from "../Classroom";
import { reduceThrough } from "../storyRun";
import { Caption, StillNotice } from "../ui";

/** State after run_started alone: everyone present, nobody called on yet. */
const IDLE_RUN = reduceThrough(1);

function Boundary({ geometry, still }: { geometry: TopologyGeometry; still: boolean }) {
  const { radius, placements } = geometry;
  const wall = radius - 46;
  // A desk off to the side: the flight is then clear of the caption running
  // across the top of the frame.
  const desk = placements[Math.floor(placements.length / 4)];
  if (!desk) return null;

  // Rest position sits just outside the desk, so the notebook never covers the
  // histogram it is meant to be a picture of.
  const deskLength = Math.hypot(desk.x, desk.y) || 1;
  const origin = {
    x: (desk.x / deskLength) * (radius + 34),
    y: (desk.y / deskLength) * (radius + 34),
  };

  // Straight line from the desk toward the teacher, stopped at the wall.
  const length = Math.hypot(origin.x, origin.y) || 1;
  const ux = -origin.x / length;
  const uy = -origin.y / length;
  const impactX = origin.x + ux * (length - wall);
  const impactY = origin.y + uy * (length - wall);
  // The rebound carries it past its own desk before it settles back.
  const overshootX = origin.x - ux * 16;
  const overshootY = origin.y - uy * 16;

  return (
    <g aria-hidden="true">
      <circle
        r={wall}
        fill="none"
        stroke="var(--ink)"
        strokeWidth={1.25}
        strokeDasharray="2 5"
        opacity={0.55}
      />
      <motion.g
        initial={{ x: origin.x, y: origin.y, opacity: 0.95 }}
        animate={
          still
            ? { x: impactX, y: impactY, opacity: 0.95 }
            : {
                x: [origin.x, impactX, overshootX, origin.x],
                y: [origin.y, impactY, overshootY, origin.y],
                opacity: [0.95, 1, 0.9, 0.95],
              }
        }
        transition={
          still
            ? { duration: 0 }
            : {
                duration: 2.6,
                times: [0, 0.38, 0.62, 1],
                ease: ["easeIn", "easeOut", "easeOut"],
                repeat: Infinity,
                repeatDelay: 0.7,
              }
        }
      >
        <rect x={-9} y={-11} width={18} height={22} fill="var(--ground-raised)" stroke="var(--client)" strokeWidth={1.5} />
        <line x1={-9} y1={-4} x2={9} y2={-4} stroke="var(--client)" strokeWidth={1} />
        <line x1={-9} y1={1} x2={9} y2={1} stroke="var(--client)" strokeWidth={1} />
        <line x1={-9} y1={6} x2={4} y2={6} stroke="var(--client)" strokeWidth={1} />
      </motion.g>
    </g>
  );
}

export function Stage1Problem() {
  const reduced = useReducedMotion() ?? false;
  return (
    <div className="flex flex-col gap-4">
      <Topology
        run={IDLE_RUN}
        highlightRound={null}
        onPin={() => {}}
        pinned={null}
        motionProfile="story"
        showDetail={false}
        forceRing
        overlay={(geometry) => (
          <>
            <ClassroomLabels geometry={geometry} />
            <Boundary geometry={geometry} still={reduced} />
          </>
        )}
      />
      <Caption>
        The notebooks never leave. Only the lessons learned from them do.
      </Caption>
      <p className="story-measure font-prose text-base">
        Every student around the edge has their own notebook, and the bars on each desk are that
        student&rsquo;s real notebook: which chapters they have notes on, and how many. The
        notebooks are hospital records, or everything you have ever typed on your phone. They are
        not allowed to reach the middle of the room, so one is thrown at the teacher and comes
        straight back off the wall.
      </p>
      {reduced ? (
        <StillNotice>
          Animation is off, so the notebook is shown held against the boundary it cannot cross.
        </StillNotice>
      ) : null}
    </div>
  );
}
