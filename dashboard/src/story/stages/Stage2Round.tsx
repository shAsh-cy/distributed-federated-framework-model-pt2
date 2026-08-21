/**
 * Stage 2 — one round, played out of the recorded stream.
 *
 * This is the protocol, unedited: the teacher picks a cohort, each of them
 * studies their own notebook for as long as they actually took, the
 * corrections travel back in the order they actually arrived with weight
 * proportional to the bytes they actually cost, the merge lands, and the new
 * edition of the textbook goes out to everyone — including the students who
 * sat this round out, because that is what FedAvg does.
 *
 * The round shown is the one where a student missed the deadline. It is not
 * staged: the recorded run lost that client in that round.
 */
import { useMemo } from "react";

import { Topology } from "../../views/Topology";
import { ClassroomLabels } from "../Classroom";
import {
  COHORT_SIZE,
  DEADLINE_CLIENT,
  DEADLINE_ROUND,
  EVENTS,
  POPULATION,
  TOTAL_ROUNDS,
  roundAggregatedIndex,
  roundStartIndex,
  useStoryRun,
} from "../storyRun";
import { Caption, StillNotice } from "../ui";

const FROM = roundStartIndex(DEADLINE_ROUND);
const TO = roundAggregatedIndex(DEADLINE_ROUND);

type Beat = { at: number; text: string };

/** What the reader is looking at right now, in one plain sentence. */
function beatsForWindow(): Beat[] {
  const beats: Beat[] = [
    { at: FROM, text: "The teacher opens the round." },
    {
      at: FROM + 1,
      text: `The teacher picks ${COHORT_SIZE} students out of ${POPULATION}. Nobody else is asked.`,
    },
  ];
  const firstReport = EVENTS.findIndex(
    (e, i) => i > FROM && i <= TO && e.type === "client_reported",
  );
  if (firstReport > 0) {
    beats.push({
      at: firstReport,
      text: "Each of them studies their own notebook, for as long as they actually took.",
    });
    beats.push({
      at: firstReport + 1,
      text: "Corrections come back — the marked-up homework, never the notebook.",
    });
  }
  const drop = EVENTS.findIndex((e, i) => i > FROM && i <= TO && e.type === "client_dropped");
  if (drop > 0) {
    beats.push({
      at: drop,
      text: `${DEADLINE_CLIENT} is late. Their corrections miss the merge and are thrown away.`,
    });
  }
  beats.push({
    at: TO,
    text: "The teacher merges what did arrive into the class textbook and hands the new edition to everyone — including the students who sat out.",
  });
  return beats;
}

const BEATS = beatsForWindow();

export function Stage2Round({ active, still }: { active: boolean; still: boolean }) {
  const { run, cursor } = useStoryRun({ from: FROM, to: TO, active, still, loop: true });

  const beat = useMemo(() => {
    let current = BEATS[0]!;
    for (const candidate of BEATS) if (cursor >= candidate.at) current = candidate;
    return current;
  }, [cursor]);

  return (
    <div className="flex flex-col gap-4">
      <Topology
        run={run}
        highlightRound={DEADLINE_ROUND}
        onPin={() => {}}
        pinned={null}
        motionProfile="story"
        showDetail={false}
        forceRing
        overlay={(geometry) => <ClassroomLabels geometry={geometry} />}
      />
      <Caption>{still ? BEATS[BEATS.length - 1]!.text : beat.text}</Caption>
      <p className="story-measure font-prose text-base">
        Round <span className="readout">{DEADLINE_ROUND}</span> of a recorded{" "}
        <span className="readout">{TOTAL_ROUNDS}</span>-round run. Line thickness is the real size
        of each student&rsquo;s corrections; the order they arrive in is the order they arrived in.
        A dashed line is the student who was too late.
      </p>
      {still ? (
        <StillNotice>
          Animation is off. The round is shown at the moment after the merge: the cohort that
          reported, the student who missed the deadline dashed and dimmed, and the merged textbook
          at the centre.
        </StillNotice>
      ) : null}
    </div>
  );
}
