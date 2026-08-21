/**
 * Replay of the recorded run that drives story mode.
 *
 * The same scripted fixture the mocked e2e streams
 * (fixtures/live_demo_scripted_events.json: real protocol through the real
 * event store, values from recorded measurements) reduced by the same
 * `applyEvent` the console uses. Story mode starts no run, needs no backend
 * and needs no mock mode — it is the recorded stream, paced for reading.
 *
 * Pacing comes from the events themselves rather than a constant tick: a
 * client's update arrives after a delay proportional to its measured
 * wall-clock, so the student who took 5.6 s visibly takes longer than the one
 * who took 3.6 s. That is the same measurement the topology breathes at.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import rawEvents from "../../fixtures/live_demo_scripted_events.json";
import { applyEvent, emptyRunState, type FlEvent, type RunState } from "../lib/events";

export const EVENTS = rawEvents as FlEvent[];

/** Wall-clock seconds → milliseconds on screen. A round lands near five seconds. */
const WALL_CLOCK_SCALE = 120;

const FIXED_DELAYS: Record<string, number> = {
  run_started: 600,
  round_started: 420,
  client_sampled: 220,
  client_dropped: 700,
  round_aggregated: 950,
  run_completed: 0,
  run_failed: 0,
};

/** How long the stage should dwell on `event` before applying the next one. */
export function delayFor(event: FlEvent): number {
  if (event.type === "client_reported") {
    return Math.round((event.wall_clock_seconds ?? 4) * WALL_CLOCK_SCALE);
  }
  return FIXED_DELAYS[event.type] ?? 300;
}

export function reduceThrough(upToIndexExclusive: number): RunState {
  return EVENTS.slice(0, Math.max(0, upToIndexExclusive)).reduce(applyEvent, emptyRunState());
}

/** Index of the `round_started` event for `round`. */
export function roundStartIndex(round: number): number {
  return EVENTS.findIndex((e) => e.type === "round_started" && e.round === round);
}

/** Index of the `round_aggregated` event for `round`. */
export function roundAggregatedIndex(round: number): number {
  return EVENTS.findIndex((e) => e.type === "round_aggregated" && e.round === round);
}

/** The round in which the recorded run lost a client to the deadline. */
export const DEADLINE_ROUND: number = (() => {
  const dropped = EVENTS.find((e) => e.type === "client_dropped");
  return dropped && "round" in dropped ? dropped.round : 1;
})();

export const DEADLINE_CLIENT: string = (() => {
  const dropped = EVENTS.find((e) => e.type === "client_dropped");
  return dropped && "client_id" in dropped ? dropped.client_id : "";
})();

export const TOTAL_ROUNDS: number = EVENTS.filter((e) => e.type === "round_started").length;

export const COHORT_SIZE: number = EVENTS.filter(
  (e) => e.type === "client_sampled" && e.round === 1,
).length;

export const POPULATION: number = (() => {
  const started = EVENTS[0];
  return started && started.type === "run_started" ? started.clients.length : 0;
})();

export type StoryRun = {
  /** State after every event in [0, cursor). */
  run: RunState;
  cursor: number;
  playing: boolean;
  /** True once the window has been played to its end. */
  finished: boolean;
  setPlaying: (playing: boolean) => void;
  toggle: () => void;
  restart: () => void;
};

/**
 * Replays events [`from`, `to`] on a timer.
 *
 * `still` (reduced motion, or a stage that wants the end state) skips straight
 * to the end: every stage's content is present either way, which is the whole
 * reduced-motion contract.
 */
export function useStoryRun(opts: {
  from: number;
  to: number;
  active: boolean;
  still: boolean;
  loop?: boolean;
}): StoryRun {
  const { from, to, active, still, loop = false } = opts;
  const base = useMemo(() => reduceThrough(from), [from]);
  const [cursor, setCursor] = useState(still ? to + 1 : from);
  const [playing, setPlaying] = useState(active && !still);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // A stage becoming active (or reduced motion turning on) resets the window.
  useEffect(() => {
    setCursor(still || !active ? to + 1 : from);
    setPlaying(active && !still);
  }, [from, to, active, still]);

  useEffect(() => {
    if (!playing || cursor > to) return;
    const event = EVENTS[cursor];
    if (!event) return;
    timer.current = setTimeout(() => {
      setCursor((c) => {
        if (c >= to) {
          if (loop) return from;
          return c + 1;
        }
        return c + 1;
      });
    }, delayFor(event));
    return () => {
      if (timer.current) clearTimeout(timer.current);
    };
  }, [playing, cursor, to, from, loop]);

  const run = useMemo(
    () => EVENTS.slice(from, Math.min(cursor, to + 1)).reduce(applyEvent, base),
    [base, from, cursor, to],
  );

  const restart = useCallback(() => {
    setCursor(from);
    setPlaying(true);
  }, [from]);

  const toggle = useCallback(() => {
    if (cursor > to) {
      restart();
      return;
    }
    setPlaying((p) => !p);
  }, [cursor, to, restart]);

  return { run, cursor, playing, finished: cursor > to, setPlaying, toggle, restart };
}
