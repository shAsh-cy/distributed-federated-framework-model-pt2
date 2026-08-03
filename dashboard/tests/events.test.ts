/**
 * Replay reducer: run state derives from the stream alone, and the derived
 * state honours the protocol's semantics (drop-dimming, curve building,
 * duplicate idempotency). Exercised against the recorded scripted stream
 * plus hand-built sequences for the edge semantics.
 */
import { describe, expect, it } from "vitest";

import liveDemo from "../fixtures/live_demo_scripted_events.json";
import {
  applyEvent,
  emptyRunState,
  reduceEvents,
  type FlEvent,
} from "../src/lib/events";

const demoEvents = liveDemo as unknown as FlEvent[];

describe("reduceEvents over the recorded scripted stream", () => {
  const state = reduceEvents(demoEvents);

  it("reconstructs the population with histograms", () => {
    expect(state.clientOrder).toHaveLength(10);
    const first = state.clients.get("client-0");
    expect(first?.info.label_histogram).toHaveLength(10);
    expect(first?.info.num_examples).toBeGreaterThan(0);
  });

  it("builds the full curve and completes", () => {
    expect(state.curve).toHaveLength(6);
    expect(state.status).toBe("completed");
    expect(state.finalAccuracy).toBeCloseTo(0.8215, 4);
    expect(state.curve.map((p) => p.round)).toEqual([1, 2, 3, 4, 5, 6]);
  });

  it("keeps sequence bookkeeping contiguous", () => {
    expect(state.lastSeq).toBe(demoEvents.length - 1);
    expect(state.log).toHaveLength(demoEvents.length);
  });

  it("is idempotent on replayed duplicates", () => {
    const replayedTwice = reduceEvents([...demoEvents, ...demoEvents]);
    expect(replayedTwice.curve).toHaveLength(6);
    expect(replayedTwice.log).toHaveLength(demoEvents.length);
  });
});

describe("drop semantics", () => {
  const base: FlEvent[] = [
    {
      type: "run_started",
      schema_version: 1,
      run_id: "r",
      seq: 0,
      ts: 0,
      config: {},
      num_classes: 3,
      clients: [
        { client_id: "client-0", num_examples: 5, label_histogram: [5, 0, 0] },
        { client_id: "client-1", num_examples: 5, label_histogram: [0, 5, 0] },
      ],
    },
    { type: "round_started", schema_version: 1, run_id: "r", seq: 1, ts: 0, round: 1, model_version: 0 },
    {
      type: "client_sampled",
      schema_version: 1,
      run_id: "r",
      seq: 2,
      ts: 0,
      round: 1,
      client_id: "client-0",
      framework: "tensorflow",
    },
    {
      type: "client_dropped",
      schema_version: 1,
      run_id: "r",
      seq: 3,
      ts: 0,
      round: 1,
      client_id: "client-0",
      reason: "deadline",
    },
  ];

  it("a deadline-misser stays dimmed through the following round", () => {
    let state = reduceEvents(base);
    expect(state.clients.get("client-0")?.phase).toBe("dropped");

    state = applyEvent(state, {
      type: "round_started",
      schema_version: 1,
      run_id: "r",
      seq: 4,
      ts: 0,
      round: 2,
      model_version: 1,
    });
    expect(state.clients.get("client-0")?.phase).toBe("dropped"); // still dimmed
    expect(state.clients.get("client-1")?.phase).toBe("idle");

    state = applyEvent(state, {
      type: "round_started",
      schema_version: 1,
      run_id: "r",
      seq: 5,
      ts: 0,
      round: 3,
      model_version: 2,
    });
    expect(state.clients.get("client-0")?.phase).toBe("idle"); // recovered
  });

  it("failure marks the state without erasing history", () => {
    const state = reduceEvents([
      ...base,
      {
        type: "run_failed",
        schema_version: 1,
        run_id: "r",
        seq: 4,
        ts: 0,
        error: "boom",
        rounds_completed: 0,
      },
    ]);
    expect(state.status).toBe("failed");
    expect(state.error).toBe("boom");
    expect(state.log).toHaveLength(5);
  });

  it("starts from a well-defined empty state", () => {
    const state = emptyRunState();
    expect(state.status).toBe("waiting");
    expect(state.curve).toHaveLength(0);
  });
});
