/**
 * Run-state hooks: live subscription and one-shot replay collection.
 *
 * Terminal runs replay their whole stream and the server closes the socket,
 * so "fetch the history of a finished run" and "follow a live run" are the
 * same code path — the replay guarantee doing its job.
 */
import { useEffect, useRef, useState } from "react";

import {
  applyEvent,
  emptyRunState,
  openEventSocket,
  reduceEvents,
  TERMINAL_TYPES,
  type FlEvent,
  type RunState,
  type SocketStatus,
} from "./events";

export function useRunEvents(runId: string | null): {
  state: RunState;
  socket: SocketStatus | null;
} {
  const [state, setState] = useState<RunState>(emptyRunState);
  const [socket, setSocket] = useState<SocketStatus | null>(null);
  const stateRef = useRef(state);
  stateRef.current = state;

  useEffect(() => {
    setState(emptyRunState());
    setSocket(null);
    if (!runId) return;
    const handle = openEventSocket({
      runId,
      onEvent: (event: FlEvent) => {
        setState((previous) => applyEvent(previous, event));
      },
      onStatus: setSocket,
    });
    return () => handle.close();
  }, [runId]);

  return { state, socket };
}

/** Collect a terminal run's full stream once (replay-then-close). */
export function fetchEventsOnce(runId: string, timeoutMs = 15_000): Promise<FlEvent[]> {
  return new Promise((resolve, reject) => {
    const events: FlEvent[] = [];
    const timer = setTimeout(() => {
      handle.close();
      // A run with no terminal event yet still resolves with what exists.
      resolve(events);
    }, timeoutMs);
    const handle = openEventSocket({
      runId,
      onEvent: (event) => {
        events.push(event);
        if ((TERMINAL_TYPES as readonly string[]).includes(event.type)) {
          clearTimeout(timer);
          setTimeout(() => {
            handle.close();
            resolve(events);
          }, 50);
        }
      },
      onStatus: (status) => {
        if (status.kind === "closed") {
          clearTimeout(timer);
          resolve(events);
        }
      },
      maxRetries: 1,
    });
    void reject;
  });
}

export async function fetchRunState(runId: string): Promise<RunState> {
  return reduceEvents(await fetchEventsOnce(runId));
}
