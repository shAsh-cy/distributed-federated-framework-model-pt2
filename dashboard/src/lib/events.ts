/**
 * Event schema (mirroring coordinator/events.py, SCHEMA_VERSION 1), the
 * replay reducer, and the reconnecting WebSocket client.
 *
 * The replay guarantee is the design: full run state derives from the event
 * stream alone. `reduceEvents` IS that derivation — the console renders its
 * output whether events arrived live or from `?since=0` replay, so a page
 * refresh mid-run reconstructs exactly what a continuously-connected client
 * would show.
 */

export const SCHEMA_VERSION = 1;

export type ClientInfo = {
  client_id: string;
  num_examples: number;
  label_histogram: number[];
};

type Base = { schema_version: number; run_id: string; seq: number; ts: number };

export type RunStarted = Base & {
  type: "run_started";
  config: Record<string, unknown>;
  num_classes: number;
  clients: ClientInfo[];
};
export type RoundStarted = Base & { type: "round_started"; round: number; model_version: number };
export type ClientSampled = Base & {
  type: "client_sampled";
  round: number;
  client_id: string;
  framework: string | null;
};
export type ClientReported = Base & {
  type: "client_reported";
  round: number;
  client_id: string;
  num_examples: number;
  local_accuracy: number | null;
  local_loss: number | null;
  wall_clock_seconds: number | null;
  bytes: number | null;
};
export type ClientDropped = Base & {
  type: "client_dropped";
  round: number;
  client_id: string;
  reason: "deadline" | "disconnect" | "stale_version" | "invalid_payload" | "stopped";
};
export type RoundAggregated = Base & {
  type: "round_aggregated";
  round: number;
  model_version: number;
  global_accuracy: number | null;
  global_loss: number | null;
  bytes_sent: number | null;
  bytes_received: number | null;
  cumulative_epsilon: number | null;
  median_update_norm: number | null;
  clipped_fraction: number | null;
};
export type RunCompleted = Base & {
  type: "run_completed";
  final_accuracy: number | null;
  final_loss: number | null;
  rounds_completed: number;
  stopped_early: boolean;
};
export type RunFailed = Base & { type: "run_failed"; error: string; rounds_completed: number };

export type FlEvent =
  | RunStarted
  | RoundStarted
  | ClientSampled
  | ClientReported
  | ClientDropped
  | RoundAggregated
  | RunCompleted
  | RunFailed;

export const TERMINAL_TYPES = ["run_completed", "run_failed"] as const;

/* -- replay reducer -------------------------------------------------------- */

export type ClientState = {
  info: ClientInfo;
  /** slate → sampled → reported | dropped, per the current round */
  phase: "idle" | "sampled" | "reported" | "dropped";
  framework: string | null;
  lastLocalAccuracy: number | null;
  lastWallClock: number | null;
  lastBytes: number | null;
  /** dropped clients stay dimmed one round; this tracks the round they missed */
  droppedInRound: number | null;
  arrivalOrder: number | null;
};

export type RoundPoint = {
  round: number;
  accuracy: number | null;
  loss: number | null;
  bytesSent: number | null;
  bytesReceived: number | null;
  epsilon: number | null;
  medianUpdateNorm: number | null;
  clippedFraction: number | null;
};

export type RunState = {
  runId: string | null;
  config: Record<string, unknown> | null;
  numClasses: number;
  clients: Map<string, ClientState>;
  clientOrder: string[];
  currentRound: number;
  modelVersion: number;
  curve: RoundPoint[];
  log: FlEvent[];
  status: "waiting" | "running" | "completed" | "failed";
  finalAccuracy: number | null;
  stoppedEarly: boolean;
  error: string | null;
  lastSeq: number;
};

export function emptyRunState(): RunState {
  return {
    runId: null,
    config: null,
    numClasses: 0,
    clients: new Map(),
    clientOrder: [],
    currentRound: 0,
    modelVersion: 0,
    curve: [],
    log: [],
    status: "waiting",
    finalAccuracy: null,
    stoppedEarly: false,
    error: null,
    lastSeq: -1,
  };
}

export function applyEvent(state: RunState, event: FlEvent): RunState {
  if (event.seq <= state.lastSeq) return state; // replay overlap: idempotent
  const next: RunState = {
    ...state,
    clients: new Map(state.clients),
    curve: state.curve,
    log: [...state.log, event],
    lastSeq: event.seq,
  };
  switch (event.type) {
    case "run_started": {
      next.runId = event.run_id;
      next.config = event.config;
      next.numClasses = event.num_classes;
      next.status = "running";
      next.clientOrder = event.clients.map((c) => c.client_id);
      for (const info of event.clients) {
        next.clients.set(info.client_id, {
          info,
          phase: "idle",
          framework: null,
          lastLocalAccuracy: null,
          lastWallClock: null,
          lastBytes: null,
          droppedInRound: null,
          arrivalOrder: null,
        });
      }
      break;
    }
    case "round_started": {
      next.currentRound = event.round;
      next.modelVersion = event.model_version;
      for (const [id, c] of next.clients) {
        // A client that missed the previous round stays dimmed through this one.
        const stillDimmed = c.droppedInRound !== null && c.droppedInRound === event.round - 1;
        next.clients.set(id, {
          ...c,
          phase: stillDimmed ? "dropped" : "idle",
          arrivalOrder: null,
        });
      }
      break;
    }
    case "client_sampled": {
      const c = next.clients.get(event.client_id);
      if (c)
        next.clients.set(event.client_id, {
          ...c,
          phase: "sampled",
          framework: event.framework ?? c.framework,
        });
      break;
    }
    case "client_reported": {
      const c = next.clients.get(event.client_id);
      if (c) {
        const order = [...next.clients.values()].filter((x) => x.phase === "reported").length;
        next.clients.set(event.client_id, {
          ...c,
          phase: "reported",
          lastLocalAccuracy: event.local_accuracy,
          lastWallClock: event.wall_clock_seconds,
          lastBytes: event.bytes,
          droppedInRound: null,
          arrivalOrder: order,
        });
      }
      break;
    }
    case "client_dropped": {
      const c = next.clients.get(event.client_id);
      if (c)
        next.clients.set(event.client_id, {
          ...c,
          phase: "dropped",
          droppedInRound: event.round,
        });
      break;
    }
    case "round_aggregated": {
      next.modelVersion = event.model_version;
      next.curve = [
        ...state.curve,
        {
          round: event.round,
          accuracy: event.global_accuracy,
          loss: event.global_loss,
          bytesSent: event.bytes_sent,
          bytesReceived: event.bytes_received,
          epsilon: event.cumulative_epsilon,
          medianUpdateNorm: event.median_update_norm,
          clippedFraction: event.clipped_fraction,
        },
      ];
      break;
    }
    case "run_completed": {
      next.status = "completed";
      next.finalAccuracy = event.final_accuracy;
      next.stoppedEarly = event.stopped_early;
      break;
    }
    case "run_failed": {
      next.status = "failed";
      next.error = event.error;
      break;
    }
  }
  return next;
}

export function reduceEvents(events: FlEvent[]): RunState {
  return events.reduce(applyEvent, emptyRunState());
}

/* -- reconnecting WebSocket client ----------------------------------------- */

export type SocketStatus =
  | { kind: "connecting"; attempt: number }
  | { kind: "open" }
  | { kind: "reconnecting"; attempt: number }
  | { kind: "closed"; reason: string };

export type EventSocket = { close: () => void };

/**
 * Connects to /runs/{id}/events, resumes from the last seen sequence number
 * on every reconnect (the server replays; we lose nothing), and reports
 * connection state so the UI can show an explicit reconnecting indicator
 * with a retry count — never a silent freeze.
 */
export function openEventSocket(opts: {
  runId: string;
  since?: number;
  onEvent: (event: FlEvent) => void;
  onStatus: (status: SocketStatus) => void;
  makeSocket?: (url: string) => WebSocket;
  maxRetries?: number;
}): EventSocket {
  const { runId, onEvent, onStatus, makeSocket, maxRetries = 8 } = opts;
  let lastSeq = (opts.since ?? 0) - 1;
  let attempt = 0;
  let closedByUser = false;
  let socket: WebSocket | null = null;
  let timer: ReturnType<typeof setTimeout> | null = null;

  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const url = () =>
    `${protocol}://${window.location.host}/api/runs/${runId}/events?since=${lastSeq + 1}`;

  function connect() {
    onStatus(attempt === 0 ? { kind: "connecting", attempt } : { kind: "reconnecting", attempt });
    socket = makeSocket ? makeSocket(url()) : new WebSocket(url());
    socket.onopen = () => {
      attempt = 0;
      onStatus({ kind: "open" });
    };
    socket.onmessage = (message: MessageEvent<string>) => {
      const event = JSON.parse(message.data) as FlEvent;
      if (event.seq > lastSeq) {
        lastSeq = event.seq;
        onEvent(event);
      }
    };
    socket.onclose = (close: CloseEvent) => {
      if (closedByUser) return;
      if (close.code === 1000) {
        onStatus({ kind: "closed", reason: close.reason || "stream ended" });
        return;
      }
      attempt += 1;
      if (attempt > maxRetries) {
        onStatus({ kind: "closed", reason: `gave up after ${maxRetries} retries` });
        return;
      }
      timer = setTimeout(connect, Math.min(8000, 250 * 2 ** attempt));
    };
  }

  connect();
  return {
    close: () => {
      closedByUser = true;
      if (timer) clearTimeout(timer);
      socket?.close();
    },
  };
}
