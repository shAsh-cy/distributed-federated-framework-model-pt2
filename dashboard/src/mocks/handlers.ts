/**
 * MSW handlers serving the RECORDED fixtures — responses captured from the
 * real coordinator API with the real 97-run history imported
 * (dashboard/dump_fixtures.py), plus the scripted live-demo stream
 * (dump_live_demo.py: real protocol through the real store, values drawn
 * from recorded measurements). Mock mode is recorded reality replayed, not
 * invented shapes.
 */
import { http, HttpResponse, ws } from "msw";

import algorithms from "../../fixtures/algorithms.json";
import architectures from "../../fixtures/architectures.json";
import datasets from "../../fixtures/datasets.json";
import liveDemoEvents from "../../fixtures/live_demo_scripted_events.json";
import runFemnistSeed from "../../fixtures/run_femnist_seed.json";
import runFemnistSeedEvents from "../../fixtures/run_femnist_seed_events.json";
import runGrpcDp from "../../fixtures/run_grpc_dp.json";
import runGrpcDpEvents from "../../fixtures/run_grpc_dp_events.json";
import runGrpcNodp from "../../fixtures/run_grpc_nodp.json";
import runGrpcNodpEvents from "../../fixtures/run_grpc_nodp_events.json";
import runAggregate from "../../fixtures/run_summary_aggregate.json";
import runs from "../../fixtures/runs.json";

type RunDetailFixture = (typeof runGrpcNodp) & Record<string, unknown>;

export const LIVE_DEMO_RUN_ID = "live-demo-0000";

const detailById = new Map<string, RunDetailFixture>(
  [runGrpcNodp, runGrpcDp, runFemnistSeed, runAggregate].map((d) => [d.id, d as RunDetailFixture]),
);
const eventsById = new Map<string, unknown[]>([
  [runGrpcNodp.id, runGrpcNodpEvents],
  [runGrpcDp.id, runGrpcDpEvents],
  [runFemnistSeed.id, runFemnistSeedEvents],
  [runAggregate.id, []],
]);

const liveDemoRun = {
  id: LIVE_DEMO_RUN_ID,
  created_at: 0,
  status: "running",
  source: "live",
  label: "live demo (scripted from recorded measurements)",
  group_key: null,
  is_aggregate: false,
  seed: 42,
};

export const eventsLink = ws.link("ws://*/api/runs/:runId/events");

/** Interval between streamed live-demo events; e2e keeps this small. */
export let liveDemoIntervalMs = 120;
export function setLiveDemoInterval(ms: number): void {
  liveDemoIntervalMs = ms;
}

export const handlers = [
  http.get("*/api/datasets", () => HttpResponse.json(datasets)),
  http.get("*/api/algorithms", () => HttpResponse.json(algorithms)),
  http.get("*/api/architectures", () => HttpResponse.json(architectures)),
  http.get("*/api/runs", () => HttpResponse.json([liveDemoRun, ...runs])),
  http.get("*/api/runs/:runId", ({ params }) => {
    const id = String(params.runId);
    if (id === LIVE_DEMO_RUN_ID)
      return HttpResponse.json({
        ...liveDemoRun,
        config: { data: { num_clients: 10 } },
        final_metrics: null,
        num_events: liveDemoEvents.length,
      });
    const detail = detailById.get(id);
    if (!detail)
      return HttpResponse.json({ detail: `unknown run ${id}` }, { status: 404 });
    return HttpResponse.json(detail);
  }),
  http.post("*/api/runs", () =>
    HttpResponse.json({ run_id: LIVE_DEMO_RUN_ID }, { status: 201 }),
  ),
  http.post("*/api/runs/:runId/stop", ({ params }) =>
    HttpResponse.json({
      run_id: String(params.runId),
      stopping: false,
      detail: "mock mode: runs are recorded streams; nothing to stop",
    }),
  ),

  eventsLink.addEventListener("connection", ({ client, params }) => {
    const id = String(params.runId);
    const url = new URL(client.url);
    const since = Number(url.searchParams.get("since") ?? "0");

    const finished: unknown[] | undefined =
      id === LIVE_DEMO_RUN_ID ? undefined : eventsById.get(id);

    if (finished !== undefined) {
      // Terminal run: replay from `since`, then close — the server's contract.
      for (const event of finished) {
        if ((event as { seq: number }).seq >= since) client.send(JSON.stringify(event));
      }
      client.close(1000, "run already terminal");
      return;
    }

    if (id !== LIVE_DEMO_RUN_ID) {
      client.close(4404, `unknown run ${id}`);
      return;
    }

    // Live demo: stream the scripted fixture with real pacing.
    let index = liveDemoEvents.findIndex((e) => (e as { seq: number }).seq >= since);
    if (index < 0) index = liveDemoEvents.length;
    const timer = setInterval(() => {
      if (index >= liveDemoEvents.length) {
        clearInterval(timer);
        client.close(1000, "run finished");
        return;
      }
      client.send(JSON.stringify(liveDemoEvents[index]));
      index += 1;
    }, liveDemoIntervalMs);
    client.addEventListener("close", () => clearInterval(timer));
  }),
];
