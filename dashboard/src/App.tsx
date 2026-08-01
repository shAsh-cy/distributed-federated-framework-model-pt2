/**
 * Application shell: masthead, view tabs, status line. Four views, no
 * router dependency — the view is state, the shell is an instrument fascia.
 */
import { useState } from "react";

import { ConfigureView } from "./views/Configure";
import { ConsoleView } from "./views/Console";
import { HistoryView } from "./views/History";
import { ResultsView } from "./views/Results";

const VIEWS = ["console", "configure", "history", "results"] as const;
export type ViewName = (typeof VIEWS)[number];

const LABELS: Record<ViewName, string> = {
  console: "Run console",
  configure: "Configure",
  history: "History",
  results: "Results",
};

export function App() {
  const [view, setView] = useState<ViewName>("console");
  // The run the console observes; set by Configure on start or History on open.
  const [activeRunId, setActiveRunId] = useState<string | null>(null);

  return (
    <div className="mx-auto flex min-h-screen max-w-6xl flex-col px-4">
      <header className="flex flex-wrap items-baseline justify-between gap-2 border-b-2 border-ink py-3">
        <h1 className="font-head text-xl uppercase tracking-head">
          Federated Learning Coordinator
        </h1>
        <span className="readout text-xs text-slate">
          gRPC inside · HTTP/WS outside · every number measured
        </span>
      </header>

      <nav aria-label="Views" className="flex gap-1 border-b border-rule">
        {VIEWS.map((name) => (
          <button
            key={name}
            onClick={() => setView(name)}
            aria-current={view === name ? "page" : undefined}
            className={`px-3 py-2 font-head text-sm uppercase tracking-head ${
              view === name
                ? "border-b-2 border-global text-global"
                : "text-slate hover:text-ink"
            }`}
          >
            {LABELS[name]}
          </button>
        ))}
      </nav>

      <main className="flex-1 py-4">
        {view === "console" && <ConsoleView runId={activeRunId} />}
        {view === "configure" && (
          <ConfigureView
            onStarted={(runId) => {
              setActiveRunId(runId);
              setView("console");
            }}
          />
        )}
        {view === "history" && (
          <HistoryView
            onOpenRun={(runId) => {
              setActiveRunId(runId);
              setView("console");
            }}
          />
        )}
        {view === "results" && <ResultsView />}
      </main>
    </div>
  );
}
