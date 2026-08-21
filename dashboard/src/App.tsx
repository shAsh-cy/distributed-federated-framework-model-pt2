/**
 * Application shell: masthead, view tabs, status line. Four views, no
 * router dependency — the view is state, the shell is an instrument fascia.
 *
 * One exception: /story is a real route, because a guided walkthrough is the
 * thing you send someone a link to. src/lib/route.ts is the whole router.
 */
import { useState } from "react";

import { useRoute } from "./lib/route";
import { StoryMode } from "./story/StoryMode";
import { Button } from "./ui/primitives";
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
  const [route, navigate] = useRoute();

  if (route === "story") return <StoryMode onExit={() => navigate("dashboard")} />;

  return (
    <div className="mx-auto flex min-h-screen max-w-6xl flex-col px-4">
      <header className="flex flex-wrap items-baseline justify-between gap-2 border-b-2 border-ink py-3">
        <h1 className="font-head text-xl uppercase tracking-head">
          Federated Learning Coordinator
        </h1>
        <div className="flex items-center gap-4">
          <span className="readout hidden text-xs text-slate sm:inline">
            gRPC inside · HTTP/WS outside · every number measured
          </span>
          <Button
            tone="primary"
            onClick={() => navigate("story")}
            aria-label="Open the guided walkthrough of how federated learning works"
          >
            Start here ▸ Story
          </Button>
        </div>
      </header>

      <p className="border-b border-rule py-2 font-prose text-sm text-slate">
        New to this? <button
          className="underline decoration-rule underline-offset-4 hover:text-global"
          onClick={() => navigate("story")}
        >
          Take the six-stage walkthrough
        </button>{" "}
        — a recorded run, narrated, with every number sourced.
      </p>

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
