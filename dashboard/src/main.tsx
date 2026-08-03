import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { App } from "./App";
import "./styles/tokens.css";
import { ToastProvider } from "./ui/primitives";

const queryClient = new QueryClient({
  defaultOptions: { queries: { staleTime: 30_000, retry: 1 } },
});

async function boot(): Promise<void> {
  // Mock mode (VITE_MOCK=1) serves the recorded fixtures through MSW in the
  // browser: the whole application runs with zero backend on data recorded
  // from the real API. Used by the e2e test and for demos.
  if (import.meta.env.VITE_MOCK === "1") {
    const { startBrowserMocks } = await import("./mocks/browser");
    await startBrowserMocks();
  }
  createRoot(document.getElementById("root")!).render(
    <StrictMode>
      <QueryClientProvider client={queryClient}>
        <ToastProvider>
          <App />
        </ToastProvider>
      </QueryClientProvider>
    </StrictMode>,
  );
}

void boot();
