/** Browser-side MSW bootstrap for mock mode (VITE_MOCK=1) and the e2e. */
import { setupWorker } from "msw/browser";

import { handlers } from "./handlers";

export async function startBrowserMocks(): Promise<void> {
  const worker = setupWorker(...handlers);
  await worker.start({ quiet: true, onUnhandledRequest: "bypass" });
}
