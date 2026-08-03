/**
 * Views over MSW serving the recorded fixtures: history honesty markers,
 * results bands, configure options from capabilities, alpha preview
 * reactivity, and the console replaying a real recorded stream over the
 * mocked WebSocket (ordering asserted by the round counter and log).
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { type ReactElement } from "react";
import { describe, expect, it } from "vitest";

import runGrpcDp from "../fixtures/run_grpc_dp.json";
import runsFixture from "../fixtures/runs.json";
import { ConfigureView } from "../src/views/Configure";
import { ConsoleView } from "../src/views/Console";
import { HistoryView } from "../src/views/History";

function renderWithQuery(ui: ReactElement) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false, staleTime: 0 } },
  });
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
}

describe("HistoryView", () => {
  it("lists the imported history with honesty markers", async () => {
    renderWithQuery(<HistoryView onOpenRun={() => {}} />);
    // 97 imported + the mock live-demo row.
    await waitFor(() => {
      expect(screen.getByText(`${runsFixture.length + 1} runs`)).toBeInTheDocument();
    });
    expect(screen.getAllByTitle(/Imported from committed results/).length).toBeGreaterThan(50);
    expect(
      screen.getAllByTitle(/Multi-seed summary carrying mean and range/).length,
    ).toBeGreaterThanOrEqual(14);
  });
});

describe("ConfigureView", () => {
  it("offers datasets and algorithms from the API, never hardcoded", async () => {
    renderWithQuery(<ConfigureView onStarted={() => {}} />);
    const datasetSelect = await screen.findByLabelText("Dataset");
    const options = within(datasetSelect).getAllByRole("option");
    expect(options.map((o) => o.textContent)).toEqual([
      expect.stringContaining("fashion_mnist"),
      expect.stringContaining("femnist"),
    ]);
    expect(screen.getByLabelText("Algorithm")).toBeInTheDocument();
    // Parameter count surfaced from /architectures:
    expect(screen.getByText(/225,034|225 034/)).toBeInTheDocument();
  });

  it("alpha slider reshapes the preview histograms live", async () => {
    renderWithQuery(<ConfigureView onStarted={() => {}} />);
    const slider = await screen.findByLabelText("Dirichlet alpha");
    const before = screen
      .getAllByRole("img", { name: /Label histogram/ })
      .map((el) => el.getAttribute("aria-label"))
      .join("|");
    // Drag to the pathological end.
    fireEvent.change(slider, { target: { value: "0.05" } });
    await waitFor(() => {
      const after = screen
        .getAllByRole("img", { name: /Label histogram/ })
        .map((el) => el.getAttribute("aria-label"))
        .join("|");
      expect(after).not.toEqual(before);
    });
  });
});

describe("ConsoleView over the mocked WebSocket", () => {
  it("replays a recorded DP run: curve, epsilon, ordered log, closed stream", async () => {
    renderWithQuery(<ConsoleView runId={runGrpcDp.id} />);
    // The recorded run has 20 rounds; replay ends in run_completed.
    await waitFor(
      () => {
        expect(screen.getByText(/run completed · 20 rounds/)).toBeInTheDocument();
      },
      { timeout: 10_000 },
    );
    // Round counter shows the final round, monospace-padded.
    expect(screen.getByText("020")).toBeInTheDocument();
    // The budget meter is present for a DP run and reads the recorded epsilon
    // (it also appears in log lines, so assert at-least-one).
    expect(screen.getAllByText(/6\.228/).length).toBeGreaterThanOrEqual(1);
    // Stream closed explicitly, never a silent freeze.
    await waitFor(() => {
      expect(screen.getByRole("status")).toHaveTextContent(/stream closed/);
    });
    // Log ordering: seq numbers strictly ascending.
    const items = screen.getAllByRole("listitem");
    const seqs = items
      .map((li) => Number(li.textContent?.slice(0, 4)))
      .filter((n) => !Number.isNaN(n));
    expect(seqs).toEqual([...seqs].sort((a, b) => a - b));
  }, 15_000);
});
