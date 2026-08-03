/** The alpha preview must actually encode heterogeneity, deterministically. */
import { describe, expect, it } from "vitest";

import { previewPartition } from "../src/lib/dirichlet";

function concentration(histograms: number[][]): number {
  // Mean top-class share per client: 1/numClasses when uniform, →1 when
  // each client holds a single class.
  const shares = histograms.map((h) => {
    const total = h.reduce((s, x) => s + x, 0) || 1;
    return Math.max(...h) / total;
  });
  return shares.reduce((s, x) => s + x, 0) / shares.length;
}

describe("previewPartition", () => {
  it("is deterministic for a fixed seed", () => {
    const a = previewPartition({ alpha: 0.5, numClients: 8, numClasses: 10, perClass: 600 });
    const b = previewPartition({ alpha: 0.5, numClients: 8, numClasses: 10, perClass: 600 });
    expect(a).toEqual(b);
  });

  it("deals every class's examples out exhaustively", () => {
    const histograms = previewPartition({
      alpha: 0.5, numClients: 6, numClasses: 10, perClass: 600,
    });
    for (let cls = 0; cls < 10; cls += 1) {
      const dealt = histograms.reduce((s, h) => s + (h[cls] ?? 0), 0);
      expect(dealt).toBeGreaterThanOrEqual(590); // rounding slack only
      expect(dealt).toBeLessThanOrEqual(610);
    }
  });

  it("low alpha is measurably more pathological than high alpha", () => {
    const uniformish = previewPartition({
      alpha: 100, numClients: 8, numClasses: 10, perClass: 600,
    });
    const skewed = previewPartition({
      alpha: 0.05, numClients: 8, numClasses: 10, perClass: 600,
    });
    expect(concentration(skewed)).toBeGreaterThan(concentration(uniformish) + 0.2);
  });
});
