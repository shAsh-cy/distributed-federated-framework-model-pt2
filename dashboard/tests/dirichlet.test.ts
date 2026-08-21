/** The alpha preview must actually encode heterogeneity, deterministically. */
import { describe, expect, it } from "vitest";

import { previewPartition, previewPartitionCoupled } from "../src/lib/dirichlet";

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

describe("previewPartitionCoupled", () => {
  const shape = { numClients: 8, numClasses: 10, perClass: 600 };

  it("deals every class exactly, with no rounding slack", () => {
    for (const alpha of [0.05, 0.5, 3, 100]) {
      const histograms = previewPartitionCoupled({ alpha, ...shape });
      for (let cls = 0; cls < shape.numClasses; cls += 1) {
        expect(histograms.reduce((s, h) => s + (h[cls] ?? 0), 0)).toBe(shape.perClass);
      }
    }
  });

  it("still encodes heterogeneity: low alpha concentrates, high alpha evens out", () => {
    const uniformish = previewPartitionCoupled({ alpha: 100, ...shape });
    const skewed = previewPartitionCoupled({ alpha: 0.05, ...shape });
    expect(concentration(skewed)).toBeGreaterThan(concentration(uniformish) + 0.2);
  });

  it("matches the Dirichlet law the uncoupled sampler draws from", () => {
    // Mean top-class share is a property of the law, not of the coupling, so
    // the two samplers must agree on it. Averaged over seeds, because one
    // 8-client realisation of a Dirichlet is a noisy thing to compare.
    for (const alpha of [0.1, 1, 10]) {
      let coupled = 0;
      let direct = 0;
      for (let seed = 1; seed <= 16; seed += 1) {
        coupled += concentration(previewPartitionCoupled({ alpha, ...shape, seed }));
        direct += concentration(previewPartition({ alpha, ...shape, seed }));
      }
      expect(Math.abs(coupled - direct) / 16).toBeLessThan(0.05);
    }
  });

  it("is continuous in alpha — this is the whole point of the coupling", () => {
    // Sweep the slider's full range and measure the largest single-step
    // rearrangement. Coupled: bars slide. Uncoupled: somewhere in the sweep
    // the RNG stream shifts and the entire partition is redealt.
    const dealt = shape.numClasses * shape.perClass;
    const step = (a: number[][], b: number[][]) => {
      let moved = 0;
      for (let c = 0; c < a.length; c += 1) {
        for (let k = 0; k < a[c]!.length; k += 1) moved += Math.abs(a[c]![k]! - b[c]![k]!);
      }
      return moved;
    };
    let worstCoupled = 0;
    let worstNaive = 0;
    let prevCoupled = previewPartitionCoupled({ alpha: 0.05, ...shape });
    let prevNaive = previewPartition({ alpha: 0.05, ...shape });
    for (let i = 1; i <= 120; i += 1) {
      const alpha = 0.05 * (10 / 0.05) ** (i / 120);
      const coupled = previewPartitionCoupled({ alpha, ...shape });
      const naive = previewPartition({ alpha, ...shape });
      worstCoupled = Math.max(worstCoupled, step(prevCoupled, coupled));
      worstNaive = Math.max(worstNaive, step(prevNaive, naive));
      prevCoupled = coupled;
      prevNaive = naive;
    }
    expect(worstCoupled / dealt).toBeLessThan(0.05);
    expect(worstNaive).toBeGreaterThan(worstCoupled * 10);
  });
});
