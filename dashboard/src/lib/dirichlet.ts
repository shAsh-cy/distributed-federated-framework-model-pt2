/**
 * Client-side Dirichlet label-skew preview, mirroring the dealing scheme in
 * fl/data.py::partition_dirichlet: for every class, proportions ~ Dir(alpha)
 * over the clients, and the class's examples are dealt in those proportions.
 * Same construction, seeded PRNG, so dragging alpha shows the real effect of
 * the parameter the run will use — uniform at high alpha, pathological
 * single-class shards as alpha approaches zero.
 *
 * This is a preview of the partition SHAPE, not the byte-identical partition
 * (the training partition is drawn server-side with numpy's PCG64; this uses
 * mulberry32 + Marsaglia-Tsang gamma). The statistics match; the exact deal
 * does not, and the UI labels it a preview.
 */

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Marsaglia–Tsang gamma sampler (shape >= 0 handled via boost for < 1). */
function gamma(rand: () => number, shape: number): number {
  if (shape < 1) {
    const u = Math.max(rand(), 1e-12);
    return gamma(rand, shape + 1) * Math.pow(u, 1 / shape);
  }
  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x: number;
    let v: number;
    do {
      // Box–Muller normal
      const u1 = Math.max(rand(), 1e-12);
      const u2 = rand();
      x = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      v = 1 + c * x;
    } while (v <= 0);
    v = v * v * v;
    const u = Math.max(rand(), 1e-12);
    if (u < 1 - 0.0331 * x * x * x * x) return d * v;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}

function dirichlet(rand: () => number, alpha: number, k: number): number[] {
  const draws = Array.from({ length: k }, () => gamma(rand, alpha));
  const total = draws.reduce((s, x) => s + x, 0) || 1;
  return draws.map((x) => x / total);
}

/**
 * Per-client label histograms for `numClients` clients over `numClasses`
 * classes with `perClass` examples each, at the given alpha.
 */
export function previewPartition(opts: {
  alpha: number;
  numClients: number;
  numClasses: number;
  perClass: number;
  seed?: number;
}): number[][] {
  const { alpha, numClients, numClasses, perClass, seed = 42 } = opts;
  const rand = mulberry32(seed);
  const histograms: number[][] = Array.from({ length: numClients }, () =>
    Array.from({ length: numClasses }, () => 0),
  );
  for (let cls = 0; cls < numClasses; cls += 1) {
    const shares = dirichlet(rand, alpha, numClients);
    let dealt = 0;
    for (let client = 0; client < numClients; client += 1) {
      const count =
        client === numClients - 1
          ? perClass - dealt
          : Math.round(shares[client]! * perClass);
      histograms[client]![cls] = Math.max(0, count);
      dealt += Math.max(0, count);
    }
  }
  return histograms;
}
