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

/* -- continuous-in-alpha coupling (story mode) ----------------------------- */

/**
 * The story's heterogeneity slider needs the SAME distribution as above but
 * continuous in alpha: dragging must morph a histogram, not resample it. A
 * fresh `previewPartition` at each alpha is a fresh deal, so the bars jump.
 *
 * The fix is the inverse-CDF coupling. Fix one uniform u per (class, client),
 * then set the unnormalised weight to the Gamma(alpha) quantile of u:
 *
 *     g_i = F⁻¹_Gamma(alpha)(u_i),   p_i = g_i / Σ g
 *
 * For fixed alpha the g_i are exactly independent Gamma(alpha,1) draws, so p
 * is exactly Dirichlet(alpha) — the same law `previewPartition` samples and
 * the same law fl/data.py::partition_dirichlet deals from. But now every g_i
 * is a continuous, monotone function of alpha, so the whole partition slides
 * as alpha moves and no bar ever teleports.
 *
 * Still a preview of the partition SHAPE, not the byte-identical partition.
 */

/** Regularised lower incomplete gamma P(s, x). Series below s+1, CF above. */
function lowerRegularisedGamma(s: number, x: number): number {
  if (x <= 0) return 0;
  const logGammaS = logGamma(s);
  if (x < s + 1) {
    let term = 1 / s;
    let sum = term;
    for (let n = 1; n < 300; n += 1) {
      term *= x / (s + n);
      sum += term;
      if (Math.abs(term) < Math.abs(sum) * 1e-15) break;
    }
    return sum * Math.exp(-x + s * Math.log(x) - logGammaS);
  }
  // Lentz's continued fraction for Q(s, x); P = 1 − Q.
  const tiny = 1e-300;
  let b = x + 1 - s;
  let c = 1 / tiny;
  let d = 1 / b;
  let h = d;
  for (let i = 1; i < 300; i += 1) {
    const an = -i * (i - s);
    b += 2;
    d = an * d + b;
    if (Math.abs(d) < tiny) d = tiny;
    c = b + an / c;
    if (Math.abs(c) < tiny) c = tiny;
    d = 1 / d;
    const delta = d * c;
    h *= delta;
    if (Math.abs(delta - 1) < 1e-15) break;
  }
  return 1 - Math.exp(-x + s * Math.log(x) - logGammaS) * h;
}

/** Lanczos log-gamma, g=7, n=9 — plenty for the alphas a slider can reach. */
function logGamma(z: number): number {
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ];
  if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - logGamma(1 - z);
  const x = z - 1;
  let a = c[0]!;
  const t = x + 7.5;
  for (let i = 1; i < 9; i += 1) a += c[i]! / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

/** F⁻¹_Gamma(shape,1)(p) by bracketing then bisection. Monotone in p and shape. */
function gammaQuantile(p: number, shape: number): number {
  const target = Math.min(Math.max(p, 1e-9), 1 - 1e-9);
  let hi = Math.max(shape, 1);
  while (lowerRegularisedGamma(shape, hi) < target && hi < 1e6) hi *= 2;
  let lo = 0;
  for (let i = 0; i < 60; i += 1) {
    const mid = 0.5 * (lo + hi);
    if (lowerRegularisedGamma(shape, mid) < target) lo = mid;
    else hi = mid;
  }
  return 0.5 * (lo + hi);
}

/**
 * Per-client label histograms at `alpha`, coupled across alpha so that
 * neighbouring alphas give neighbouring partitions.
 *
 * `perClass` examples of each class are dealt to `numClients` clients in
 * Dirichlet(alpha) proportions, largest-remainder rounded so the class total
 * is exact and the deal stays continuous as the proportions slide.
 */
export function previewPartitionCoupled(opts: {
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
    const weights = Array.from({ length: numClients }, () => gammaQuantile(rand(), alpha));
    const total = weights.reduce((s, w) => s + w, 0) || 1;
    const exact = weights.map((w) => (w / total) * perClass);
    const floors = exact.map(Math.floor);
    let remainder = perClass - floors.reduce((s, n) => s + n, 0);
    // Largest remainder first, so a client whose share is creeping upward
    // gains its extra example exactly when it crosses the boundary.
    const order = exact
      .map((value, index) => ({ index, frac: value - Math.floor(value) }))
      .sort((a, b) => b.frac - a.frac);
    for (const { index } of order) {
      if (remainder <= 0) break;
      floors[index] = floors[index]! + 1;
      remainder -= 1;
    }
    for (let client = 0; client < numClients; client += 1) {
      histograms[client]![cls] = floors[client]!;
    }
  }
  return histograms;
}
