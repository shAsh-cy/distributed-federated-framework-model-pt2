/**
 * docs/inversion/manifest.json, resolved for the browser.
 *
 * The manifest is the contract between this explainer and whoever produces the
 * gradient-inversion figures. Nothing here — and nothing in the stage
 * components — names an image file. Paths come from the manifest, the PNGs are
 * picked up by a glob over the same directory, and the arms are selected by
 * their noise multiplier rather than by position or id, so a manifest with the
 * arms in the other order still works.
 *
 * See docs/inversion/README.md for the swap procedure.
 */
import manifest from "../../../docs/inversion/manifest.json";

export type InversionEntry = {
  id: string;
  batch_size: number;
  noise_multiplier: number;
  epsilon: number | null;
  original_png: string;
  reconstruction_png: string;
  caption: string;
  placeholder: boolean;
};

const IMAGES = import.meta.glob<string>("../../../docs/inversion/*.png", {
  eager: true,
  query: "?url",
  import: "default",
});

const BY_FILENAME = new Map(
  Object.entries(IMAGES).map(([path, url]) => [path.slice(path.lastIndexOf("/") + 1), url]),
);

export const ENTRIES = manifest.entries as InversionEntry[];

/** Resolve a manifest-relative filename to a URL the bundler will serve. */
export function panelUrl(filename: string): string {
  const url = BY_FILENAME.get(filename);
  if (!url) {
    throw new Error(
      `docs/inversion/${filename} is named in manifest.json but is not on disk. ` +
        "Copy it in, or correct the manifest.",
    );
  }
  return url;
}

/** The attack with no defence in the way: stage 4. */
export const UNDEFENDED: InversionEntry | undefined = ENTRIES.find(
  (entry) => entry.noise_multiplier === 0,
);

/** The same attack against a clipped and noised update: stage 5. */
export const DEFENDED: InversionEntry | undefined = ENTRIES.find(
  (entry) => entry.noise_multiplier > 0,
);

/** True while the panels are generated stand-ins rather than results. */
export const AWAITING_REAL_FIGURES = ENTRIES.some((entry) => entry.placeholder);
