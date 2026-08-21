/**
 * A pair of panels — the page from the notebook, and what the attack rebuilt
 * from the corrections alone — drawn entirely from docs/inversion/manifest.json.
 *
 * Nothing here names a file, an aspect ratio or a condition. The manifest
 * supplies the paths, the batch size, the noise multiplier, the epsilon and the
 * caption; this renders them. See docs/inversion/README.md.
 */
import { panelUrl, type InversionEntry } from "./inversion";

function Panel({ title, file, alt }: { title: string; file: string; alt: string }) {
  return (
    <figure className="flex min-w-0 flex-col gap-1">
      <figcaption className="font-head text-xs uppercase tracking-head text-slate">
        {title}
      </figcaption>
      <img
        src={panelUrl(file)}
        alt={alt}
        className="w-full max-w-[240px] border border-rule bg-ground-raised"
      />
    </figure>
  );
}

export function AttackPanel({
  entry,
  originalTitle = "The page",
  reconstructionTitle = "What the attack rebuilt",
}: {
  entry: InversionEntry;
  originalTitle?: string;
  reconstructionTitle?: string;
}) {
  // Three decimals on both, because a manifest carries full float precision
  // and a caption should not.
  const condition = [
    `batch ${entry.batch_size}`,
    `noise ×${entry.noise_multiplier.toFixed(3)}`,
    entry.epsilon === null ? "no privacy budget spent" : `ε ${entry.epsilon.toFixed(3)}`,
  ].join(" · ");

  return (
    <div className="flex flex-col gap-2" data-attack-arm={entry.id}>
      <div className="flex flex-wrap gap-4">
        <Panel title={originalTitle} file={entry.original_png} alt={`Original, ${condition}`} />
        <Panel
          title={reconstructionTitle}
          file={entry.reconstruction_png}
          alt={`Reconstruction, ${condition}`}
        />
      </div>
      <p className="readout text-xs text-slate" data-attack-condition>
        {condition}
      </p>
      <p className="story-measure font-prose text-base">{entry.caption}</p>
    </div>
  );
}

/** Shown while the manifest still points at generated stand-ins. */
export function PendingNotice() {
  return (
    <p
      className="border border-client px-3 py-2 font-prose text-sm text-client"
      data-inversion-pending
    >
      These two panels are placeholders. The reconstructions are being produced on another branch;
      the panels above say so rather than showing you something that looks like a result. Every
      other number and picture in this walkthrough is measured.
    </p>
  );
}
