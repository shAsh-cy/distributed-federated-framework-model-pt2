/**
 * Stage 4 — the nosy teacher.
 *
 * ---------------------------------------------------------------------------
 * ON THE IMAGES. This stage reads docs/inversion/manifest.json and renders
 * whatever it points at. It selects this arm by `noise_multiplier == 0` and
 * never names a filename, so swapping the real gradient-inversion figures in
 * is a FILE COPY PLUS A MANIFEST EDIT — drop the PNGs into docs/inversion/,
 * repoint `original_png` / `reconstruction_png`, set `placeholder` false, and
 * rewrite `caption` to the condition actually run. No change to this file, to
 * AttackPanel, or to anything else in the dashboard. The condition line and
 * the caption both come from the manifest precisely so that they cannot drift
 * away from the images they describe. docs/inversion/README.md has the
 * four-step procedure; tests/test_explainer_assets.py enforces the schema and
 * that every path exists.
 * ---------------------------------------------------------------------------
 *
 * This stage earns stage 5. Without it the defence is an abstraction.
 */
import { AttackPanel, PendingNotice } from "../AttackPanel";
import { AWAITING_REAL_FIGURES, UNDEFENDED } from "../inversion";
import { Caption } from "../ui";

export function Stage4Attack() {
  if (!UNDEFENDED) {
    return (
      <p className="font-prose text-base text-client">
        docs/inversion/manifest.json has no entry with noise_multiplier 0, so there is no
        undefended arm to show. Add one; nothing in this stage needs to change.
      </p>
    );
  }
  return (
    <div className="flex flex-col gap-4">
      <AttackPanel entry={UNDEFENDED} />
      <Caption>
        From nothing but the corrections, someone rebuilt a page of a student&rsquo;s notebook.
      </Caption>
      <p className="story-measure font-prose text-base">
        The teacher never saw the notebook. They only ever received the marked-up homework — the
        corrections. That turns out to be enough, under the right conditions, to work backwards to
        the page itself.
      </p>
      <p className="story-measure font-prose text-base">
        Be careful how much this proves. It is one student, one very small batch, one study step,
        and no defence in the way. It gets much harder as any of those grow. But &ldquo;the raw
        data never left the device&rdquo; is not, on its own, a promise about privacy — and that is
        why the next stage exists.
      </p>
      {AWAITING_REAL_FIGURES ? <PendingNotice /> : null}
    </div>
  );
}
