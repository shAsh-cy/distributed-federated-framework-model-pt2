# Gradient-inversion figures

Stage 4 of the explainer ("the nosy teacher") and the attack panel in stage 5
read `manifest.json` from this directory. Nothing in the dashboard or in
`docs/how-it-works.html` names an image file; both resolve every path through
the manifest.

While the real reconstructions are being produced on `feat/secagg-live`, the
four PNGs here are **generated placeholders** — flat graphite-on-vellum panels
that read `RECONSTRUCTION PENDING` and `PLACEHOLDER — NOT A RESULT`. They are
deliberately not plausible images. Both the dashboard and the standalone page
render an explicit "not yet measured" banner while `placeholder` is true.

## Swapping the real images in

1. Copy the PNGs into this directory.
2. Edit `manifest.json`: point `original_png` / `reconstruction_png` at them,
   set `placeholder` to `false`, and rewrite `caption` to state the condition
   that was actually run (batch size, number of local steps, optimiser).
   Correct `batch_size`, `noise_multiplier` and `epsilon` to match.
3. `python scripts/build_how_it_works.py` — regenerates the standalone page
   with the new images inlined as data URIs.
4. `cd dashboard && npm test && npm run build`.

No source file changes. `tests/test_explainer_assets.py` checks the schema and
that every path in the manifest exists.

## Manifest contract

```jsonc
{
  "schema_version": 1,
  "note": "...",
  "entries": [
    {
      "id": "undefended",          // stable key, used for React keys and ordering
      "batch_size": 1,             // examples in the batch the update came from
      "noise_multiplier": 0.0,     // z; 0 selects the undefended arm (stage 4)
      "epsilon": null,             // spent budget, or null when there is no mechanism
      "original_png": "...png",    // relative to this directory
      "reconstruction_png": "...png",
      "caption": "...",            // shown verbatim; states the condition honestly
      "placeholder": true          // false once these are real reconstructions
    }
  ]
}
```

The explainer selects the **undefended** arm by `noise_multiplier == 0` and the
**defended** arm by `noise_multiplier > 0`. Exactly one of each is required.

## Regenerating the placeholders

```
python scripts/build_inversion_placeholders.py
```

Reads `docs/_final_batch_b.json` for the calibrated noise multiplier and the
achieved epsilon, so the defended arm's condition is never retyped by hand.
