"""The explainer's generated assets: schema, existence, and drift.

Three things this file is responsible for.

1. docs/inversion/manifest.json is a contract, not a convenience. The
   dashboard's stage 4 and stage 5 resolve every image path through it and
   select their arm by noise multiplier, so the real reconstructions can be
   dropped in as a file copy plus a manifest edit. That only holds if the
   schema holds and every path exists — hence these tests, which will start
   failing the moment someone points the manifest at a file they forgot to
   copy.

2. dashboard/fixtures/story_figures.json and docs/how-it-works.html are
   generated from committed results. Both are committed too, because the
   dashboard imports one and the other has to open on a double-click with no
   build step. That means both can drift, so both are regenerated here and
   compared. Images are compared by raster rather than by compressed bytes:
   deflate output is a property of the zlib build, not of the artefact, and a
   guard that fires on the runner's zlib is a guard nobody keeps.

3. docs/how-it-works.html must fetch nothing. It is going to be emailed and
   opened from file://; one stray CDN reference and it renders wrong on the
   machine of the person you most wanted to show it to.

No TensorFlow, no training, no network.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import struct
import sys
import zlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

MANIFEST_PATH = ROOT / "docs" / "inversion" / "manifest.json"
FIGURES_PATH = ROOT / "dashboard" / "fixtures" / "story_figures.json"
PAGE_PATH = ROOT / "docs" / "how-it-works.html"

REQUIRED_ENTRY_FIELDS: dict[str, tuple[type, ...]] = {
    "batch_size": (int,),
    "noise_multiplier": (int, float),
    "epsilon": (int, float, type(None)),
    "original_png": (str,),
    "reconstruction_png": (str,),
    "caption": (str,),
}
# Extensions to the declared contract, both load-bearing: `id` gives the arms
# stable keys, `placeholder` is how the swap flips the "not a result" banner
# off without a code change.
OPTIONAL_ENTRY_FIELDS: dict[str, tuple[type, ...]] = {
    "id": (str,),
    "placeholder": (bool,),
}

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def png_raster(blob: bytes) -> tuple[bytes, bytes]:
    """(IHDR, decompressed pixel data) for a PNG, using stdlib only.

    Compressed bytes are the wrong thing to compare between machines: deflate
    output depends on the zlib build, and CI's is not this laptop's. The raster
    is the thing that is actually supposed to be identical.
    """
    assert blob[:8] == PNG_MAGIC, "not a PNG"
    header, pixels, offset = b"", b"", 8
    while offset < len(blob):
        (length,) = struct.unpack(">I", blob[offset : offset + 4])
        tag = blob[offset + 4 : offset + 8]
        payload = blob[offset + 8 : offset + 8 + length]
        if tag == b"IHDR":
            header = payload
        elif tag == b"IDAT":
            pixels += payload
        offset += 12 + length
    return header, zlib.decompress(pixels)


def raster_digest(blob: bytes) -> str:
    header, pixels = png_raster(blob)
    return hashlib.sha256(header + pixels).hexdigest()


DATA_URI = re.compile(rb"data:image/png;base64,([A-Za-z0-9+/=]+)")


def with_images_by_content(page: bytes) -> bytes:
    """The page, with each inlined PNG replaced by a digest of its raster.

    docs/how-it-works.html embeds its images as base64, so comparing the page
    byte for byte also compares deflate output — which, as above, is a property
    of the zlib build rather than of the page. Normalising the payloads keeps
    the comparison on the thing that matters: the page's text is current and its
    pictures are the right pictures. That the digests are the MANIFEST's
    pictures is asserted separately below.
    """
    return DATA_URI.sub(
        lambda match: (
            b"data:image/png;base64,<raster "
            + raster_digest(base64.b64decode(match.group(1))).encode()
            + b">"
        ),
        page,
    )


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


class TestInversionManifest:
    def test_has_a_schema_version_and_entries(self, manifest):
        assert manifest["schema_version"] == 1
        assert isinstance(manifest["entries"], list)
        assert manifest["entries"], "the explainer needs at least the two arms"

    def test_every_entry_carries_the_declared_fields_with_the_declared_types(self, manifest):
        for entry in manifest["entries"]:
            for field, types in REQUIRED_ENTRY_FIELDS.items():
                assert field in entry, f"{entry.get('id', entry)} is missing {field}"
                assert isinstance(entry[field], types), (
                    f"{entry.get('id')}.{field} is {type(entry[field]).__name__}, "
                    f"expected one of {[t.__name__ for t in types]}"
                )
            unknown = set(entry) - set(REQUIRED_ENTRY_FIELDS) - set(OPTIONAL_ENTRY_FIELDS)
            assert not unknown, f"unrecognised manifest fields: {sorted(unknown)}"
            for field, types in OPTIONAL_ENTRY_FIELDS.items():
                if field in entry:
                    assert isinstance(entry[field], types)

    def test_captions_say_something(self, manifest):
        for entry in manifest["entries"]:
            assert len(entry["caption"].split()) >= 8, (
                f"{entry.get('id')} needs a caption stating the condition, not a label"
            )

    def test_exactly_one_undefended_and_one_defended_arm(self, manifest):
        """The dashboard selects by noise multiplier, so the selection must be
        unambiguous. Two entries at z=0 and stage 4 would silently pick one."""
        undefended = [e for e in manifest["entries"] if e["noise_multiplier"] == 0]
        defended = [e for e in manifest["entries"] if e["noise_multiplier"] > 0]
        assert len(undefended) == 1, "stage 4 needs exactly one arm with noise_multiplier 0"
        assert len(defended) == 1, "stage 5 needs exactly one arm with noise_multiplier > 0"

    def test_an_arm_with_noise_carries_the_budget_it_spent(self, manifest):
        for entry in manifest["entries"]:
            if entry["noise_multiplier"] > 0:
                assert entry["epsilon"] is not None, (
                    f"{entry.get('id')} adds noise, so it spent a budget; say what it was"
                )

    def test_every_referenced_image_exists_and_is_a_png(self, manifest):
        for entry in manifest["entries"]:
            for field in ("original_png", "reconstruction_png"):
                name = entry[field]
                assert "/" not in name and "\\" not in name, (
                    f"{field} must be a bare filename beside the manifest, got {name!r}"
                )
                path = MANIFEST_PATH.parent / name
                assert path.is_file(), f"manifest names {name}, which is not in docs/inversion/"
                assert path.read_bytes()[:8] == PNG_MAGIC, f"{name} is not a PNG"

    def test_no_orphan_pngs(self, manifest):
        """A leftover image means somebody half-finished a swap."""
        referenced = {
            entry[field]
            for entry in manifest["entries"]
            for field in ("original_png", "reconstruction_png")
        }
        on_disk = {p.name for p in MANIFEST_PATH.parent.glob("*.png")}
        assert on_disk == referenced, (
            f"docs/inversion/ has images the manifest does not name: {sorted(on_disk - referenced)}"
        )

    def test_placeholders_regenerate_identically(self, manifest):
        """While the panels are stand-ins they must be exactly what the
        generator produces, so nobody hand-draws something that looks more
        convincing than a placeholder should."""
        if not any(entry.get("placeholder") for entry in manifest["entries"]):
            pytest.skip("the real reconstructions have landed; nothing here is generated")
        import build_inversion_placeholders as generator

        def snapshot() -> dict[str, object]:
            state: dict[str, object] = {
                path.name: png_raster(path.read_bytes())
                for path in sorted(MANIFEST_PATH.parent.glob("*.png"))
            }
            state[MANIFEST_PATH.name] = MANIFEST_PATH.read_bytes()
            return state

        before = snapshot()
        generator.main()
        after = snapshot()
        assert after == before, (
            "docs/inversion/ has been hand-edited while still marked placeholder. "
            "Either run scripts/build_inversion_placeholders.py, or set placeholder "
            "false because these are now real."
        )


class TestStoryFigures:
    def test_regenerates_identically(self):
        """The committed figures file must be exactly what the generator makes
        from the committed results — otherwise a number on screen could have
        been edited into the fixture by hand."""
        import build_story_figures as generator

        before = FIGURES_PATH.read_bytes()
        generator.main()
        after = FIGURES_PATH.read_bytes()
        assert before == after, (
            "dashboard/fixtures/story_figures.json is stale. "
            "Run: python scripts/build_story_figures.py"
        )

    def test_every_figure_carries_a_resolvable_pointer(self):
        import build_story_figures as generator

        data = json.loads(FIGURES_PATH.read_text(encoding="utf-8"))
        for name, entry in data["figures"].items():
            source = entry["source"]
            document = json.loads((ROOT / source["file"]).read_text(encoding="utf-8"))
            if "series" in source:
                continue  # derived scalars are checked against their series below
            assert generator.resolve(document, source["pointer"]) == entry["value"], (
                f"{name} says it came from {source['file']}{source['pointer']}, but that "
                "pointer resolves to something else"
            )

    def test_derived_scalars_match_the_series_they_came_from(self):
        data = json.loads(FIGURES_PATH.read_text(encoding="utf-8"))
        for name, entry in data["figures"].items():
            source = entry["source"]
            if "series" not in source:
                continue
            points = data["series"][source["series"]]["points"]
            expected = (
                points[-1] if source["reduce_series"] == "last" else sum(points) / len(points)
            )
            assert entry["value"] == pytest.approx(expected), f"{name} does not match its series"

    def test_the_two_accuracy_curves_end_where_the_summaries_say(self):
        """The curves and the headline figures are read from different places;
        if they ever disagree, the story is telling two stories."""
        data = json.loads(FIGURES_PATH.read_text(encoding="utf-8"))
        assert data["series"]["dpCurve"]["points"][-1] == pytest.approx(
            data["figures"]["dpFinal"]["value"]
        )
        assert data["series"]["nodpCurve"]["points"][-1] == pytest.approx(
            data["figures"]["nodpFinal"]["value"]
        )

    def test_the_quoted_privacy_cost_is_the_difference_of_the_two_arms(self):
        data = json.loads(FIGURES_PATH.read_text(encoding="utf-8"))
        gap = data["figures"]["nodpFinal"]["value"] - data["figures"]["dpFinal"]["value"]
        assert data["figures"]["dpCost"]["value"] == pytest.approx(gap, abs=1e-9)

    def test_the_two_arms_were_run_at_the_same_cohort_and_rounds(self):
        """68.2 % against 72.8 % is only a privacy cost if nothing else moved."""
        batch_b = json.loads((ROOT / "docs" / "_final_batch_b.json").read_text(encoding="utf-8"))
        budget_e = json.loads(
            (ROOT / "docs" / "_femnist_budget_e.json").read_text(encoding="utf-8")
        )
        assert batch_b["m"] == budget_e["budget"]["m"]
        assert batch_b["rounds"] == budget_e["budget"]["rounds"]
        assert batch_b["writers"] == budget_e["budget"]["writers"]
        control_cell = budget_e["budget"]["cells"][2]
        assert control_cell["local_epochs"] == batch_b["local_epochs"]
        assert all(not run["dp"] for run in control_cell["runs"]), "the control must be no-DP"
        assert all(run["dp"] for run in batch_b["runs"]), "the DP arm must be DP"


class TestStandalonePage:
    def test_regenerates_identically(self):
        import build_how_it_works as generator

        before = with_images_by_content(PAGE_PATH.read_bytes())
        generator.main()
        after = with_images_by_content(PAGE_PATH.read_bytes())
        if after != before:
            at = next(
                (i for i, (a, b) in enumerate(zip(after, before, strict=False)) if a != b),
                min(len(after), len(before)),
            )
            window = slice(max(0, at - 90), at + 90)
            raise AssertionError(
                "docs/how-it-works.html is stale, or this machine generates it differently.\n"
                f"first difference at byte {at} of {len(before)}\n"
                f"committed: ...{before[window]!r}...\n"
                f"generated: ...{after[window]!r}...\n"
                "Run: python scripts/build_how_it_works.py"
            )

    def test_fetches_nothing(self):
        """Every src, every stylesheet, every font: inline or a data URI. The
        only remote URLs allowed are anchor hrefs, which are clicks."""
        page = PAGE_PATH.read_text(encoding="utf-8")
        for src in re.findall(r'\ssrc\s*=\s*"([^"]*)"', page):
            assert src.startswith("data:"), f"src fetches {src!r}"
        assert "@import" not in page, "an @import is a network request"
        assert "url(" not in page, "a CSS url() is a network request"
        assert "<link" not in page, "a <link> is a network request"
        for href in re.findall(r'\shref\s*=\s*"([^"]*)"', page):
            assert href.startswith("#") or href.startswith("https://"), f"unexpected href {href!r}"

    def test_is_one_file(self):
        page = PAGE_PATH.read_text(encoding="utf-8")
        assert page.startswith("<!doctype html>")
        assert "<style>" in page and "<script>" in page
        assert PAGE_PATH.stat().st_size < 4_000_000, "too big to attach to an email"

    def test_shows_the_same_numbers_the_dashboard_does(self):
        page = PAGE_PATH.read_text(encoding="utf-8")
        figures = json.loads(FIGURES_PATH.read_text(encoding="utf-8"))["figures"]
        for name in ("nodpFinal", "dpFinal", "dpCost", "epsilon", "pooled", "longRoundsAcc"):
            display = figures[name]["display"]
            assert display in page, f"the page never shows {name} ({display})"

    def test_the_inlined_images_are_the_manifest_images(self):
        """The drift test above compares the page's pictures by raster digest,
        which proves they have not changed. This is what proves they are the
        right ones — the four the manifest names, and nothing else."""
        page = PAGE_PATH.read_bytes()
        embedded = {raster_digest(base64.b64decode(match)) for match in DATA_URI.findall(page)}
        expected = {
            raster_digest((MANIFEST_PATH.parent / name).read_bytes())
            for name in sorted(
                {
                    entry[field]
                    for entry in json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["entries"]
                    for field in ("original_png", "reconstruction_png")
                }
            )
        }
        assert embedded == expected, (
            "docs/how-it-works.html has images that are not the manifest's. "
            "Run: python scripts/build_how_it_works.py"
        )

    def test_the_notebook_sampler_is_the_same_on_every_interpreter(self):
        """The three heterogeneity pictures are floating point and the page is
        compared byte for byte, so the sampler behind them has to give the same
        answer on every interpreter. random.gammavariate does not — its
        implementation has changed between CPython versions, which silently
        redeals these three pictures and fails the drift test above with no
        clue why. build_how_it_works therefore rolls its own on top of Mersenne
        Twister, and this pins it.
        """
        import random

        import build_how_it_works as generator

        rng = random.Random(42)
        draws = [generator.gamma_draw(rng, 0.5) for _ in range(4)]
        assert draws == pytest.approx(
            [0.312735047405, 1.128012610588, 0.324311843026, 0.000251132746], abs=1e-11
        )
        assert generator.deal(0.5)[0] == [24, 123, 6, 131, 1, 151, 144, 7, 9, 49]

    def test_the_engineers_appendix_points_at_documents_that_exist(self):
        import build_how_it_works as generator

        for _analogy, _mechanism, _detail, docs in generator.ANALOGIES:
            for relative, _what in docs:
                assert (ROOT / relative).exists(), f"the appendix links {relative}, which is gone"

    def test_the_appendix_covers_every_analogy_the_story_uses(self):
        import build_how_it_works as generator

        named = {analogy.lower() for analogy, *_ in generator.ANALOGIES}
        for required in ("trimming", "ink", "the ink budget", "the merge", "sealed envelopes"):
            assert required in named, f"the appendix does not map {required!r}"

    def test_says_so_while_the_reconstructions_are_placeholders(self, manifest):
        page = PAGE_PATH.read_text(encoding="utf-8")
        if any(entry.get("placeholder") for entry in manifest["entries"]):
            assert "placeholders" in page, (
                "the manifest is still on stand-ins and the page does not admit it"
            )
