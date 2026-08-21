"""Turn a Playwright recording into the README's GIF.

Release tooling, run by hand once per release, not part of any test or build.

Two steps, because Playwright's bundled ffmpeg is a deliberately stripped
build: it has `scale` and a PNG encoder but no `fps` filter, no `palettegen`
and no GIF muxer, so it can decode and resize the webm but cannot write the
GIF. ffmpeg therefore does step one — decode and scale to numbered PNGs — and
Pillow does step two, quantising and assembling them.

    python scripts/webm_to_gif.py dashboard/test-results/record/.../video.webm \
        docs/story_mode.gif

Pillow is imported lazily and is not a project dependency; `pip install
pillow` if the second step complains. A real ffmpeg on PATH is used in
preference to Playwright's, and if it is a full build it does the whole job
itself in one pass with a proper palette.

Output geometry matches docs/dashboard_live.gif so the two sit together in the
README without one jumping.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

WIDTH = 800
FPS = 12


def playwright_ffmpeg() -> Path | None:
    """Playwright's bundled ffmpeg, wherever this platform keeps browsers."""
    roots = [
        Path.home() / "AppData" / "Local" / "ms-playwright",  # Windows
        Path.home() / "Library" / "Caches" / "ms-playwright",  # macOS
        Path.home() / ".cache" / "ms-playwright",  # Linux
    ]
    for root in roots:
        if not root.is_dir():
            continue
        for directory in sorted(root.glob("ffmpeg-*"), reverse=True):
            for binary in directory.glob("ffmpeg-*"):
                if binary.is_file():
                    return binary
    return None


def find_ffmpeg() -> tuple[Path, bool]:
    """(binary, is_full_build). A full build can write the GIF on its own."""
    system = shutil.which("ffmpeg")
    if system:
        binary = Path(system)
        filters = subprocess.run(
            [str(binary), "-hide_banner", "-filters"], capture_output=True, text=True, check=False
        ).stdout
        return binary, "palettegen" in filters
    bundled = playwright_ffmpeg()
    if bundled is None:
        raise SystemExit(
            "no ffmpeg found. Install one, or run `npx playwright install chromium` in "
            "dashboard/ to get the bundled build."
        )
    return bundled, False


def convert_with_full_ffmpeg(ffmpeg: Path, source: Path, target: Path) -> None:
    """One pass, palettegen then paletteuse: 256 colours chosen properly."""
    palette = target.with_suffix(".palette.png")
    chain = f"fps={FPS},scale={WIDTH}:-2:flags=lanczos"
    subprocess.run(
        [
            str(ffmpeg),
            "-y",
            "-i",
            str(source),
            "-vf",
            f"{chain},palettegen=stats_mode=diff",
            str(palette),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            str(ffmpeg),
            "-y",
            "-i",
            str(source),
            "-i",
            str(palette),
            "-lavfi",
            f"{chain}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
            "-loop",
            "0",
            str(target),
        ],
        check=True,
        capture_output=True,
    )
    palette.unlink(missing_ok=True)


def convert_via_frames(ffmpeg: Path, source: Path, target: Path) -> None:
    """Decode to PNGs with whatever ffmpeg we have, assemble with Pillow."""
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - a message, not a code path
        raise SystemExit(
            "this ffmpeg cannot write GIFs, so Pillow does the assembly: pip install pillow"
        ) from None

    with tempfile.TemporaryDirectory(prefix="story-gif-") as workspace:
        frames_dir = Path(workspace)
        subprocess.run(
            # -r is an output option rather than the fps FILTER, which the
            # bundled build does not have. scale it does have.
            [
                str(ffmpeg),
                "-y",
                "-i",
                str(source),
                "-r",
                str(FPS),
                "-vf",
                f"scale={WIDTH}:-2",
                str(frames_dir / "f%05d.png"),
            ],
            check=True,
            capture_output=True,
        )
        paths = sorted(frames_dir.glob("f*.png"))
        if not paths:
            raise SystemExit(f"ffmpeg produced no frames from {source}")

        # One palette for the whole animation, taken from a frame in the middle
        # of the run so it sees the ochre meter and both accuracy curves rather
        # than only the opening still.
        reference = Image.open(paths[len(paths) // 2]).convert("RGB")
        palette = reference.quantize(colors=255, method=Image.Quantize.MAXCOVERAGE)
        frames = [
            Image.open(path).convert("RGB").quantize(palette=palette, dither=Image.Dither.NONE)
            for path in paths
        ]
        frames[0].save(
            target,
            save_all=True,
            append_images=frames[1:],
            duration=round(1000 / FPS),
            loop=0,
            optimize=True,
            # disposal=1 (leave the frame in place) is what lets Pillow write
            # only the changed rectangle of each frame. disposal=2 clears to
            # background first, which forces every frame to be full size and
            # multiplied this file by ten.
            disposal=1,
        )
        print(f"{len(frames)} frames at {FPS} fps")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert a Playwright webm to a GIF.")
    parser.add_argument("source", type=Path, help="the .webm Playwright recorded")
    parser.add_argument("target", type=Path, help="the .gif to write")
    args = parser.parse_args(argv)

    if not args.source.is_file():
        raise SystemExit(f"{args.source} is not a file")

    ffmpeg, full_build = find_ffmpeg()
    print(f"ffmpeg: {ffmpeg} ({'full' if full_build else 'stripped; Pillow will assemble'})")
    if full_build:
        convert_with_full_ffmpeg(ffmpeg, args.source, args.target)
    else:
        convert_via_frames(ffmpeg, args.source, args.target)
    print(f"wrote {args.target} — {args.target.stat().st_size:,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
