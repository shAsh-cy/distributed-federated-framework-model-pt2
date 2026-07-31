"""One-time FEMNIST cache preparation.

Downloads TFF's federated EMNIST (62 classes, partitioned by writer — the
LEAF-derived preprocessing) and packs it into the flat npz cache that
``fl.data.load_femnist`` reads. Everything lands under the gitignored ``data/``
directory; nothing about this step is committed.

Usage:
    python scripts/prepare_femnist.py [--cache data/femnist/femnist62.npz]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fl.data import FEMNIST_CACHE, load_femnist, prepare_femnist_cache  # noqa: E402

LOGGER = logging.getLogger("prepare_femnist")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=str(FEMNIST_CACHE))
    parser.add_argument("--download-dir", default="data/tff_cache")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    path = prepare_femnist_cache(args.cache, args.download_dir)
    LOGGER.info("cache ready at %s", path)

    train, test, shards = load_femnist(cache_path=path)
    sizes = [s.size for s in shards]
    LOGGER.info(
        "writers=%d train=%d test=%d shard sizes min/median/max = %d/%d/%d",
        len(shards),
        len(train),
        len(test),
        min(sizes),
        int(sorted(sizes)[len(sizes) // 2]),
        max(sizes),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
