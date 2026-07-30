"""Generate the gRPC/protobuf stubs from ``fl_comm.proto`` when they are missing.

Keeping generated code out of version control means the ``.proto`` cannot drift
from the stubs. The cost is that something must run ``protoc``; doing it here
means tests, the Docker images and CI all get correct stubs without a separate
build step that someone will eventually forget to run.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROTO = _HERE / "fl_comm.proto"
_OUTPUTS = (_HERE / "fl_comm_pb2.py", _HERE / "fl_comm_pb2_grpc.py")


def _is_stale() -> bool:
    if not all(p.exists() for p in _OUTPUTS):
        return True
    proto_mtime = _PROTO.stat().st_mtime
    return any(p.stat().st_mtime < proto_mtime for p in _OUTPUTS)


def generate() -> None:
    """Run ``grpc_tools.protoc`` unconditionally."""
    if not _PROTO.exists():
        raise FileNotFoundError(f"missing schema: {_PROTO}")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            f"--proto_path={_HERE}",
            f"--python_out={_HERE}",
            f"--grpc_python_out={_HERE}",
            str(_PROTO),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "protoc failed while generating gRPC stubs.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    # grpc_tools emits `import fl_comm_pb2 as ...`, which only resolves if the
    # proto directory happens to be on sys.path. Rewrite it to a package-relative
    # import so `fl.proto` works from anywhere.
    grpc_stub = _HERE / "fl_comm_pb2_grpc.py"
    text = grpc_stub.read_text(encoding="utf-8")
    text = text.replace(
        "import fl_comm_pb2 as fl__comm__pb2",
        "from . import fl_comm_pb2 as fl__comm__pb2",
    )
    grpc_stub.write_text(text, encoding="utf-8")


def ensure_generated() -> None:
    """Generate the stubs if they are absent or older than the schema."""
    if _is_stale():
        generate()


if __name__ == "__main__":  # pragma: no cover
    generate()
    print(f"generated: {', '.join(p.name for p in _OUTPUTS)}")
