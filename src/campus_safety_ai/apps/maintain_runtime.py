"""Inspect one runtime and quarantine expired delivered evidence with --apply."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from campus_safety_ai.core.retention import maintain_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True,
                        help="dedicated output directory for this outbox only")
    parser.add_argument("--outbox", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--retention-days", type=float, default=30)
    parser.add_argument("--apply", action="store_true",
                        help="stop producers first; move eligible evidence into recoverable quarantine")
    args = parser.parse_args()
    try:
        report = maintain_runtime(runtime_root=args.runtime_root, database=args.outbox,
                                  evidence_root=args.evidence_dir,
                                  retention_seconds=args.retention_days * 86400, apply=args.apply)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"maintenance refused: {error}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
