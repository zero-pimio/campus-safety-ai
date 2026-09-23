from __future__ import annotations

import argparse
import json
from pathlib import Path

from campus_safety_ai.training.continuous_manifest import audit_continuous_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit collected continuous videos and split leakage")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--strict", action="store_true", help="Require complete metadata, files and verified SHA-256")
    parser.add_argument("--output", type=Path, help="Write JSON to a new file; existing reports are never replaced")
    args = parser.parse_args()
    report = audit_continuous_manifest(args.manifest, project_root=args.project_root, strict=args.strict)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(rendered)
        except FileExistsError:
            parser.error(f"output already exists: {args.output}")
    print(rendered, end="")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
