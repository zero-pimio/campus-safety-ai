"""Check ONNX parity on validation videos and time CPU inference on one sampled clip.

Run from the repository root with PYTHONPATH=src. This verifies the PC export,
not live-camera throughput or RKNN compatibility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from campus_safety_ai.training.manifest import read_manifest
from campus_safety_ai.training.model import FightTsn
from campus_safety_ai.training.video_dataset import _sample_frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, default=Path("datasets/manifests/fight-v1.csv"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("repeats must be at least 2")
    root = args.project_root.resolve()
    output = root / args.output
    if output.exists():
        raise FileExistsError(f"refusing to overwrite verification: {output}")

    import numpy as np
    import onnxruntime as ort
    import torch

    checkpoint_path = root / args.checkpoint
    model_path = root / args.onnx
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    torch.set_num_threads(1)
    model = FightTsn(frame_count=checkpoint["frame_count"], pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    selected = {}
    for sample in read_manifest(root / args.manifest):
        if sample.split == "val":
            selected.setdefault((sample.dataset, sample.label), sample)
    if not selected:
        raise ValueError("validation split must not be empty")
    results = []
    with torch.inference_mode():
        for sample in selected.values():
            inputs = _sample_frames(root / sample.video_path, checkpoint["frame_count"],
                                    checkpoint["image_size"], False)[None, ...]
            torch_logits = model(torch.from_numpy(inputs)).numpy()
            onnx_logits = session.run(None, {input_name: inputs})[0]
            error = float(np.max(np.abs(torch_logits - onnx_logits)))
            if not np.isfinite(error) or error > 1e-4:
                raise RuntimeError(f"ONNX parity failed on {sample.video_path}: {error}")
            results.append({"video_path": sample.video_path, "dataset": sample.dataset, "label": sample.label,
                            "max_abs_logit_error": error, "torch_logits": torch_logits.tolist(),
                            "onnx_logits": onnx_logits.tolist()})

    for _ in range(5):
        session.run(None, {input_name: inputs})
    milliseconds = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        session.run(None, {input_name: inputs})
        milliseconds.append((time.perf_counter() - started) * 1000)
    report = {
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "onnx_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "onnx_bytes": model_path.stat().st_size,
        "input_shape": list(inputs.shape), "validation_parity": results,
        "max_abs_logit_error": max(item["max_abs_logit_error"] for item in results),
        "cpu_latency": {"provider": "CPUExecutionProvider", "threads": 1, "warmup": 5,
                        "repeats": args.repeats, "median_ms": float(np.median(milliseconds)),
                        "p95_ms": float(np.percentile(milliseconds, 95)),
                        "raw_ms": milliseconds, "video_path": sample.video_path,
                        "scope": "ONNX session.run only; fixed validation clip; excludes decode/preprocess, "
                                 "network, evidence, and event delivery; not a live or RKNN benchmark"},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(output), "max_abs_logit_error": report["max_abs_logit_error"],
                      "median_ms": report["cpu_latency"]["median_ms"],
                      "p95_ms": report["cpu_latency"]["p95_ms"]}))


if __name__ == "__main__":
    main()
