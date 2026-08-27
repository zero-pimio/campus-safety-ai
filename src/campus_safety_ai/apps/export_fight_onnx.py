from __future__ import annotations

import argparse
import json
from pathlib import Path

from campus_safety_ai.training.model import FightTsn


def export(checkpoint_path: Path, output_path: Path, opset: int = 13) -> Path:
    import numpy as np
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    frame_count = int(checkpoint["frame_count"])
    image_size = int(checkpoint["image_size"])
    model = FightTsn(frame_count=frame_count, pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    example = torch.zeros(1, frame_count * 3, image_size, image_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        example,
        output_path,
        input_names=["frames"],
        output_names=["logits"],
        opset_version=opset,
        dynamo=False,
    )

    import onnx

    graph = onnx.load(output_path)
    onnx.checker.check_model(graph)
    with torch.inference_mode():
        torch_logits = model(example).numpy()
    parity = None
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
        onnx_logits = session.run(["logits"], {"frames": example.numpy()})[0]
        parity = float(np.max(np.abs(torch_logits - onnx_logits)))
        if parity > 1e-4:
            raise RuntimeError(f"ONNX parity error is too high: {parity}")
    except ImportError:
        pass

    metadata = {
        "model_name": checkpoint["model_name"],
        "frame_count": frame_count,
        "input_shape": [1, frame_count * 3, image_size, image_size],
        "input_layout": "N,(T*C),H,W",
        "color": "RGB",
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "class_names": checkpoint["class_names"],
        "opset": opset,
        "max_abs_parity_error": parity,
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"onnx -> {output_path}")
    print(json.dumps(metadata, ensure_ascii=False))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a fight TSN checkpoint to ONNX")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=13)
    arguments = parser.parse_args()
    export(arguments.checkpoint, arguments.output, arguments.opset)


if __name__ == "__main__":
    main()
