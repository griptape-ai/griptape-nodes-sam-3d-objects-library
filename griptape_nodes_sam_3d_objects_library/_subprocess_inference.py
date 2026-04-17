"""Subprocess entry-point for SAM 3D Objects inference.

This script is executed by the library's venv Python in a subprocess so that
torch 2.5.1 + kaolin can load without conflicting with the engine's torch version.

Protocol (over stdin/stdout as JSON):
  Request:  {"action": "single"|"multi", "image_path": str, "mask_paths": [str],
             "config_path": str, "output_ply_path": str, "seed": int,
             "submodule_root": str}
  Response: {"status": "ok", "ply_path": str} | {"status": "error", "message": str}
"""

import json
import os
import sys


def _setup(submodule_root: str) -> None:
    """Add submodule paths and set environment variables."""
    notebook_dir = os.path.join(submodule_root, "notebook")
    for p in (submodule_root, notebook_dir):
        if p not in sys.path:
            sys.path.insert(0, p)
    if "CONDA_PREFIX" not in os.environ:
        os.environ["CONDA_PREFIX"] = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    os.environ["LIDRA_SKIP_INIT"] = "true"


_inference_cache: dict[str, object] = {}


def _get_inference(config_path: str) -> object:
    if config_path not in _inference_cache:
        from inference import Inference  # type: ignore[import-not-found]

        _inference_cache[config_path] = Inference(config_path, compile=False)
    return _inference_cache[config_path]


def _run_single(request: dict) -> dict:
    import numpy as np
    from PIL import Image

    inference = _get_inference(request["config_path"])

    image = np.array(Image.open(request["image_path"]).convert("RGB"), dtype=np.uint8)
    mask = np.array(Image.open(request["mask_paths"][0]))
    if mask.ndim == 3:
        mask = mask[..., -1]
    mask = mask > 0

    output = inference(image, mask, seed=request["seed"])
    output["gaussian"][0].save_ply(request["output_ply_path"])
    return {"status": "ok", "ply_path": request["output_ply_path"]}


def _run_multi(request: dict) -> dict:
    import numpy as np
    from PIL import Image

    inference = _get_inference(request["config_path"])
    from inference import make_scene  # type: ignore[import-not-found]

    image = np.array(Image.open(request["image_path"]).convert("RGB"), dtype=np.uint8)
    masks = []
    for mp in request["mask_paths"]:
        m = np.array(Image.open(mp))
        if m.ndim == 3:
            m = m[..., -1]
        masks.append(m > 0)

    outputs = [inference(image, m, seed=request["seed"]) for m in masks]
    scene_gs = make_scene(*outputs)
    scene_gs.save_ply(request["output_ply_path"])
    return {"status": "ok", "ply_path": request["output_ply_path"]}


def main() -> None:
    request = json.loads(sys.stdin.readline())
    try:
        _setup(request["submodule_root"])
        if request["action"] == "single":
            result = _run_single(request)
        elif request["action"] == "multi":
            result = _run_multi(request)
        else:
            result = {"status": "error", "message": f"Unknown action: {request['action']}"}
    except Exception as e:
        result = {"status": "error", "message": f"{type(e).__name__}: {e}"}
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
