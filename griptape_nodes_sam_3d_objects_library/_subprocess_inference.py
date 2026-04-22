"""Subprocess entry-point for SAM 3D Objects inference.

This script is executed by the library's venv Python in a subprocess so that
torch 2.5.1 + kaolin can load without conflicting with the engine's torch version.

Protocol (over stdin/stdout as JSON):
  Request:  {"action": "single"|"multi", "image_path": str, "mask_paths": [str],
             "config_path": str, "output_path": str, "output_format": "ply"|"obj",
             "seed": int, "submodule_root": str}
  Response: {"status": "ok", "output_path": str, ...} | {"status": "error", "message": str}
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
    from inference import (
        make_scene,
        ready_gaussian_for_video_rendering,
        render_video,
    )

    image = np.array(Image.open(request["image_path"]).convert("RGB"), dtype=np.uint8)
    mask = np.array(Image.open(request["mask_paths"][0]))
    if mask.ndim == 3:
        mask = mask[..., -1]
    mask = mask > 0

    output = inference(image, mask, seed=request["seed"])
    output_path = request["output_path"]
    output_format = request.get("output_format", "ply")
    if output_format in ("obj", "glb"):
        mesh = make_scene_untextured_separate_meshes(output)
        if mesh is None:
            return {"status": "error", "message": "No mesh could be generated"}
        mesh.export(output_path)
    else:
        output["gaussian"][0].save_ply(output_path)

    # Render turntable video preview
    scene_gs = make_scene(output)
    scene_gs_video = ready_gaussian_for_video_rendering(scene_gs)
    video_frames = render_video(
        scene_gs_video,
        r=1,
        fov=60,
        resolution=512,
        num_frames=60,
    )["color"]

    import imageio  # noqa: PLC0415

    video_path = os.path.splitext(output_path)[0] + ".mp4"
    writer = imageio.get_writer(
        video_path,
        fps=30,
        codec="libx264",
        output_params=["-movflags", "+faststart", "-pix_fmt", "yuv420p"],
    )
    for frame in video_frames:
        writer.append_data(frame)
    writer.close()

    return {"status": "ok", "output_path": output_path, "video_path": video_path}


import torch
from pytorch3d.transforms import quaternion_to_matrix, Transform3d
# From sam3d github: https://github.com/facebookresearch/sam-3d-objects/blob/main/sam3d_objects/data/dataset/tdfy/transforms_3d.py
def compose_transform(

    scale: torch.Tensor, rotation: torch.Tensor, translation: torch.Tensor
) -> Transform3d:
    """
    Args:
        scale: (..., 3) tensor of scale factors
        rotation: (..., 3, 3) tensor of rotation matrices
        translation: (..., 3) tensor of translation vectors
    """
    tfm = Transform3d(dtype=scale.dtype, device=scale.device)
    return tfm.scale(scale).rotate(rotation).translate(translation)


from copy import deepcopy
import numpy as np
# From sam3d github issues: https://github.com/facebookresearch/sam-3d-objects/issues/56#issuecomment-3614031150
def make_scene_untextured_separate_meshes(*outputs, in_place=False):
    import trimesh
    _R_ZUP_TO_YUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
    _R_YUP_TO_ZUP = _R_ZUP_TO_YUP.T

    if not in_place:
        outputs = [deepcopy(output) for output in outputs]

    all_meshes = []
    for output in outputs:
        mesh = output["glb"]
        if mesh is None:
            continue

        # GLB is Y-up, transforms are Z-up; convert, apply, convert back
        vertices = mesh.vertices.astype(np.float32) @ _R_YUP_TO_ZUP
        vertices_tensor = torch.from_numpy(vertices).float().to(output["rotation"].device)
        R_l2c = quaternion_to_matrix(output["rotation"])
        l2c_transform = compose_transform(
            scale=output["scale"],
            rotation=R_l2c,
            translation=output["translation"],
        )
        vertices = l2c_transform.transform_points(vertices_tensor.unsqueeze(0))
        mesh.vertices = vertices.squeeze(0).cpu().numpy() @ _R_ZUP_TO_YUP
        all_meshes.append(mesh)

    if not all_meshes:
        return None

    if len(all_meshes) == 1:
        return all_meshes[0]

    # merge meshes into single mesh
    return trimesh.util.concatenate(all_meshes)
    
    
def _run_multi(request: dict) -> dict:
    import numpy as np
    from PIL import Image

    inference = _get_inference(request["config_path"])
    from inference import (
        make_scene, 
        ready_gaussian_for_video_rendering,
        render_video,
    )

    image = np.array(Image.open(request["image_path"]).convert("RGB"), dtype=np.uint8)
    masks = []
    for mp in request["mask_paths"]:
        m = np.array(Image.open(mp))
        if m.ndim == 3:
            m = m[..., -1]
        masks.append(m > 0)

    outputs = [inference(image, m, seed=request["seed"]) for m in masks]
    scene_gs = make_scene(*outputs)

    output_path = request["output_path"]
    output_format = request.get("output_format", "ply")
    if output_format in ("obj", "glb"):
        mesh = make_scene_untextured_separate_meshes(*outputs)
        if mesh is None:
            return {"status": "error", "message": "No mesh could be generated"}
        mesh.export(output_path)
    else:
        scene_gs.save_ply(output_path)

    # Prepare scene for video rendering
    scene_gs_video = ready_gaussian_for_video_rendering(scene_gs)
    # Render a turntable preview video
    video_frames = render_video(
        scene_gs_video,
        r=1,
        fov=60,
        resolution=512,
        num_frames=60,
    )["color"]

    # Convert frames to numpy uint8 arrays for imageio
    import imageio  # noqa: PLC0415

    # Save as MP4 next to the output file
    video_path = os.path.splitext(output_path)[0] + ".mp4"
    writer = imageio.get_writer(
        video_path,
        fps=30,
        codec="libx264",
        output_params=["-movflags", "+faststart", "-pix_fmt", "yuv420p"],
    )
    for frame in video_frames:
        writer.append_data(frame)
    writer.close()
    return {"status": "ok", "output_path": output_path, "video_path": video_path}


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
