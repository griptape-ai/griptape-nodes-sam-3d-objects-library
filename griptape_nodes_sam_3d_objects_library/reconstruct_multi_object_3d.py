import json
import logging
import os
import sys
from typing import Any

import numpy as np
from griptape.artifacts import ImageArtifact, ImageUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, SuccessFailureNode
from griptape_nodes.exe_types.param_components.seed_parameter import SeedParameter
from griptape_nodes.files.file import File
from PIL import Image

logger = logging.getLogger("sam_3d_objects_library")


class ReconstructMultiObject3D(SuccessFailureNode):
    """Reconstructs a merged 3D Gaussian splat scene from multiple masked objects in a single image.

    Takes an image and a list of mask images, runs SAM 3D Objects inference once per mask,
    merges all Gaussians into a single scene, and outputs a PLY file.
    """

    # Class-level model cache - shared across all instances, keyed on config_path
    _inference_cache: dict[str, Any] = {}

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # SeedParameter must be created first because after_value_set can be called
        # during parameter initialization, before _seed_param would otherwise exist.
        self._seed_param = SeedParameter(self)

        self.add_parameter(
            Parameter(
                name="image",
                allowed_modes={ParameterMode.INPUT},
                type="ImageUrlArtifact",
                input_types=["ImageUrlArtifact", "ImageArtifact"],
                default_value=None,
                tooltip="Input image (RGB or RGBA) containing the objects to reconstruct",
            )
        )
        self.add_parameter(
            Parameter(
                name="masks",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="str",
                default_value=None,
                tooltip="JSON array of image URLs, each a binary mask (white = object, black = background) for one object",
            )
        )

        self._seed_param.add_input_parameters()

        self.add_parameter(
            Parameter(
                name="config_path",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="str",
                default_value="checkpoints/hf/pipeline.yaml",
                tooltip="Path to the Hydra pipeline.yaml config file (relative to the sam-3d-objects repo root or absolute)",
            )
        )
        self.add_parameter(
            Parameter(
                name="output_ply_path",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="str",
                default_value="/tmp/sam3d_multi_output.ply",  # noqa: S108
                tooltip="File path where the merged output Gaussian splat PLY will be saved",
            )
        )
        self.add_parameter(
            Parameter(
                name="ply_path",
                allowed_modes={ParameterMode.OUTPUT},
                output_type="str",
                default_value=None,
                tooltip="Absolute path to the saved PLY file containing the merged 3D Gaussian splat scene",
            )
        )

        # Status parameters MUST be last
        self._create_status_parameters()

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        super().after_value_set(parameter, value)
        self._seed_param.after_value_set(parameter, value)

    def validate_before_node_run(self) -> list[Exception] | None:
        """Validate that required inputs are present."""
        errors: list[Exception] = []

        image = self.parameter_values.get("image")
        if not isinstance(image, (ImageArtifact, ImageUrlArtifact)):
            errors.append(ValueError("image is required and must be an ImageUrlArtifact or ImageArtifact"))

        masks_str = self.parameter_values.get("masks")
        if not masks_str:
            errors.append(ValueError("masks is required (JSON array of image URLs)"))
        else:
            try:
                masks_list = json.loads(masks_str)
                if not isinstance(masks_list, list) or len(masks_list) == 0:
                    errors.append(ValueError("masks must be a non-empty JSON array of image URLs"))
            except (json.JSONDecodeError, TypeError) as e:
                errors.append(ValueError(f"masks must be a valid JSON array: {e}"))

        config_path = self.parameter_values.get("config_path")
        if not config_path:
            errors.append(ValueError("config_path is required"))

        output_ply_path = self.parameter_values.get("output_ply_path")
        if not output_ply_path:
            errors.append(ValueError("output_ply_path is required"))

        return errors if errors else None

    def _get_submodule_root(self) -> str:
        assert __file__ is not None
        return os.path.join(os.path.dirname(__file__), "sam-3d-objects")

    def _ensure_sys_path(self) -> None:
        """Ensure the submodule root and notebook directory are on sys.path."""
        submodule_root = self._get_submodule_root()
        notebook_dir = os.path.join(submodule_root, "notebook")
        if submodule_root not in sys.path:
            sys.path.insert(0, submodule_root)
        if notebook_dir not in sys.path:
            sys.path.insert(0, notebook_dir)

    def _load_inference(self, config_path: str) -> Any:
        """Load and cache the Inference object keyed on config_path."""
        if config_path in ReconstructMultiObject3D._inference_cache:
            return ReconstructMultiObject3D._inference_cache[config_path]

        self._ensure_sys_path()

        # DEFERRED IMPORT: inference.py is in notebook/ inside the submodule and is
        # not an installable package. It must be accessed via sys.path injection.
        # notebook/inference.py unconditionally does:
        #   os.environ["CUDA_HOME"] = os.environ["CONDA_PREFIX"]
        # so CONDA_PREFIX must exist before import or it raises KeyError.
        # In non-conda environments (e.g. Griptape Nodes), fall back to CUDA_HOME
        # or the standard CUDA install path.
        if "CONDA_PREFIX" not in os.environ:
            os.environ["CONDA_PREFIX"] = os.environ.get("CUDA_HOME", "/usr/local/cuda")
        os.environ["LIDRA_SKIP_INIT"] = "true"

        from inference import Inference  # type: ignore[import-not-found]  # noqa: PLC0415

        logger.info(f"Loading SAM 3D Objects inference pipeline from {config_path}...")
        inference = Inference(config_path, compile=False)
        ReconstructMultiObject3D._inference_cache[config_path] = inference
        logger.info("Inference pipeline loaded successfully")
        return inference

    def _artifact_to_bytes(self, artifact: ImageArtifact | ImageUrlArtifact) -> bytes:
        """Read raw bytes from either an ImageArtifact (bytes value) or ImageUrlArtifact (path value)."""
        if isinstance(artifact, ImageArtifact):
            # ImageArtifact.value is bytes - use to_bytes() to get the raw bytes
            return artifact.to_bytes()
        # ImageUrlArtifact.value is a str path or macro path
        artifact_path = artifact.value
        if not isinstance(artifact_path, str):
            raise ValueError(f"ImageUrlArtifact.value must be a str, got {type(artifact_path)}")
        return File(artifact_path).read_bytes()

    def _read_image_as_array(self, artifact: ImageArtifact | ImageUrlArtifact) -> np.ndarray:
        """Read an image artifact into a uint8 numpy array (HxWxC)."""
        from io import BytesIO  # noqa: PLC0415

        image_bytes = self._artifact_to_bytes(artifact)
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        return np.array(pil_image, dtype=np.uint8)

    def _read_mask_as_bool_array(self, url: str) -> np.ndarray:
        """Read a mask URL into a 2D boolean numpy array (True = object pixel)."""
        from io import BytesIO  # noqa: PLC0415

        mask_bytes = File(url).read_bytes()
        pil_mask = Image.open(BytesIO(mask_bytes))
        mask_array = np.array(pil_mask)
        # Reduce to 2D if the mask has channels
        if mask_array.ndim == 3:
            mask_array = mask_array[..., -1]
        return mask_array > 0

    def process(self) -> AsyncResult[None]:
        """Kick off async inference."""
        yield lambda: self._run_inference()

    def _run_inference(self) -> None:
        """Run SAM 3D Objects multi-object inference (called in background thread)."""
        image_artifact = self.parameter_values.get("image")
        masks_str = self.parameter_values.get("masks")
        config_path = self.parameter_values.get("config_path")
        output_ply_path = self.parameter_values.get("output_ply_path")

        if not isinstance(image_artifact, (ImageArtifact, ImageUrlArtifact)):
            raise ValueError("image is required")
        if not isinstance(masks_str, str):
            raise ValueError("masks is required")
        if not isinstance(config_path, str):
            raise ValueError("config_path is required")
        if not isinstance(output_ply_path, str):
            raise ValueError("output_ply_path is required")

        mask_urls: list[str] = json.loads(masks_str)
        image_array = self._read_image_as_array(image_artifact)
        mask_arrays = [self._read_mask_as_bool_array(url) for url in mask_urls]

        self._seed_param.preprocess()
        seed = self._seed_param.get_seed()

        inference = self._load_inference(config_path)

        # DEFERRED IMPORT: make_scene is in notebook/inference.py, accessed via sys.path injection
        from inference import make_scene  # type: ignore[import-not-found]  # noqa: PLC0415

        logger.info(f"Running SAM 3D Objects multi-object inference on {len(mask_arrays)} masks...")
        outputs = [inference(image_array, mask_array, seed=seed) for mask_array in mask_arrays]

        scene_gs = make_scene(*outputs)
        scene_gs.save_ply(output_ply_path)
        logger.info(f"Saved merged Gaussian splat PLY to {output_ply_path}")

        self.parameter_output_values["ply_path"] = output_ply_path
