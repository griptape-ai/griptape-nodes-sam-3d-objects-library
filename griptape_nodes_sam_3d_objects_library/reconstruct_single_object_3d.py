import json
import logging
import os
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
from griptape.artifacts import ImageArtifact, ImageUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, SuccessFailureNode
from griptape_nodes.exe_types.param_components.seed_parameter import SeedParameter
from griptape_nodes.files.file import File
from PIL import Image

logger = logging.getLogger("sam_3d_objects_library")


class ReconstructSingleObject3D(SuccessFailureNode):
    """Reconstructs a 3D Gaussian splat of a single masked object from a 2D image.

    Takes an image and a binary mask (white object on black background), runs SAM 3D Objects
    inference, and outputs a PLY file path containing the Gaussian splat.
    """

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
                tooltip="Input image (RGB or RGBA) containing the object to reconstruct",
            )
        )
        self.add_parameter(
            Parameter(
                name="mask",
                allowed_modes={ParameterMode.INPUT},
                type="ImageUrlArtifact",
                input_types=["ImageUrlArtifact", "ImageArtifact"],
                default_value=None,
                tooltip="Binary mask image (white = object, black = background) indicating which object to reconstruct",
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
                default_value="/tmp/sam3d_output.ply",  # noqa: S108
                tooltip="File path where the output Gaussian splat PLY will be saved",
            )
        )
        self.add_parameter(
            Parameter(
                name="ply_path",
                allowed_modes={ParameterMode.OUTPUT},
                output_type="str",
                default_value=None,
                tooltip="Absolute path to the saved PLY file containing the 3D Gaussian splat",
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

        mask = self.parameter_values.get("mask")
        if not isinstance(mask, (ImageArtifact, ImageUrlArtifact)):
            errors.append(ValueError("mask is required and must be an ImageUrlArtifact or ImageArtifact"))

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

    def _get_venv_python(self) -> str:
        assert __file__ is not None
        lib_root = os.path.dirname(__file__)
        if sys.platform == "win32":
            return os.path.join(lib_root, ".venv", "Scripts", "python.exe")
        return os.path.join(lib_root, ".venv", "bin", "python")

    def _artifact_to_bytes(self, artifact: ImageArtifact | ImageUrlArtifact) -> bytes:
        """Read raw bytes from either an ImageArtifact (bytes value) or ImageUrlArtifact (path value)."""
        if isinstance(artifact, ImageArtifact):
            return artifact.to_bytes()
        artifact_path = artifact.value
        if not isinstance(artifact_path, str):
            raise ValueError(f"ImageUrlArtifact.value must be a str, got {type(artifact_path)}")
        return File(artifact_path).read_bytes()

    def _save_artifact_to_temp(self, artifact: ImageArtifact | ImageUrlArtifact, suffix: str = ".png") -> str:
        """Save an image artifact to a temp file and return the path."""
        from io import BytesIO  # noqa: PLC0415

        image_bytes = self._artifact_to_bytes(artifact)
        pil_image = Image.open(BytesIO(image_bytes))
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        pil_image.save(tmp.name)
        tmp.close()
        return tmp.name

    def process(self) -> AsyncResult[None]:
        """Kick off async inference."""
        yield lambda: self._run_inference()

    def _run_inference(self) -> None:
        """Run SAM 3D Objects single-object inference via subprocess."""
        image_artifact = self.parameter_values.get("image")
        mask_artifact = self.parameter_values.get("mask")
        config_path = self.parameter_values.get("config_path")
        output_ply_path = self.parameter_values.get("output_ply_path")

        if not isinstance(image_artifact, (ImageArtifact, ImageUrlArtifact)):
            raise ValueError("image is required")
        if not isinstance(mask_artifact, (ImageArtifact, ImageUrlArtifact)):
            raise ValueError("mask is required")
        if not isinstance(config_path, str):
            raise ValueError("config_path is required")
        if not isinstance(output_ply_path, str):
            raise ValueError("output_ply_path is required")

        self._seed_param.preprocess()
        seed = self._seed_param.get_seed()

        # Save inputs to temp files so the subprocess can read them
        image_path = self._save_artifact_to_temp(image_artifact, suffix=".png")
        mask_path = self._save_artifact_to_temp(mask_artifact, suffix=".png")

        try:
            request = {
                "action": "single",
                "image_path": image_path,
                "mask_paths": [mask_path],
                "config_path": config_path,
                "output_ply_path": output_ply_path,
                "seed": seed,
                "submodule_root": self._get_submodule_root(),
            }

            runner_script = os.path.join(os.path.dirname(__file__), "_subprocess_inference.py")
            venv_python = self._get_venv_python()
            logger.info("Running SAM 3D Objects single-object inference in subprocess...")

            proc = subprocess.run(
                [venv_python, runner_script],
                input=json.dumps(request) + "\n",
                capture_output=True,
                text=True,
            )

            if proc.returncode != 0:
                raise RuntimeError(f"Inference subprocess failed (exit {proc.returncode}):\n{proc.stderr}")

            result = json.loads(proc.stdout.strip().split("\n")[-1])
            if result.get("status") != "ok":
                raise RuntimeError(f"Inference error: {result.get('message', 'unknown error')}")

            logger.info(f"Saved Gaussian splat PLY to {output_ply_path}")
            self.parameter_output_values["ply_path"] = result["ply_path"]
        finally:
            # Clean up temp files
            for path in (image_path, mask_path):
                try:
                    os.unlink(path)
                except OSError:
                    pass
