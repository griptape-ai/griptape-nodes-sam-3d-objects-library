import json
import logging
import os
import subprocess
import sys
import tempfile
from typing import Any, cast

import huggingface_hub
from griptape.artifacts import ImageArtifact, ImageUrlArtifact, VideoUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, SuccessFailureNode
from griptape_nodes.exe_types.param_components.huggingface.huggingface_repo_file_parameter import (
    HuggingFaceRepoFileParameter,
)
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_components.seed_parameter import SeedParameter
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo
from griptape_nodes.files.file import File
from griptape_nodes.traits.options import Options
from PIL import Image

logger = logging.getLogger("sam_3d_objects_library")


class ReconstructSingleObject3D(SuccessFailureNode):
    """Reconstructs a 3D Gaussian splat of a single masked object from a 2D image.

    Takes an image and an optional binary mask (white object on black background). If no mask
    is connected, the image's alpha channel is used when present; otherwise a full-image mask
    is synthesized. Runs SAM 3D Objects inference and outputs a PLY file path.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # SeedParameter must be created first because after_value_set can be called
        # during parameter initialization, before _seed_param would otherwise exist.
        self._seed_param = SeedParameter(self)

        self._hf_repo_param = HuggingFaceRepoFileParameter(
            self,
            repo_files=[("facebook/sam-3d-objects", "checkpoints/pipeline.yaml")],
        )
        self._hf_repo_param.add_input_parameters()

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
                tooltip="(Optional) Binary mask image (white = object, black = background). If not connected, the image's alpha channel is used when present; otherwise the full image is treated as the object.",
            )
        )

        self._seed_param.add_input_parameters()

        output_format_param = Parameter(
            name="output_format",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            type="str",
            default_value="ply",
            tooltip="Output format: PLY (Gaussian splat), OBJ (mesh), or GLB (mesh)",
        )
        output_format_param.add_trait(Options(choices=["ply", "obj", "glb"]))
        self.add_parameter(output_format_param)

        self.add_parameter(
            Parameter(
                name="output_file_path",
                allowed_modes={ParameterMode.OUTPUT},
                output_type="str",
                default_value=None,
                tooltip="Absolute path to the saved 3D file (PLY, OBJ, or GLB)",
            )
        )
        self.add_parameter(
            ParameterVideo(
                name="video_preview",
                tooltip="Turntable preview of the reconstructed object",
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                settable=False,
                ui_options={"pulse_on_run": True, "expander": True},
            )
        )

        self._output_file = ProjectFileParameter(
            node=self,
            name="output_file",
            default_filename="sam3d_output.ply",
        )
        self._output_file.add_parameter()

        # Status parameters MUST be last
        self._create_status_parameters()

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        super().after_value_set(parameter, value)
        self._seed_param.after_value_set(parameter, value)

    def validate_before_node_run(self) -> list[Exception] | None:
        """Validate that required inputs are present."""
        errors: list[Exception] = []

        hf_errors = self._hf_repo_param.validate_before_node_run()
        if hf_errors:
            errors.extend(hf_errors)

        image = self.parameter_values.get("image")
        if not isinstance(image, (ImageArtifact, ImageUrlArtifact)):
            errors.append(ValueError("image is required and must be an ImageUrlArtifact or ImageArtifact"))

        mask = self.parameter_values.get("mask")
        if mask is not None and not isinstance(mask, (ImageArtifact, ImageUrlArtifact)):
            errors.append(ValueError("mask, if provided, must be an ImageUrlArtifact or ImageArtifact"))

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

    def _synthesize_mask_png(self, image_artifact: ImageArtifact | ImageUrlArtifact) -> str:
        """Synthesize a mask PNG when no mask is provided.

        Uses the image alpha channel when it is non-trivial (i.e. the image has transparent
        regions), otherwise generates a full-white mask treating the whole image as the object.
        """
        from io import BytesIO  # noqa: PLC0415

        image_bytes = self._artifact_to_bytes(image_artifact)
        pil_image = Image.open(BytesIO(image_bytes))

        if pil_image.mode == "RGBA":
            alpha = pil_image.split()[-1]
            alpha_min = cast(tuple[int, int], alpha.getextrema())[0]
            if alpha_min < 255:
                mask_image = alpha
            else:
                mask_image = Image.new("L", pil_image.size, 255)
        else:
            mask_image = Image.new("L", pil_image.size, 255)

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        mask_image.save(tmp.name)
        tmp.close()
        return tmp.name

    def process(self) -> AsyncResult[None]:
        """Kick off async inference."""
        yield lambda: self._run_inference()

    def _run_inference(self) -> None:
        """Run SAM 3D Objects single-object inference via subprocess."""
        image_artifact = self.parameter_values.get("image")
        mask_artifact = self.parameter_values.get("mask")
        output_format = self.parameter_values.get("output_format", "ply")

        if not isinstance(image_artifact, (ImageArtifact, ImageUrlArtifact)):
            raise ValueError("image is required")

        # Resolve config path from HuggingFace cache
        repo_id, revision = self._hf_repo_param.get_repo_revision()
        config_path = huggingface_hub.hf_hub_download(
            repo_id=repo_id,
            revision=revision,
            filename="checkpoints/pipeline.yaml",
            local_files_only=True,
        )

        # Use a temp file for subprocess output, then copy to project outputs
        tmp_output = tempfile.NamedTemporaryFile(suffix=f".{output_format}", delete=False)
        tmp_output.close()
        output_path = tmp_output.name

        self._seed_param.preprocess()
        seed = self._seed_param.get_seed()

        # Save inputs to temp files so the subprocess can read them
        image_path = self._save_artifact_to_temp(image_artifact, suffix=".png")
        if isinstance(mask_artifact, (ImageArtifact, ImageUrlArtifact)):
            mask_path = self._save_artifact_to_temp(mask_artifact, suffix=".png")
        else:
            mask_path = self._synthesize_mask_png(image_artifact)

        try:
            request = {
                "action": "single",
                "image_path": image_path,
                "mask_paths": [mask_path],
                "config_path": config_path,
                "output_path": output_path,
                "output_format": output_format,
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

            # Copy the output 3D file to the project outputs folder
            with open(result["output_path"], "rb") as f:
                output_bytes = f.read()
            # Update output filename to match the selected format
            current_name = self.get_parameter_value("output_file") or "sam3d_output.ply"
            base, _ = os.path.splitext(current_name)
            self.set_parameter_value("output_file", f"{base}.{output_format}")
            dest = self._output_file.build_file()
            saved = dest.write_bytes(output_bytes)
            logger.info(f"Saved 3D output ({output_format}) to {saved.location}")
            self.parameter_output_values["output_file_path"] = saved.location

            # Save the video preview and set the video output
            video_path = result.get("video_path")
            if video_path and os.path.isfile(video_path):
                with open(video_path, "rb") as f:
                    video_bytes = f.read()
                from griptape_nodes.files.project_file import ProjectFileDestination

                preview_dest = cast(Any, ProjectFileDestination).from_situation(
                    "sam3d_preview.mp4", "save_node_output", node_name=self.name
                )
                preview_saved = preview_dest.write_bytes(video_bytes)
                self.parameter_output_values["video_preview"] = VideoUrlArtifact(
                    value=preview_saved.location, name=preview_saved.name
                )
                logger.info(f"Saved turntable preview to {preview_saved.location}")
        finally:
            # Clean up temp files
            for path in [image_path, mask_path, output_path]:
                try:
                    os.unlink(path)
                except OSError:
                    pass
            # Also clean up the temp video if it was created
            video_tmp = os.path.splitext(output_path)[0] + ".mp4"
            try:
                os.unlink(video_tmp)
            except OSError:
                pass
