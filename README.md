# Griptape Nodes SAM 3D Objects Library

A [Griptape Nodes](https://www.griptapenodes.com/) library for 3D object reconstruction from single images using [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects).

## Overview

This library wraps Meta's SAM 3D Objects foundation model, which reconstructs full 3D shape geometry, texture, and spatial layout from a single 2D image. Given an image and one or more binary object masks, the model produces 3D representations of the masked objects in PLY (Gaussian splat), OBJ (mesh), or GLB (mesh) format. Both single-object and multi-object (merged scene) reconstruction workflows are supported, with turntable video previews. The model handles real-world challenges including occlusion, clutter, and unusual object poses.

## Requirements

- **GPU**: CUDA required (Linux 64-bit with NVIDIA GPU, minimum 32 GB VRAM)
- **Griptape Nodes Engine**: Version 0.80.0 or later

## Nodes

### Reconstruct Single Object 3D

Reconstructs a 3D object from a single masked region in a 2D image. Outputs the 3D file and a turntable video preview.

**Parameters:**

| Parameter | Type | Direction | Description |
|-----------|------|-----------|-------------|
| `model` | str | Input | HuggingFace model selector (downloads via Model Management if not cached) |
| `image` | ImageUrlArtifact | Input | Input image (RGB or RGBA) containing the object to reconstruct |
| `mask` | ImageUrlArtifact | Input | Binary mask image (white = object, black = background) |
| `randomize_seed` | bool | Property | Randomize the seed on each run |
| `seed` | int | Property | Random seed for reproducible results (default: 42) |
| `output_format` | str | Property | Output format: `ply`, `obj`, or `glb` (default: `ply`) |
| `output_file` | str | Property | Output filename (auto-increments in project outputs folder) |
| `output_file_path` | str | Output | Absolute path to the saved 3D file |
| `video_preview` | VideoUrlArtifact | Output | Turntable MP4 preview of the reconstructed object |

### Reconstruct Multi-Object 3D

Reconstructs a merged 3D scene from multiple masked objects in a single image. Runs inference once per mask, merges all results, and outputs the 3D file with a turntable video preview.

**Parameters:**

| Parameter | Type | Direction | Description |
|-----------|------|-----------|-------------|
| `model` | str | Input | HuggingFace model selector (downloads via Model Management if not cached) |
| `image` | ImageUrlArtifact | Input | Input image (RGB or RGBA) containing the objects to reconstruct |
| `masks` | str | Input | JSON array of image URLs, each a binary mask for one object |
| `randomize_seed` | bool | Property | Randomize the seed on each run |
| `seed` | int | Property | Random seed for reproducible results (default: 42) |
| `output_format` | str | Property | Output format: `ply`, `obj`, or `glb` (default: `ply`) |
| `output_file` | str | Property | Output filename (auto-increments in project outputs folder) |
| `output_file_path` | str | Output | Absolute path to the saved 3D file |
| `video_preview` | VideoUrlArtifact | Output | Turntable MP4 preview of the reconstructed scene |

## Model Download

The `facebook/sam-3d-objects` model is a gated HuggingFace repository (~12 GB). When the node is first added, it will show a **"Huggingface Model Download Required"** warning with a link to Model Management.

1. Request access at [huggingface.co/facebook/sam-3d-objects](https://huggingface.co/facebook/sam-3d-objects)
2. Authenticate with `huggingface-cli login`
3. Download the model via **Settings → Model Management** in the Griptape Nodes editor
4. After download, a model selector dropdown will appear on the node

## Installation

### Prerequisites

- [Griptape Nodes](https://github.com/griptape-ai/griptape-nodes) installed and running
- A Linux 64-bit machine with an NVIDIA GPU (minimum 32 GB VRAM)
- CUDA 12.1 drivers installed
- A HuggingFace account with approved access to [facebook/sam-3d-objects](https://huggingface.co/facebook/sam-3d-objects)

### Install the Library

1. **Clone the repository** to your Griptape Nodes workspace directory:

   ```bash
   cd `gtn config show workspace_directory`
   git clone --recurse-submodules https://github.com/griptape-ai/griptape-nodes-sam-3d-objects-library.git
   ```

2. **Add the library** in the Griptape Nodes Editor:

   - Open the Settings menu and navigate to the *Libraries* settings
   - Click on *+ Add Library* at the bottom of the settings panel
   - Enter the path to the library JSON file:
     ```
     <workspace_directory>/griptape-nodes-sam-3d-objects-library/griptape_nodes_sam_3d_objects_library/griptape-nodes-library.json
     ```
   - Close the Settings Panel
   - Click on *Refresh Libraries*

3. **Download the model** via Settings → Model Management (search for `facebook/sam-3d-objects`).

4. **Verify installation** by checking that the nodes appear in the node palette under the "3D Reconstruction" category.

## Usage

### Reconstruct Single Object 3D

1. Add a **Reconstruct Single Object 3D** node to your workflow.
2. Select the downloaded model from the dropdown (or follow the download prompt).
3. Connect an image artifact to `image` and a binary mask to `mask`.
4. Choose your desired `output_format` (ply, obj, or glb).
5. Run the node. The 3D file is saved to the project outputs folder and the turntable preview appears in the `video_preview` output.

### Reconstruct Multi-Object 3D

1. Add a **Reconstruct Multi-Object 3D** node to your workflow.
2. Select the downloaded model from the dropdown.
3. Connect an image artifact to `image`.
4. Set `masks` to a JSON array of mask image URL strings.
5. Choose your desired `output_format`.
6. Run the node. All objects are merged into a single 3D scene.

## Troubleshooting

### Library Not Loading

- Ensure the git submodule is initialized. If you cloned without `--recurse-submodules`, run:
  ```bash
  git submodule update --init --recursive
  ```

### CUDA Not Available

- Verify your NVIDIA GPU drivers and CUDA 12.1 are correctly installed.
- Run `nvidia-smi` to confirm your GPU is detected.
- This library requires Linux 64-bit; it will not run on macOS or Windows.

### Out of Memory Errors

- A minimum of 32 GB GPU VRAM is required per the official SAM 3D Objects documentation.
- Close other GPU-intensive applications before running inference.

### HuggingFace Access Denied

- The `facebook/sam-3d-objects` model is gated. Request access at https://huggingface.co/facebook/sam-3d-objects and wait for approval before attempting to download.
- Authenticate with `hf auth login` using your HuggingFace credentials before downloading.

### Hydra Patch Not Applied

- The library advanced loader applies the hydra patch automatically on first run. If you see Hydra-related errors, check that the patch script exists at `sam-3d-objects/patching/hydra` and is executable.

## Additional Resources

- [SAM 3D Objects GitHub](https://github.com/facebookresearch/sam-3d-objects)
- [Griptape Nodes Documentation](https://docs.griptapenodes.com/)
- [Griptape Discord](https://discord.gg/griptape)

## License

This library is provided under the Apache License 2.0. The bundled SAM 3D Objects submodule is subject to its own license: SAM License (Meta custom license, non-standard; see the LICENSE file in the submodule for details, and note that explicit access request approval is required for HuggingFace checkpoints).
