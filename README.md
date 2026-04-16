# Griptape Nodes SAM 3D Objects Library

A [Griptape Nodes](https://www.griptapenodes.com/) library for 3D object reconstruction from single images using [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects).

## Overview

This library wraps Meta's SAM 3D Objects foundation model, which reconstructs full 3D shape geometry, texture, and spatial layout from a single 2D image. Given an image and one or more binary object masks, the model produces Gaussian splat representations (PLY format) of the masked objects. Both single-object and multi-object (merged scene) reconstruction workflows are supported. The model handles real-world challenges including occlusion, clutter, and unusual object poses.

## Requirements

- **GPU**: CUDA required (Linux 64-bit with NVIDIA GPU, minimum 32 GB VRAM)
- **Griptape Nodes Engine**: Version 0.80.0 or later

## Nodes

### Reconstruct Single Object 3D

Reconstructs a 3D Gaussian splat of a single masked object from a 2D image. Takes an image and a binary mask indicating which object to reconstruct, then outputs a PLY file path containing the Gaussian splat.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `image` | ImageUrlArtifact | Input image (RGB or RGBA) containing the object to reconstruct |
| `mask` | ImageUrlArtifact | Binary mask image (white = object, black = background) |
| `randomize_seed` | bool | Randomize the seed on each run |
| `seed` | int | Random seed for reproducible results (default: 42) |
| `config_path` | str | Path to the Hydra pipeline.yaml config file (default: `checkpoints/hf/pipeline.yaml`) |
| `output_ply_path` | str | File path where the output Gaussian splat PLY will be saved |
| `ply_path` (output) | str | Absolute path to the saved PLY file containing the 3D Gaussian splat |

### Reconstruct Multi-Object 3D

Reconstructs a merged 3D Gaussian splat scene from multiple masked objects in a single image. Runs inference once per mask, merges all Gaussian splats into a single scene, and outputs a PLY file.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `image` | ImageUrlArtifact | Input image (RGB or RGBA) containing the objects to reconstruct |
| `masks` | str | JSON array of image URLs, each a binary mask for one object |
| `randomize_seed` | bool | Randomize the seed on each run |
| `seed` | int | Random seed for reproducible results, applied to each object (default: 42) |
| `config_path` | str | Path to the Hydra pipeline.yaml config file (default: `checkpoints/hf/pipeline.yaml`) |
| `output_ply_path` | str | File path where the merged output Gaussian splat PLY will be saved |
| `ply_path` (output) | str | Absolute path to the saved PLY file containing the merged 3D Gaussian splat scene |

## Available Models

The following model is available from HuggingFace (gated - requires access request approval):

| Model | Description |
|-------|-------------|
| `facebook/sam-3d-objects` | Full SAM 3D Objects model weights. Contains the `checkpoints/` directory with `pipeline.yaml` and associated model weights. Requires 32+ GB VRAM. |

Models must be downloaded manually using the HuggingFace CLI after receiving access approval:

```bash
hf auth login
hf download facebook/sam-3d-objects
```

The downloaded checkpoints directory should be placed (or symlinked) inside the `sam-3d-objects` submodule, matching the path expected by `config_path` (default: `checkpoints/hf/pipeline.yaml` relative to the submodule root).

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

2. **Download the model checkpoints** from HuggingFace after access is approved:

   ```bash
   hf auth login
   hf download facebook/sam-3d-objects --local-dir \
     griptape-nodes-sam-3d-objects-library/griptape_nodes_sam_3d_objects_library/sam-3d-objects/checkpoints/hf
   ```

3. **Add the library** in the Griptape Nodes Editor:

   - Open the Settings menu and navigate to the *Libraries* settings
   - Click on *+ Add Library* at the bottom of the settings panel
   - Enter the path to the library JSON file:
     ```
     <workspace_directory>/griptape-nodes-sam-3d-objects-library/griptape_nodes_sam_3d_objects_library/griptape-nodes-library.json
     ```
   - You can check your workspace directory with `gtn config show workspace_directory`
   - Close the Settings Panel
   - Click on *Refresh Libraries*

4. **Verify installation** by checking that the nodes appear in the node palette under the "3D Reconstruction" category.

## Usage

### Reconstruct Single Object 3D

1. Add a **Reconstruct Single Object 3D** node to your workflow.
2. Connect an image artifact to the `image` input (the RGB or RGBA image containing your object).
3. Connect a binary mask image artifact to the `mask` input (white pixels = object, black = background).
4. Set `config_path` to the absolute path of `pipeline.yaml` inside the downloaded checkpoints directory, or leave it at the default if using the expected directory structure.
5. Set `output_ply_path` to the file path where you want the PLY saved.
6. Run the node. The `ply_path` output will contain the path to the saved PLY file, which can be loaded in a 3D viewer.

### Reconstruct Multi-Object 3D

1. Add a **Reconstruct Multi-Object 3D** node to your workflow.
2. Connect an image artifact to the `image` input.
3. Set `masks` to a JSON array of image URL strings, one per object mask. For example:
   ```json
   ["/path/to/mask_0.png", "/path/to/mask_1.png", "/path/to/mask_2.png"]
   ```
4. Set `config_path` and `output_ply_path` as with the single-object node.
5. Run the node. All Gaussians are merged into a single scene PLY file.

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
