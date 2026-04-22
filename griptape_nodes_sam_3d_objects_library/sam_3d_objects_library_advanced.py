import logging
import os
import subprocess
import sys
from pathlib import Path

import pygit2
from griptape_nodes.node_library.advanced_node_library import AdvancedNodeLibrary
from griptape_nodes.node_library.library_registry import Library, LibrarySchema

logger = logging.getLogger("sam_3d_objects_library")


class Sam3DObjectsLibraryAdvanced(AdvancedNodeLibrary):
    def before_library_nodes_loaded(self, library_data: LibrarySchema, library: Library) -> None:
        logger.info(f"Loading '{library_data.name}' library...")
        submodule_path = self._init_submodule()
        if not self._is_installed(submodule_path):
            self._install_from_requirements(submodule_path)
            self._apply_patches(submodule_path)
            self._write_installed_sentinel(submodule_path)
        else:
            logger.info("Dependencies already installed, skipping install steps")
        # sys.path injection must happen every engine start
        self._install_package(submodule_path)

    def after_library_nodes_loaded(self, library_data: LibrarySchema, library: Library) -> None:
        logger.info(f"Finished loading '{library_data.name}' library")

    def _get_library_root(self) -> Path:
        return Path(__file__).parent

    def _get_venv_python_path(self) -> Path:
        root = self._get_library_root()
        if sys.platform == "win32":
            return root / ".venv" / "Scripts" / "python.exe"
        return root / ".venv" / "bin" / "python"

    def _update_submodules_recursive(self, repo_path: Path) -> None:
        repo = pygit2.Repository(str(repo_path))
        repo.submodules.update(init=True)
        for sub in repo.submodules:
            sub_path = repo_path / sub.path
            if sub_path.exists() and (sub_path / ".git").exists():
                self._update_submodules_recursive(sub_path)

    def _init_submodule(self) -> Path:
        library_root = self._get_library_root()
        submodule_dir = library_root / "sam-3d-objects"
        if submodule_dir.exists() and any(submodule_dir.iterdir()):
            logger.info("Submodule already initialized")
            return submodule_dir
        self._update_submodules_recursive(library_root.parent)
        if not submodule_dir.exists() or not any(submodule_dir.iterdir()):
            raise RuntimeError(f"Submodule init failed: {submodule_dir}")
        logger.info("Submodule initialized successfully")
        return submodule_dir

    def _ensure_pip(self) -> None:
        venv_python = self._get_venv_python_path()
        result = subprocess.run([str(venv_python), "-m", "pip", "--version"], capture_output=True)
        if result.returncode == 0:
            return
        subprocess.check_call([str(venv_python), "-m", "ensurepip", "--upgrade"])

    def _get_torch_version(self) -> str:
        """Return the base torch version (e.g. '2.5.1') installed in the venv."""
        venv_python = self._get_venv_python_path()
        result = subprocess.run(
            [str(venv_python), "-c", "import torch; print(torch.__version__.split('+')[0])"],
            capture_output=True, text=True,
        )
        return result.stdout.strip()

    def _get_cuda_home(self) -> str:
        """Return a usable CUDA toolkit path with nvcc for building extensions.

        Searches for a system CUDA toolkit (>= 12.x) since the pip nvidia-cuda-nvcc
        package doesn't include a full nvcc binary.
        """
        # Prefer CUDA_HOME if already set
        env_home = os.environ.get("CUDA_HOME")
        if env_home and Path(env_home, "bin", "nvcc").exists():
            return env_home
        # Search common system locations
        for candidate in sorted(Path("/usr/local").glob("cuda-12.*"), reverse=True):
            if (candidate / "bin" / "nvcc").exists():
                return str(candidate)
        if Path("/usr/local/cuda/bin/nvcc").exists():
            return "/usr/local/cuda"
        raise RuntimeError(
            "No usable CUDA toolkit found. Install the CUDA toolkit (>=12.x) "
            "or set CUDA_HOME to a directory containing bin/nvcc."
        )

    def _get_submodule_commit(self, submodule_path: Path) -> str:
        """Return the HEAD commit SHA of the submodule (the version pinned by the library author)."""
        repo = pygit2.Repository(str(submodule_path))
        return str(repo.head.target)

    def _get_installed_sentinel(self) -> Path:
        return self._get_library_root() / ".installed_commit"

    def _write_installed_sentinel(self, submodule_path: Path) -> None:
        self._get_installed_sentinel().write_text(self._get_submodule_commit(submodule_path))

    def _is_installed(self, submodule_path: Path) -> bool:
        """Return True only if deps are installed AND match the currently-pinned submodule commit.

        This ensures that when a new library version ships with a different submodule commit,
        the package is reinstalled rather than reusing a stale installation.
        """
        sentinel = self._get_installed_sentinel()
        if not sentinel.exists():
            return False
        if sentinel.read_text().strip() != self._get_submodule_commit(submodule_path):
            return False
        # Verify a key dependency is actually importable in the venv
        venv_python = self._get_venv_python_path()
        result = subprocess.run(
            [str(venv_python), "-c", "import kaolin"],
            capture_output=True,
        )
        return result.returncode == 0

    def _install_from_requirements(self, submodule_path: Path) -> None:
        """Install all inference dependencies required by sam-3d-objects.

        The upstream requirements.inference.txt is incomplete — many transitive
        dependencies needed at runtime are only listed in the full requirements.txt
        (which contains many unrelated dev packages).  Additionally, several packages
        require special install procedures:
          - kaolin: must come from NVIDIA's S3 wheels matched to the torch+cuda version.
          - gsplat / pytorch3d: must be built from source with --no-build-isolation
            and CUDA_HOME pointing at a system CUDA toolkit.
          - MoGe / utils3d: must be installed from specific git commits.
          - spconv: needs the -cu121 variant.
          - numpy must stay <2.0 (kaolin constraint).

        This method therefore ignores requirements.inference.txt and performs a
        deterministic, ordered install sequence.
        """
        venv_python = self._get_venv_python_path()
        self._ensure_pip()

        torch_ver = self._get_torch_version()
        cuda_home = self._get_cuda_home()
        kaolin_find_links = f"https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-{torch_ver}_cu121.html"

        env = os.environ.copy()
        env["CUDA_HOME"] = cuda_home
        env["FORCE_CUDA"] = "1"
        env["TORCH_CUDA_ARCH_LIST"] = "8.6"
        logger.info(f"Using CUDA_HOME={cuda_home}")

        def _pip(args: list[str], **kwargs: object) -> None:
            subprocess.check_call([str(venv_python), "-m", "pip", "install", *args], env=env, **kwargs)

        # --- Step 1: kaolin from NVIDIA S3 (wheel only, no deps) ---
        logger.info(f"Installing kaolin from {kaolin_find_links}...")
        _pip(["--no-index", "--no-deps", "-f", kaolin_find_links, "kaolin==0.17.0"])

        # --- Step 2: Pure-Python / wheel runtime deps ---
        # These are packages the sam3d_objects code imports at runtime that aren't
        # pulled in by requirements.inference.txt.
        logger.info("Installing runtime dependencies...")
        _pip([
            "numpy<2.0",
            "loguru",
            "astor",
            "opencv-python",
            "easydict",
            "python-igraph",
            "imageio",
            "imageio-ffmpeg",
            "lightning",
            "omegaconf",
            "open3d",
            "optree",
            "plotly",
            "plyfile",
            "pymeshfix",
            "pyvista",
            "safetensors",
            "seaborn==0.13.2",
            "timm",
            "trimesh",
            "xatlas",
            "spconv-cu121==2.3.8",
            # kaolin dependencies (installed with --no-deps so we must provide these)
            "ipycanvas",
            "ipyevents",
            "jupyter-client<8",
            "pygltflib",
            "tornado",
            "usd-core",
            "warp-lang",
            # gradio (used by inference.py)
            "gradio==5.49.0",
            # huggingface_hub (used for checkpoint downloads)
            "huggingface_hub",
        ])
        # Pin numpy back down (open3d/opencv may have upgraded it)
        _pip(["numpy<2.0"])

        # --- Step 3: Packages from git that need specific commits ---
        logger.info("Installing MoGe...")
        _pip([
            "--no-deps", "--no-build-isolation",
            "MoGe @ git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b",
        ])

        logger.info("Installing utils3d (MoGe-pinned commit)...")
        _pip([
            "--force-reinstall", "--no-deps",
            "utils3d @ git+https://github.com/EasternJournalist/utils3d.git@3913c65d81e05e47b9f367250cf8c0f7462a0900",
        ])

        # --- Step 4: CUDA extensions built from source ---
        logger.info("Building gsplat from source...")
        _pip([
            "--no-build-isolation", "--no-deps",
            "gsplat @ git+https://github.com/nerfstudio-project/gsplat.git@2323de5905d5e90e035f792fe65bad0fedd413e7",
        ])

        logger.info("Building pytorch3d from source...")
        _pip([
            "--no-build-isolation", "--no-deps",
            "git+https://github.com/facebookresearch/pytorch3d.git",
        ])

        logger.info("Inference dependencies installed successfully")

    def _install_package(self, submodule_path: Path) -> None:
        if str(submodule_path) not in sys.path:
            sys.path.insert(0, str(submodule_path))
        notebook_path = submodule_path / "notebook"
        if str(notebook_path) not in sys.path:
            sys.path.insert(0, str(notebook_path))
        logger.info(f"Added {submodule_path} and {notebook_path} to sys.path")

    def _apply_patches(self, submodule_path: Path) -> None:
        """Apply the hydra patch required for correct Hydra operation.

        The patch at ./patching/hydra fixes https://github.com/facebookresearch/hydra/pull/2863.
        """
        patch_script = submodule_path / "patching" / "hydra"
        if not patch_script.exists():
            logger.warning(f"Hydra patch script not found at {patch_script}, skipping")
            return
        venv_python = self._get_venv_python_path()
        # hydra-core is required by the patch script but not in requirements.inference.txt
        logger.info("Installing hydra-core for patching...")
        subprocess.check_call([str(venv_python), "-m", "pip", "install", "hydra-core==1.3.2"])
        logger.info("Applying hydra patch...")
        subprocess.check_call([str(venv_python), str(patch_script)], cwd=str(submodule_path))
        logger.info("Hydra patch applied successfully")
