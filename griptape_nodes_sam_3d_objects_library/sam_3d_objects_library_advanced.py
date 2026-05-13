import json
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

    def _init_submodules_from_gitmodules(self, gitmodules_path: Path) -> None:
        """Run git submodule update --init --recursive from the repo root."""
        repo_root = gitmodules_path.parent
        logger.info(f"Running git submodule update --init --recursive from {repo_root}")
        subprocess.run(
            ["git", "-C", str(repo_root), "submodule", "update", "--init", "--recursive"],
            check=True,
            capture_output=True,
            text=True,
        )

    def _clone_submodule_from_library_json(self, submodule_dir: Path) -> None:
        """Clone submodule directly using metadata in griptape-nodes-library.json."""
        json_path = self._get_library_root() / "griptape-nodes-library.json"
        with json_path.open(encoding="utf-8") as f:
            data = json.load(f)

        submodule_info = data["metadata"]["submodule_info"]
        url = submodule_info["url"]
        commit = submodule_info.get("commit")

        logger.info(f"Cloning SAM 3D Objects submodule from {url} (commit: {commit})")
        submodule_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", url, str(submodule_dir)], check=True, capture_output=True, text=True)
        if commit:
            subprocess.run(
                ["git", "-C", str(submodule_dir), "checkout", commit],
                check=True,
                capture_output=True,
                text=True,
            )

    def _init_submodule(self) -> Path:
        library_root = self._get_library_root()
        submodule_dir = library_root / "sam-3d-objects"
        if submodule_dir.exists() and any(submodule_dir.iterdir()):
            logger.info("Submodule already initialized")
            return submodule_dir

        # Walk up to find .gitmodules (same strategy as the SAM3 library).
        current = library_root.resolve()
        while current != current.parent:
            gitmodules_path = current / ".gitmodules"
            if gitmodules_path.exists():
                logger.info(f"Found .gitmodules at {gitmodules_path}")
                self._init_submodules_from_gitmodules(gitmodules_path)
                break
            current = current.parent
        else:
            logger.info("No .gitmodules found, falling back to griptape-nodes-library.json")
            self._clone_submodule_from_library_json(submodule_dir)

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
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def _get_torch_cuda_version(self) -> str | None:
        """Return the major.minor CUDA version torch was compiled against (e.g. '12.1'), or None."""
        venv_python = self._get_venv_python_path()
        result = subprocess.run(
            [str(venv_python), "-c", "import torch; print(torch.version.cuda)"],
            capture_output=True,
            text=True,
        )
        v = result.stdout.strip()
        if not v or v == "None":
            return None
        parts = v.split(".")
        return f"{parts[0]}.{parts[1]}" if len(parts) >= 2 else v

    def _toolkit_version(self, cuda_home: str) -> str | None:
        """Return the major.minor version of a CUDA toolkit by reading its version.json."""
        import json as _json

        version_json = Path(cuda_home, "version.json")
        if not version_json.exists():
            return None
        try:
            data = _json.loads(version_json.read_text())
            v = data.get("cuda", {}).get("version", "")
            parts = v.split(".")
            if len(parts) >= 2:
                return f"{parts[0]}.{parts[1]}"
        except Exception:
            pass
        return None

    def _get_gpu_arch_list(self) -> str:
        """Return TORCH_CUDA_ARCH_LIST string based on the GPUs present in the venv.

        Queries all available CUDA devices and builds a semicolon-separated list of
        'major.minor' compute capabilities, with '+PTX' appended to the highest entry
        so kernels can JIT-forward-compile for future GPU generations.
        """
        venv_python = self._get_venv_python_path()
        script = (
            "import torch; "
            "avail = torch.cuda.is_available(); "
            "caps = sorted(set(torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count()))) if avail else []; "
            "print(avail, *[f'{m}.{n}' for m, n in caps])"
        )
        result = subprocess.run(
            [str(venv_python), "-c", script],
            capture_output=True,
            text=True,
        )
        parts = result.stdout.strip().split()
        if not parts or parts[0] != "True":
            raise RuntimeError(
                "No CUDA-capable GPU detected. The SAM 3D Objects library requires a CUDA GPU. "
                "Ensure your GPU drivers are installed and torch.cuda.is_available() returns True."
            )
        caps = parts[1:]
        if not caps:
            raise RuntimeError("No GPU compute capabilities detected. Cannot determine TORCH_CUDA_ARCH_LIST.")
        caps[-1] = caps[-1] + "+PTX"
        return ";".join(caps)

    def _get_cuda_home(self) -> str:
        """Return a usable CUDA toolkit path whose version matches torch's CUDA build.

        Checks CUDA_HOME / CUDA_PATH env vars and nvcc on PATH. When the found toolkit
        version doesn't match what torch was compiled against, raises a clear error
        rather than letting the build fail deep inside setuptools.
        """
        import shutil

        nvcc_exe = "nvcc.exe" if sys.platform == "win32" else "nvcc"
        torch_cuda = self._get_torch_cuda_version()

        def _check_and_return(candidate: str) -> str:
            toolkit_ver = self._toolkit_version(candidate)
            if torch_cuda and toolkit_ver and not toolkit_ver.startswith(torch_cuda):
                raise RuntimeError(
                    f"CUDA version mismatch: the toolkit at '{candidate}' is CUDA {toolkit_ver}, "
                    f"but PyTorch in this library's venv was compiled against CUDA {torch_cuda}. "
                    f"Install CUDA {torch_cuda} alongside your existing toolkit and set "
                    f"CUDA_HOME / CUDA_PATH to that directory before loading this library."
                )
            return candidate

        # Prefer explicit env vars (CUDA_HOME on Linux/Mac, CUDA_PATH on Windows)
        for var in ("CUDA_HOME", "CUDA_PATH"):
            candidate = os.environ.get(var)
            if candidate and Path(candidate, "bin", nvcc_exe).exists():
                return _check_and_return(candidate)

        # Fall back to locating nvcc on PATH
        nvcc = shutil.which(nvcc_exe) or shutil.which("nvcc")
        if nvcc:
            return _check_and_return(str(Path(nvcc).parent.parent))

        torch_hint = f"CUDA {torch_cuda} to match PyTorch, " if torch_cuda else ""
        raise RuntimeError(
            f"No usable CUDA toolkit found. Install the CUDA toolkit "
            f"({torch_hint}>=12.x) and set CUDA_HOME / CUDA_PATH to a directory "
            f"containing bin/nvcc."
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

        cuda_home = self._get_cuda_home()
        arch_list = self._get_gpu_arch_list()

        env = os.environ.copy()
        env["CUDA_HOME"] = cuda_home
        env["FORCE_CUDA"] = "1"
        env["TORCH_CUDA_ARCH_LIST"] = arch_list
        logger.info(f"Using CUDA_HOME={cuda_home}, TORCH_CUDA_ARCH_LIST={arch_list}")

        def _pip(args: list[str]) -> None:
            subprocess.check_call([str(venv_python), "-m", "pip", "install", *args], env=env)

        # --- Step 0: CUDA-enabled PyTorch (must come before everything else) ---
        # CPU-only torch wheels omit the C++ extension headers (ATen/TensorUtils.h etc.)
        # that gsplat and pytorch3d need at compile time. Install from the PyTorch CUDA 12.1
        # index so the venv has a full CUDA build before any source-compiled extension runs.
        logger.info("Installing PyTorch with CUDA 12.1 support...")
        _pip(
            [
                "--index-url",
                "https://download.pytorch.org/whl/cu121",
                "torch",
                "torchvision",
                "torchaudio",
            ]
        )

        torch_ver = self._get_torch_version()
        kaolin_find_links = f"https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-{torch_ver}_cu121.html"

        # --- Step 1: kaolin from NVIDIA S3 (wheel only, no deps) ---
        logger.info(f"Installing kaolin from {kaolin_find_links}...")
        _pip(["--no-index", "--no-deps", "-f", kaolin_find_links, "kaolin==0.17.0"])

        # --- Step 2: Pure-Python / wheel runtime deps ---
        # These are packages the sam3d_objects code imports at runtime that aren't
        # pulled in by requirements.inference.txt.
        logger.info("Installing runtime dependencies...")
        _pip(
            [
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
            ]
        )
        # Pin numpy back down (open3d/opencv may have upgraded it)
        _pip(["numpy<2.0"])

        # --- Step 3: Packages from git that need specific commits ---
        logger.info("Installing MoGe...")
        _pip(
            [
                "--no-deps",
                "--no-build-isolation",
                "MoGe @ git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b",
            ]
        )

        logger.info("Installing utils3d (MoGe-pinned commit)...")
        _pip(
            [
                "--force-reinstall",
                "--no-deps",
                "utils3d @ git+https://github.com/EasternJournalist/utils3d.git@3913c65d81e05e47b9f367250cf8c0f7462a0900",
            ]
        )

        # --- Step 4: CUDA extensions built from source ---
        logger.info("Building gsplat from source...")
        _pip(
            [
                "--no-build-isolation",
                "--no-deps",
                "gsplat @ git+https://github.com/nerfstudio-project/gsplat.git@2323de5905d5e90e035f792fe65bad0fedd413e7",
            ]
        )

        logger.info("Building pytorch3d from source...")
        _pip(
            [
                "--no-build-isolation",
                "--no-deps",
                "git+https://github.com/facebookresearch/pytorch3d.git",
            ]
        )

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
