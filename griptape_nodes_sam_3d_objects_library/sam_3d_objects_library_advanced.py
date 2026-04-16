import logging
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
            self._install_package(submodule_path)
            self._apply_patches(submodule_path)
            self._write_installed_sentinel(submodule_path)

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

    def _get_submodule_commit(self, submodule_path: Path) -> str:
        """Return the HEAD commit SHA of the submodule (the version pinned by the library author)."""
        repo = pygit2.Repository(str(submodule_path))
        return str(repo.head.target)

    def _get_installed_sentinel(self) -> Path:
        return self._get_library_root() / ".installed_commit"

    def _write_installed_sentinel(self, submodule_path: Path) -> None:
        self._get_installed_sentinel().write_text(self._get_submodule_commit(submodule_path))

    def _is_installed(self, submodule_path: Path) -> bool:
        """Return True only if the package is importable AND was installed from the currently-pinned commit.

        This ensures that when a new library version ships with a different submodule commit,
        the package is reinstalled rather than reusing a stale installation.
        """
        venv_python = self._get_venv_python_path()
        result = subprocess.run(
            [str(venv_python), "-c", "import sam3d_objects"],
            capture_output=True,
        )
        if result.returncode != 0:
            return False
        sentinel = self._get_installed_sentinel()
        if not sentinel.exists():
            return False
        return sentinel.read_text().strip() == self._get_submodule_commit(submodule_path)

    def _install_from_requirements(self, submodule_path: Path) -> None:
        """Install only the inference dependencies from requirements.inference.txt.

        The full requirements.txt is a bloated dev environment dump with many
        packages that don't work on Windows. We only need the minimal inference deps.
        """
        venv_python = self._get_venv_python_path()
        self._ensure_pip()

        # Install only from requirements.inference.txt (minimal deps for inference)
        # Use --no-build-isolation so packages that need torch at build time can find it
        inference_reqs = submodule_path / "requirements.inference.txt"
        if inference_reqs.exists():
            logger.info(f"Installing inference dependencies from {inference_reqs}...")
            subprocess.check_call(
                [str(venv_python), "-m", "pip", "install", "--no-build-isolation", "-r", str(inference_reqs)]
            )
            logger.info("Inference dependencies installed successfully")
        else:
            logger.warning(f"No requirements.inference.txt found at {inference_reqs}")

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
        logger.info("Applying hydra patch...")
        subprocess.check_call([str(patch_script)], cwd=str(submodule_path))
        logger.info("Hydra patch applied successfully")
