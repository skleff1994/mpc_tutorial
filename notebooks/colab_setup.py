"""
Shared setup script for running mpc_tutorial notebooks in Google Colab.

Usage in each notebook:

    try:
        import google.colab  # noqa: F401
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False

    if IN_COLAB:
        !wget -q https://raw.githubusercontent.com/skleff1994/mpc_tutorial/master/notebooks/colab_setup.py -O /content/colab_setup.py
        %run /content/colab_setup.py
    else:
        %run colab_setup.py
"""

from pathlib import Path
import os
import shutil
import subprocess
import sys


REPO_URL = "https://github.com/skleff1994/mpc_tutorial.git"
REPO_BRANCH = "master"
REPO_NAME = "mpc_tutorial"

CONDA_PACKAGES = [
    "matplotlib",
    "numpy",
    "scipy",
    "osqp",
    "pybullet",
    "mim-solvers=0.2.0",
    "crocoddyl=3.2.0",
    "pinocchio=3.8.0",
    "pinocchio-python=3.8.0",
    "libpinocchio=3.8.0",
    "coal=3.0.1",
    "coal-python=3.0.1",
    "libcoal=3.0.1",
]


def running_in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def run(cmd, *, check=True):
    cmd = [str(c) for c in cmd]
    print("+ " + " ".join(cmd))
    return subprocess.run(cmd, check=check)


def ensure_conda_in_colab():
    """Install condacolab if conda is not available yet.

    condacolab.install() restarts the runtime. After the restart, the user
    should run the same notebook cell again.
    """
    if shutil.which("conda") is not None:
        return

    print("Installing condacolab. The Colab runtime will restart automatically.")
    run([sys.executable, "-m", "pip", "install", "-q", "condacolab"])

    import condacolab
    condacolab.install()

    print("Runtime restart requested. After the restart, run this cell again.")
    raise SystemExit


def clone_or_update_repo_in_colab() -> Path:
    repo_root = Path("/content") / REPO_NAME

    if repo_root.exists():
        print(f"Repository already exists: {repo_root}")
        run(["git", "-C", repo_root, "pull", "--ff-only"], check=False)
    else:
        run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                REPO_BRANCH,
                REPO_URL,
                repo_root,
            ]
        )

    return repo_root


def find_local_repo_root() -> Path:
    cwd = Path.cwd().resolve()

    repo_root = next(
        (
            p for p in [cwd, *cwd.parents]
            if (p / "notebooks").is_dir()
            and (p / "README.md").is_file()
        ),
        None,
    )

    if repo_root is None:
        raise RuntimeError(
            "Could not find the mpc_tutorial repository root. "
            "Please run this notebook from inside the repository."
        )

    return repo_root


def ocp_stack_imports_ok() -> bool:
    try:
        import numpy  # noqa: F401
        import pinocchio  # noqa: F401
        import crocoddyl  # noqa: F401
        import mim_solvers  # noqa: F401
        return True
    except Exception as exc:
        print(f"OCP stack import check failed: {type(exc).__name__}: {exc}")
        return False


def install_colab_dependencies(repo_root: Path):
    if not ocp_stack_imports_ok():
        run(
            [
                "conda",
                "install",
                "-y",
                "-c",
                "conda-forge",
                "--override-channels",
            ]
            + CONDA_PACKAGES
        )
    else:
        print("OCP stack already available: skipping conda install.")

    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "-e",
            str(repo_root),
            "--no-deps",
        ]
    )

    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "git+https://github.com/machines-in-motion/mim_robots.git",
            "--no-deps",
        ]
    )


IN_COLAB = running_in_colab()

if IN_COLAB:
    ensure_conda_in_colab()
    repo_root = clone_or_update_repo_in_colab()
    install_colab_dependencies(repo_root)
else:
    repo_root = find_local_repo_root()

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

notebooks_dir = repo_root / "notebooks"
os.chdir(notebooks_dir)

import numpy as np  # noqa: E402
import pinocchio as pin  # noqa: E402
import crocoddyl  # noqa: E402
import mim_solvers  # noqa: E402

print("")
print("✅ mpc_tutorial setup complete")
print(f"Repository root: {repo_root}")
print(f"Working directory: {Path.cwd()}")
print(f"Pinocchio: {pin.__version__}")
print(f"Crocoddyl: {crocoddyl.__version__}")