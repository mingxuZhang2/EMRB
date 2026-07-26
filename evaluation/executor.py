"""Isolated local Python code executor for EMRB evaluation."""
from contextlib import contextmanager
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from .config import CODE_TIMEOUT, MAX_OUTPUT_LEN

PREAMBLE = """\
import warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
"""


_SIGNAL_FILE_RE = re.compile(r"\b([A-Za-z0-9_.-]+\.npy)\b")


def extract_signal_filename(question_text):
    """Return the single signal filename named in an EMRB question."""
    filenames = list(dict.fromkeys(_SIGNAL_FILE_RE.findall(question_text)))
    if len(filenames) != 1:
        raise ValueError(
            f"Expected exactly one .npy filename in the question, found {filenames}"
        )
    filename = filenames[0]
    if Path(filename).name != filename:
        raise ValueError(f"Signal filename must not contain a path: {filename}")
    return filename


@contextmanager
def isolated_signal_workspace(question_text, sample_dir):
    """Create a per-problem workspace containing only the requested signal."""
    filename = extract_signal_filename(question_text)
    source_dir = Path(sample_dir).resolve()
    source = (source_dir / filename).resolve()
    if source.parent != source_dir or not source.is_file():
        raise FileNotFoundError(f"Signal file is unavailable: {source}")

    with tempfile.TemporaryDirectory(prefix=f"emrb-{source.stem}-") as tmp:
        workspace = Path(tmp)
        shutil.copy2(source, workspace / filename)

        # Reuse the host font index without exposing the rest of the home directory.
        mpl_cache = workspace / ".mplconfig"
        mpl_cache.mkdir()
        host_cache = Path.home() / ".cache" / "matplotlib"
        for font_index in host_cache.glob("fontlist-*.json"):
            shutil.copy2(font_index, mpl_cache / font_index.name)

        yield str(workspace)


def _directory_mount_args(path):
    """Create destination parents before a nested bubblewrap bind mount."""
    parents = list(Path(path).parents)
    args = []
    for parent in reversed(parents[:-1]):
        args.extend(("--dir", str(parent)))
    return args


def _sandbox_command(working_dir):
    bwrap = shutil.which("bwrap")
    if not bwrap:
        raise RuntimeError(
            "bubblewrap is required for benchmark code execution isolation"
        )

    command = [bwrap, "--die-with-parent", "--unshare-all", "--new-session"]
    for system_dir in ("/usr", "/bin", "/lib", "/lib64", "/etc"):
        if Path(system_dir).exists():
            command.extend(("--ro-bind", system_dir, system_dir))

    python_prefix = str(Path(sys.prefix).resolve())
    if not python_prefix.startswith("/usr"):
        command.extend(_directory_mount_args(python_prefix))
        command.extend(("--ro-bind", python_prefix, python_prefix))

    user_site = Path.home() / ".local" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    if user_site.is_dir():
        command.extend(_directory_mount_args(user_site))
        command.extend(("--ro-bind", str(user_site), str(user_site)))

    command.extend((
        "--proc", "/proc",
        "--dev", "/dev",
        "--tmpfs", "/tmp",
        "--bind", str(Path(working_dir).resolve()), "/workspace",
        "--chdir", "/workspace",
        "--setenv", "HOME", "/tmp",
        "--setenv", "MPLCONFIGDIR", "/workspace/.mplconfig",
        "--setenv", "PATH", f"{python_prefix}/bin:/usr/bin:/bin",
    ))
    if user_site.is_dir():
        command.extend(("--setenv", "PYTHONPATH", str(user_site)))
    command.extend((sys.executable, "-"))
    return command


def _sandbox_environment():
    """Keep credentials and host-specific Python paths out of executed code."""
    return {
        "PATH": "/usr/bin:/bin",
        "HOME": "/tmp",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }


def execute_python(code, working_dir, timeout=None):
    """Execute Python code in a subprocess, return stdout+stderr."""
    if timeout is None:
        timeout = CODE_TIMEOUT

    full_code = PREAMBLE + code
    try:
        result = subprocess.run(
            _sandbox_command(working_dir),
            input=full_code,
            capture_output=True, text=True,
            timeout=timeout, cwd=working_dir,
            env=_sandbox_environment(),
        )
        out = result.stdout
        if result.stderr:
            err_lines = result.stderr.strip().split('\n')
            err_short = '\n'.join(err_lines[-20:])
            out += f"\n[STDERR]:\n{err_short}"
        if len(out) > MAX_OUTPUT_LEN:
            out = out[:MAX_OUTPUT_LEN] + "\n...[truncated]"
        return out if out.strip() else "[No output]"
    except subprocess.TimeoutExpired:
        return f"[ERROR] Execution timed out after {timeout}s."
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"
