#!/usr/bin/env python3
"""Install optional pinned DeepFilterNet dependencies into a private environment.

No changes to system Python. Package and model downloads are free and local;
lecture media is never transmitted. All failures produce a machine-readable
state and leave the FFmpeg baseline usable.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile

MODEL_URL = "https://raw.githubusercontent.com/Rikorose/DeepFilterNet/v0.5.6/models/DeepFilterNet3.zip"
MODEL_SHA256 = "49c52edc8947ae1f9bf50d81530beaf3a2c3245aeaf34b6f31ff535cd22284d2"
CODE_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd(),
                        help="Workspace holding the private environment, models and reports")
    parser.add_argument("--python", help="Existing Python 3.11 executable; otherwise uv may download Python 3.11")
    parser.add_argument("--check-only", action="store_true", help="Verify an existing environment/model without downloading")
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    venv = root / ".venv-neural"
    python = venv / "bin" / "python"
    state_path = root / "reports" / "neural_setup.json"
    log_path = root / "logs" / "neural_setup.log"
    model_cache = root / ".models" / "deepfilternet"
    model_dir = model_cache / "DeepFilterNet3"
    started = time.monotonic()
    state = {"ready": False, "status": "starting", "environment": str(venv),
             "python": str(python), "model_cache": str(model_cache), "model": "DeepFilterNet3",
             "created_utc": datetime.now(timezone.utc).isoformat(),
             "architecture": platform.machine(), "platform": platform.platform(),
             "log": str(log_path), "device": "cpu", "media_uploaded": False,
             "sources": ["https://github.com/Rikorose/DeepFilterNet/tree/v0.5.6",
                         "https://pytorch.org/get-started/previous-versions/"],
             "packages": {}}
    atomic_json(state_path, state)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        def run(command, timeout=600):
            log.write("\n" + json.dumps(command) + "\n")
            log.flush()
            result = subprocess.run(command, cwd=root, text=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, timeout=timeout)
            log.write(result.stdout)
            log.flush()
            if result.returncode:
                raise RuntimeError(f"Command failed ({result.returncode}): {command[0]}; see {log_path}\n{result.stdout[-2000:]}")
            return result.stdout
        try:
            if platform.system() != "Darwin" or platform.machine() not in {"arm64", "x86_64"}:
                raise RuntimeError("This pinned environment supports native arm64 and x86_64 macOS")
            uv = shutil.which("uv")
            requirements = CODE_ROOT / "requirements-neural-lock.txt"
            if not requirements.is_file():
                requirements = CODE_ROOT / "requirements-neural.txt"
            if not args.check_only:
                if not python.is_file():
                    if uv:
                        run([uv, "venv", "--python", args.python or "3.11", str(venv)])
                    else:
                        selected = args.python or shutil.which("python3.11")
                        if not selected:
                            raise RuntimeError("Python 3.11 or uv is required; baseline does not need either")
                        run([selected, "-m", "venv", str(venv)])
                version = run([str(python), "-c", "import sys; print('.'.join(map(str, sys.version_info[:3])))"]).strip()
                if not version.startswith("3.11."):
                    raise RuntimeError(f"Existing private environment uses Python {version}; use a fresh Python 3.11 environment")
                if uv:
                    command = [uv, "pip", "sync" if requirements.name.endswith("-lock.txt") else "install",
                               "--python", str(python), "--only-binary", ":all:"]
                    if requirements.name.endswith("-lock.txt"):
                        command.extend(["--require-hashes", str(requirements)])
                    else:
                        command.extend(["-r", str(requirements)])
                    run(command)
                else:
                    run([str(python), "-m", "pip", "install", "--only-binary", ":all:", "-r", str(requirements)])
                model_cache.mkdir(parents=True, exist_ok=True)
                archive = model_cache / "DeepFilterNet3-v0.5.6.zip"
                if not archive.is_file() or sha256(archive) != MODEL_SHA256:
                    with tempfile.NamedTemporaryFile(prefix=".download-", dir=model_cache, delete=False) as handle:
                        temporary_archive = Path(handle.name)
                        try:
                            with urllib.request.urlopen(MODEL_URL, timeout=60) as response:
                                while block := response.read(1024 * 1024):
                                    handle.write(block)
                            handle.flush()
                            if sha256(temporary_archive) != MODEL_SHA256:
                                raise RuntimeError("Model archive checksum failed")
                            os.replace(temporary_archive, archive)
                        finally:
                            temporary_archive.unlink(missing_ok=True)
                # Extract validated expected members only. Model cache can be
                # rebuilt after a partial prior setup; no user media is touched.
                with zipfile.ZipFile(archive) as zipped:
                    allowed = {"DeepFilterNet3/config.ini", "DeepFilterNet3/checkpoints/model_120.ckpt.best"}
                    if not allowed.issubset(zipped.namelist()):
                        raise RuntimeError("Model archive structure changed")
                    for member in sorted(allowed):
                        target = model_cache / member
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with zipped.open(member) as source, target.open("wb") as destination:
                            shutil.copyfileobj(source, destination)
                manifest = {"model": "DeepFilterNet3", "release": "v0.5.6", "source_url": MODEL_URL,
                            "archive_sha256": MODEL_SHA256,
                            "file_sha256": {str(p.relative_to(model_dir)): sha256(p)
                                            for p in sorted(model_dir.rglob("*")) if p.is_file()},
                            "post_filter": False, "device": "cpu"}
                atomic_json(model_cache / "manifest.json", manifest)
            if not python.is_file():
                raise RuntimeError("Private Python environment is missing")
            payload = run([str(python), "-c", "import importlib.metadata as m,json,sys,platform,torch,torchaudio; from df.enhance import enhance; print(json.dumps({'packages':{p:m.version(p) for p in ('torch','torchaudio','numpy','deepfilternet','deepfilterlib','soundfile')},'python_version':sys.version,'machine':platform.machine(),'cuda_available':torch.cuda.is_available(),'torch_mps_available':torch.backends.mps.is_available()}))"])
            state.update(json.loads(payload.splitlines()[-1]))
            if not state["python_version"].startswith("3.11."):
                raise RuntimeError("Private environment must use Python 3.11")
            state["frozen_packages"] = run([str(python), "-c", "import importlib.metadata as m,json;print(json.dumps(sorted(p.metadata['Name']+'=='+p.version for p in m.distributions())))"])
            state["frozen_packages"] = json.loads(state["frozen_packages"].splitlines()[-1])
            manifest = json.loads((model_cache / "manifest.json").read_text())
            for name, expected in manifest["file_sha256"].items():
                if sha256(model_dir / name) != expected:
                    raise RuntimeError(f"Cached model checksum failed: {name}")
            state["model_manifest"] = manifest
            with tempfile.TemporaryDirectory(prefix=".neural-smoke-", dir=root) as work:
                work = Path(work)
                run([str(python), "-c", "import numpy as np,soundfile as sf,sys; x=np.zeros((144017,2),dtype=np.float32);x[5000,0]=0.1;x[100001,1]=0.1;sf.write(sys.argv[1],x,48000,subtype='FLOAT')", str(work / "input.wav")])
                original_digest = sha256(work / "input.wav")
                run([str(python), str(CODE_ROOT / "lecture_restore" / "neural.py"), "--worker",
                     "--input", str(work / "input.wav"), "--output", str(work / "output.wav"),
                     "--model", str(model_dir), "--metrics", str(work / "metrics.json"),
                     "--attenuation", "12", "--chunk-seconds", "2"], timeout=180)
                smoke = json.loads((work / "metrics.json").read_text())
                state["smoke_test"] = {key: smoke[key] for key in
                                       ("input_frames", "output_frames", "channels", "exact_sample_count", "delay_compensated", "chunks", "sample_peak", "elapsed_seconds")}
                peak_positions = json.loads(run([str(python), "-c", "import numpy as np,soundfile as sf,sys,json; x,sr=sf.read(sys.argv[1],always_2d=True); print(json.dumps(np.argmax(np.abs(x),axis=0).tolist()))", str(work / "output.wav")]).splitlines()[-1])
                state["smoke_test"]["impulse_input_peak_samples"] = [5000, 100001]
                state["smoke_test"]["impulse_output_peak_samples"] = peak_positions
                state["smoke_test"]["source_hash_unchanged"] = sha256(work / "input.wav") == original_digest
                if peak_positions != [5000, 100001] or not state["smoke_test"]["source_hash_unchanged"]:
                    raise RuntimeError("Synthetic stereo impulse timing/source integrity check failed")
            state.update(ready=True, status="ready")
        except (Exception, SystemExit) as exc:
            state.update(ready=False, status="failed", error=str(exc),
                         recovery="FFmpeg baseline remains available. Inspect the setup log and rerun setup; never install into global Python.")
        finally:
            state["elapsed_seconds"] = time.monotonic() - started
            atomic_json(state_path, state)
    print(json.dumps(state, indent=2))
    return 0 if state["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
