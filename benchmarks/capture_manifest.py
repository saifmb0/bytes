"""Capture the software/hardware provenance required by the paper artifact."""
import argparse
import json
import os
import platform
import subprocess
import glob
import hashlib

import torch


def version(name):
    try:
        module = __import__(name)
        return getattr(module, "__version__", "unknown")
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="results/paper/manifest.json")
    a = p.parse_args()
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    except Exception:
        revision = "unknown"
        dirty = "unknown"
    evidence = []
    for path in sorted(glob.glob(os.path.join(os.path.dirname(a.output), "*.json"))):
        if os.path.abspath(path) == os.path.abspath(a.output):
            continue
        item = json.load(open(path))
        metadata = item if isinstance(item, dict) else {}
        digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
        evidence.append({"file": os.path.basename(path), "model": metadata.get("model"),
                         "model_revision": metadata.get("model_revision"),
                         "dataset_fingerprint": metadata.get("dataset_fingerprint"),
                         "seed": metadata.get("seed"), "sha256": digest})
    data = {
        "schema_version": 1,
        "git_revision": revision,
        "git_dirty": bool(dirty.strip()),
        "git_status_sha256": hashlib.sha256(dirty.encode()).hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "transformers": version("transformers"),
        "datasets": version("datasets"),
        "triton": version("triton"),
        "evidence": evidence,
        "commands": ["bash reproduce.sh"],
    }
    os.makedirs(os.path.dirname(a.output) or ".", exist_ok=True)
    with open(a.output, "w") as f:
        json.dump(data, f, indent=2)
    print(f"saved -> {a.output}")


if __name__ == "__main__":
    main()
