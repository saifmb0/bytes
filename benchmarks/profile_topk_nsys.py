"""Capture the paper's top-k operation with Nsight Systems, without GPU counters.

The NVTX range in profile_microkernels.py excludes setup.  The report records
the library kernels actually launched; it does not infer stalls, bank conflicts,
or a particular bottleneck without hardware-counter permission.
"""
import argparse
import json
import os
import shutil
import subprocess
import tempfile


def find_nsys():
    explicit = os.environ.get("NSYS_BIN")
    candidates = [explicit, shutil.which("nsys")]
    prefix = os.environ.get("CONDA_PREFIX", "")
    candidates.append(os.path.join(prefix, "nsight-compute", "2024.1.1", "host", "target-linux-x64", "nsys"))
    candidates.append("/home/202311016/.conda/envs/saif/nsight-compute/2024.1.1/host/target-linux-x64/nsys")
    for path in candidates:
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    raise RuntimeError("Nsight Systems not found; set NSYS_BIN to its executable")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-json", default="results/paper/topk_nsys.json")
    p.add_argument("--lengths", default="4096,8192,16384")
    p.add_argument("--fracs", default="0.01,0.05,0.25")
    a = p.parse_args()
    nsys = find_nsys()
    rows = []
    with tempfile.TemporaryDirectory(prefix="topk_nsys_") as tmp:
        for S in [int(v) for v in a.lengths.split(",")]:
            for frac in [float(v) for v in a.fracs.split(",")]:
                k = int(round(S * frac))
                stem = os.path.join(tmp, f"topk_S{S}_k{k}")
                target = f"paper_target_topk_S{S}_k{k}"
                subprocess.run([nsys, "profile", "--trace=cuda,nvtx", "--sample=none",
                                "--force-overwrite=true", "-o", stem,
                                "python", "-m", "benchmarks.profile_microkernels", "topk",
                                "--S", str(S), "--frac", str(frac)], check=True)
                rep = stem + ".nsys-rep"
                report = subprocess.run([nsys, "stats", "--report", "cuda_gpu_kern_sum",
                                         "--filter-nvtx", target, rep], check=True,
                                        text=True, stdout=subprocess.PIPE).stdout
                # Keep the range-filtered report verbatim: it is the raw provenance
                # for named CUDA library kernels, robust to report-column changes.
                named = [line.strip() for line in report.splitlines()
                         if any(token in line for token in
                                ("mbtopk::", "radixSortKVInPlace", "scan_by_key", "gatherTopK"))]
                rows.append({"S": S, "frac": frac, "k": k,
                             "nvtx_range": target, "named_kernel_lines": named,
                             "raw_report": report})
                print(f"S={S} k={k}: {len(named)} named top-k report lines")
    os.makedirs(os.path.dirname(a.output_json), exist_ok=True)
    with open(a.output_json, "w") as f:
        json.dump({"schema": "topk_nsys_v1", "profiler": "Nsight Systems",
                   "counter_access": "not required", "rows": rows}, f, indent=2)


if __name__ == "__main__":
    main()
