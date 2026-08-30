# Hardware-conscious KV-cache characterization study

This repository contains the correctness-gated evaluation harness and paper for a
hardware-conscious characterization of KV-cache selection and compression. Matched
conditions are the method: they isolate how representation, access pattern, score
computation, and top-$k$ determine quality, memory traffic, and decode latency.
Quality experiments use real language models; decode-attention latency uses disclosed
synthetic tensor shapes on an RTX 4000 Ada.

## Full reproduction

Run the complete publication pipeline with:

```bash
bash reproduce.sh
```

It resumes valid completed cells. To discard and recreate only the tracked
publication evidence, use `REPRO_CLEAN=1 bash reproduce.sh`.

The runner attempts Qwen2.5-7B BF16 at 16K and uses the pre-registered 8K fallback
only if 16K fails. It also produces the 100-case hard NIAH panel. Model and dataset
downloads require Hugging Face access. Set `KIVI_REPO` to an official KIVI
checkout when regenerating KIVI latency results.

Compact, publication-used JSON belongs in `results/paper/` and is tracked. Temporary
and exploratory results under `results/` remain ignored. `validate_evidence` rejects
aggregate-only publication files that omit their per-window or per-trial observations.

## Tests

```bash
python -m benchmarks.test_sparse_eval
python -m benchmarks.test_correctness
python -m benchmarks.test_statistics
```

The first suite validates chunked selection against a dense reference and checks
sub-quadratic allocation growth. The second validates the Triton attention kernels.

## Evidence policy

The paper is a hardware-conscious characterization of KV-cache decode cost, not a new
named algorithm or a bits-per-token leaderboard. Its controlled comparisons establish
when nominal cache-byte savings do and do not translate to latency savings. Publication
evidence is the tracked per-observation JSON in `results/paper/`; tables and figures
are generated from that evidence. The validator rejects incomplete publication files,
and `reproduce.sh` runs the tests, experiments, aggregation, validation, provenance
capture, and paper build in order.
