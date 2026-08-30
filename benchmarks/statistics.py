"""Small deterministic statistics helpers used by paper result emitters."""
import math
import random


def bootstrap_geomean_ci(losses, confidence=0.95, n_resamples=10_000, seed=0):
    """Paired-window bootstrap CI for perplexity from per-window mean NLL values."""
    if not losses:
        return [float("nan"), float("nan")]
    if len(losses) == 1:
        value = math.exp(losses[0])
        return [value, value]
    rng = random.Random(seed)
    n = len(losses)
    samples = []
    for _ in range(n_resamples):
        samples.append(math.exp(sum(losses[rng.randrange(n)] for _ in range(n)) / n))
    samples.sort()
    alpha = (1.0 - confidence) / 2.0
    lo = samples[max(0, int(alpha * n_resamples))]
    hi = samples[min(n_resamples - 1, int((1.0 - alpha) * n_resamples) - 1)]
    return [lo, hi]


def wilson_interval(successes, total, confidence=0.95):
    """Wilson score interval. The paper protocol fixes confidence at 95%."""
    if total <= 0:
        return [float("nan"), float("nan")]
    if confidence != 0.95:
        raise ValueError("only the pre-registered 95% Wilson interval is supported")
    z = 1.959963984540054
    p = successes / total
    den = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / den
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return [max(0.0, center - half), min(1.0, center + half)]


def paired_accuracy_difference_ci(a, b, n_resamples=10_000, seed=0):
    """Paired bootstrap CI for mean(a-b), where inputs are binary trial outcomes."""
    if len(a) != len(b) or not a:
        raise ValueError("paired non-empty outcome arrays are required")
    rng = random.Random(seed)
    n = len(a)
    samples = []
    for _ in range(n_resamples):
        samples.append(sum(a[i] - b[i] for i in (rng.randrange(n) for _ in range(n))) / n)
    samples.sort()
    return [samples[int(0.025 * n_resamples)], samples[int(0.975 * n_resamples) - 1]]
