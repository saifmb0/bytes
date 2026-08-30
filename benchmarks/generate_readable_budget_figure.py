"""Generate the paper's readable selection-budget figure from the emitted LaTeX table.

The table remains the source of benchmark values. This plot intentionally shows only
the representative methods needed to explain the two regimes in the main paper; the
complete grid remains in tables/a2_qwen0.5b_16384.tex.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TABLE = Path("tables/a2_qwen0.5b_16384.tex")
OUTPUT = Path("figures/fig_study_budget_readable.pdf")
DENSE_PPL = 9.17
BUDGETS = [1, 2, 5, 10]

METHODS = {
    "exact (fp32 score)": ("Exact-key oracle", "#111111", "o", "--"),
    "fp8 scalar": ("fp8 scalar", "#6b6b6b", "s", "-"),
    "PQ-LUT (PQCache)": ("PQ-LUT", "#1f77b4", "D", "-"),
    "int4 scalar": ("int4 scalar", "#2ca02c", "^", "-"),
    "sign-VQ (Self-Index.)": ("sign-VQ", "#d62728", "v", "-"),
    "int2 scalar": ("int2 scalar", "#9467bd", "<", "-"),
    "Quest (page)": ("Quest (page)", "#8c564b", "h", "-."),
    "H2O (evict)": ("H2O (eviction)", "#e377c2", "x", ":"),
}


def read_rows():
    rows = {}
    for raw in TABLE.read_text().splitlines():
        if "&" not in raw or not raw.rstrip().endswith(r"\\"):
            continue
        cells = [cell.strip() for cell in raw[:-2].split("&")]
        if len(cells) != 6 or cells[0] not in METHODS:
            continue
        rows[cells[0]] = [float(value) for value in cells[2:]]
    missing = set(METHODS) - set(rows)
    if missing:
        raise RuntimeError(f"missing methods in {TABLE}: {sorted(missing)}")
    return rows


def main():
    rows = read_rows()
    fig, (overview, zoom) = plt.subplots(1, 2, figsize=(10.8, 4.0), sharex=True)

    for method, (label, color, marker, linestyle) in METHODS.items():
        values = rows[method]
        overview.plot(
            BUDGETS,
            values,
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.8,
            markersize=5.5,
        )
        if method not in {"Quest (page)", "H2O (evict)"}:
            zoom.plot(
                BUDGETS,
                values,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.8,
                markersize=5.5,
            )

    for axis in (overview, zoom):
        axis.axhline(DENSE_PPL, color="#444444", linestyle="--", linewidth=1.1)
        axis.fill_between(BUDGETS, DENSE_PPL, DENSE_PPL + 0.5, color="#d9ead3", alpha=0.45)
        axis.set_xticks(BUDGETS)
        axis.set_xlabel("retained historical keys (%)")
        axis.grid(alpha=0.25)

    overview.set_ylabel("perplexity (lower is better)")
    overview.set_ylim(8.9, 17.5)
    overview.set_title("(a) Overall: per-key, page, and eviction regimes")
    overview.annotate(
        "dense attention: 9.17",
        xy=(9.8, DENSE_PPL),
        xytext=(6.0, 10.0),
        arrowprops={"arrowstyle": "->", "color": "#444444"},
        fontsize=8.5,
    )

    zoom.set_ylim(9.0, 10.9)
    zoom.set_title("(b) Zoom: differences among per-key selectors")
    zoom.text(9.8, 9.61, "within +0.5 PPL of dense", ha="right", fontsize=8.5, color="#315c2b")

    handles, labels = overview.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8.5)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.25, wspace=0.18)
    fig.savefig(OUTPUT, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
