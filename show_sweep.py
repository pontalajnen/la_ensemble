#!/usr/bin/env python3
"""Pretty-print sweep_sgld_results.csv grouped by variant.

Usage:
    python show_sweep.py                       # sweep_sgld_results.csv
    python show_sweep.py my_results.csv        # any csv with the same header
    python show_sweep.py --sort acc            # sort by accuracy (default: nll)
"""

import argparse
import csv
import os
import sys


# metric direction: True = higher is better, False = lower is better
METRICS = [
    ("clean_accuracy", "acc",   True,  "{:.4f}"),
    ("ECE",            "ECE",   False, "{:.2f}"),
    ("nll",            "NLL",   False, "{:.3f}"),
    ("OOD AUROC",      "AUROC", True,  "{:.4f}"),
    ("SHIFT ACCURACY", "s-acc", True,  "{:.4f}"),
    ("SHIFT ECE",      "s-ECE", False, "{:.2f}"),
    ("val_loss",       "vloss", False, "{:.3f}"),
]

HPARAMS = [
    ("lr",          "lr",   "{:g}"),
    ("temperature", "T",    "{:g}"),
    ("sampling_lr", "s_lr", "{:g}"),
]


def parse_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def format_row(row, col_widths, cols):
    parts = []
    for name, fmt in cols:
        v = row.get(name)
        if v is None or v == "":
            s = "—"
        elif isinstance(v, float):
            s = fmt.format(v)
        else:
            s = str(v)
        parts.append(s.rjust(col_widths[name]))
    return "  ".join(parts)


def is_baseline(row):
    return str(row.get("config", "")).startswith("sgd")


def print_variant(variant, rows, sort_key):
    if not rows:
        return

    # coerce metric strings to floats where possible
    for r in rows:
        for m, *_ in METRICS:
            r[m] = parse_float(r.get(m))
        for h, *_ in HPARAMS:
            r[h] = parse_float(r.get(h))

    baselines = [r for r in rows if is_baseline(r)]
    sgld = [r for r in rows if not is_baseline(r)]

    # sort direction: NLL down (lower better) unless overridden
    if sort_key == "acc":
        key_fn = lambda r: -(r["clean_accuracy"] or -1)
    elif sort_key == "ece":
        key_fn = lambda r: (r["ECE"] if r["ECE"] is not None else 1e9)
    else:
        key_fn = lambda r: (r["nll"] if r["nll"] is not None else 1e9)
    sgld.sort(key=key_fn)
    baselines.sort(key=key_fn)
    rows = baselines + sgld

    print("\n" + "=" * 100)
    print(f"{variant.upper()}  ({len(rows)} rows: {len(baselines)} SGD baseline(s) + "
          f"{len(sgld)} SGLD configs, sorted by {sort_key})")
    print("=" * 100)

    # header
    cols = [("config", "{}")]
    cols += [(h, fmt) for h, _, fmt in HPARAMS]
    for m, _, up, fmt in METRICS:
        cols.append((m, fmt))

    labels = {"config": "config"}
    labels.update({h: alias for h, alias, _ in HPARAMS})
    for m, alias, up, _ in METRICS:
        labels[m] = alias + ("↑" if up else "↓")

    # width per column: max of label and formatted values
    col_widths = {}
    for name, fmt in cols:
        w = len(labels[name])
        for r in rows:
            v = r.get(name)
            if v is None or v == "":
                s = "—"
            elif isinstance(v, float):
                s = fmt.format(v)
            else:
                s = str(v)
            w = max(w, len(s))
        col_widths[name] = w

    header = "  ".join(labels[n].rjust(col_widths[n]) for n, _ in cols)
    print(header)
    print("-" * len(header))
    for r in baselines:
        print(format_row(r, col_widths, cols))
    if baselines and sgld:
        print("-" * len(header))
    for r in sgld:
        print(format_row(r, col_widths, cols))

    # best per metric — SGLD and baselines reported separately
    def _best(rs, m, up):
        vals = [(r[m], r["config"]) for r in rs if r[m] is not None]
        if not vals:
            return None
        return max(vals) if up else min(vals)

    if sgld:
        print("\n  Best SGLD per metric:")
        for m, alias, up, fmt in METRICS:
            b = _best(sgld, m, up)
            if b:
                arrow = "↑" if up else "↓"
                print(f"    {alias+arrow:8s}  {fmt.format(b[0])}   ({b[1]})")
    if baselines:
        print("\n  SGD baselines:")
        for r in baselines:
            cells = []
            for m, alias, up, fmt in METRICS:
                if r[m] is not None:
                    cells.append(f"{alias}={fmt.format(r[m])}")
            print(f"    {r['config']:14s}  " + "  ".join(cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="sweep_sgld_results.csv")
    ap.add_argument("--sort", choices=["nll", "acc", "ece"], default="nll")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        sys.exit(f"[error] not found: {args.csv}")

    rows = load(args.csv)
    if not rows:
        sys.exit(f"[error] empty: {args.csv}")

    print(f"[source] {args.csv}   ({len(rows)} rows)")

    variants = {}
    for r in rows:
        variants.setdefault(r.get("variant", "unknown"), []).append(r)

    for v in sorted(variants):
        print_variant(v, variants[v], args.sort)


if __name__ == "__main__":
    main()
