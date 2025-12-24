import argparse
import pandas as pd
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="physionet_metrics_table.csv", help="input metrics table csv")
    ap.add_argument("--out", default="metrics_summary_by_label.csv", help="output summary csv")
    args = ap.parse_args()

    df = pd.read_csv(args.input)

    # expected metric columns (keep only the ones that exist)
    metric_cols = ["step_length", "symmetry", "loading_rate", "pronation_angle"]
    metric_cols = [c for c in metric_cols if c in df.columns]

    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column (0=Healthy, 1=PD).")

    # group by label and compute stats
    summary = []
    for label, g in df.groupby("label"):
        for m in metric_cols:
            vals = g[m].dropna().astype(float)
            if len(vals) == 0:
                continue
            summary.append({
                "label": int(label),
                "metric": m,
                "count": int(vals.count()),
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if vals.count() > 1 else 0.0,
                "min": float(vals.min()),
                "max": float(vals.max()),
            })

    out_df = pd.DataFrame(summary)

    # add readable label names
    out_df["class_name"] = out_df["label"].map({0: "Healthy", 1: "PD"})
    out_df = out_df[["label", "class_name", "metric", "count", "mean", "std", "min", "max"]]

    out_df.to_csv(args.out, index=False)

    print("✅ Saved:", args.out)
    print(out_df.head(12).to_string(index=False))

if __name__ == "__main__":
    main()
