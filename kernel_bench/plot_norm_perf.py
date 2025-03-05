import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import argparse

plt.rcParams["pdf.fonttype"] = 42


def clean_value(val):
    if isinstance(val, str):
        val = val.strip()
        val = re.sub(r"\*", "", val)
    return float(val)

def extract_data(input_file, metrics):
    df = pd.read_csv(input_file, sep="|", engine="python", skipinitialspace=True)
    df = df.iloc[:, 1:-1]
    df.columns = [col.strip() for col in df.columns]
    df = df[~df["node-num-per-level"].str.strip().str.match(r"^-+$")]
    df["config"] = "N: " + df["node-num-per-level"].str.strip() + "\nC: " + df["node-context-per-level"].str.strip()
    df["Query Heads"] = df["Query Heads"].apply(lambda x: int(x.strip()))
    df["KV Heads"] = df["KV Heads"].apply(lambda x: int(x.strip()))
    df["GQA"] = df["Query Heads"] // df["KV Heads"]
    for col in metrics:
        df[col] = df[col].apply(clean_value)
    df[metrics] = df[metrics].div(df[metrics].max(axis=1), axis=0)
    gqa_groups = df.groupby("GQA")
    gqa_groups = sorted(gqa_groups, key=lambda x: x[0])
    return gqa_groups

def plot_axes(fig, gqa_groups, metrics, nrows, ncols, fontsize):
    labels = ["Flash-Attn", "SGLang", "FlashInfer", "DeFT", "CascadeAttn", "FastTree w/o Greedy", "FastTree"]
    colors = ["#CCCCCC", "#B4C7E7", "#C5E0B4", "#FFE699", "#F6CAE5", "#F8CBAD", "#C5796A"]
    hatches = ["" for _ in colors]

    axes = fig.subplots(nrows, ncols, sharex="col", sharey="row")
    
    configs = gqa_groups[0][1]["config"].tolist()
    n_configs = len(configs)
    n_metrics = len(metrics)

    for r in range(nrows):
        for c in range(ncols):
            if nrows == 1 or ncols == 1:
                ax = axes[r * ncols + c]
            else:
                ax = axes[r, c]
            gqa_value, group = gqa_groups[r * ncols + c]
            width = 1.0 / (n_metrics + 2)
            location = np.arange(n_configs)
            for i, (metric, label, color, hatch) in enumerate(zip(metrics, labels, colors, hatches)):
                values = group[metric].values
                ax.bar(location + width * i, values, width=width, label=label,
                       color=color, hatch=hatch, edgecolor="k", alpha=1)
            ax.set_xticks(location + width * n_metrics / 2)
            ax.set_xticklabels(configs, rotation=30, ha="right", fontsize=fontsize-8)
            for idx, tick in enumerate(ax.get_xticklabels()):
                tick.set_color("black" if idx % 2 == 0 else "#525252")
            ax.set_xlim(-width, n_configs - width)
            ax.set_ylim(0, 1.2)
            ax.grid(axis="y")
            ax.tick_params(axis="y", labelsize=fontsize)
            ax.set_title(f"Query Heads={group['Query Heads'].iloc[0]}  KV Heads={group['KV Heads'].iloc[0]}",
                         fontsize=fontsize-6, y=0.85)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="./parsed_output.md")
    parser.add_argument("--output", type=str, default="./norm_perf.pdf")
    args = parser.parse_args()

    nrows = 3
    ncols = 1
    fontsize = 20
    fig = plt.figure(figsize=(12 * ncols + 3, 2 * nrows + 3.25))

    metrics = ["Flash-Attn", "SGLang-Triton", "FlashInfer", "DeFT", "MultiCascade", "FastTree (naive)", "FastTree"]
    gqa_groups = extract_data(args.input, metrics)
    plot_axes(fig, gqa_groups, metrics, nrows, ncols, fontsize)

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor="none", top=False,
                    bottom=False, left=False, right=False)
    plt.grid(False)
    plt.ylabel("Normalized Kernel Performance", fontsize=fontsize, labelpad=20)
    bars, legend_labels = fig.axes[0].get_legend_handles_labels()
    plt.legend(bars, legend_labels, bbox_to_anchor=(0.5, 1.0), loc="lower center", ncol=7,
               handletextpad=0.6, fontsize=fontsize-6, columnspacing=1.5, frameon=False)
    fig.tight_layout()
    # plt.show()
    fig.savefig(args.output, bbox_inches="tight")