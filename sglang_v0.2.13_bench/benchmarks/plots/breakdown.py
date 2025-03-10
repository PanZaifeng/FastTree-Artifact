import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["pdf.fonttype"] = 42


def plot_axes(fig, benchmarks, data, nrows=2, ncols=2, fontsize=22):
    engines = ["SGLang", "FlashInfer", "FastTree"]
    labels = ["Prefill", "Decoding", "Preprocess", "Other"]
    colors = ["#B4C7E7", "#C5E0B4", "#FFE699", "#CCCCCC"]
    hatches = ["\\", "/", "-", ""]

    # axes = fig.subplots(nrows, ncols, sharex="col", sharey="row")
    axes = fig.subplots(nrows, ncols, sharey="row")

    for r in range(nrows):
        for c in range(ncols):
            if nrows == 1 or ncols == 1:
                ax = axes[r * ncols + c]
            else:
                ax = axes[r, c]
            datum = data[r * ncols + c]  # (engine, part)
            datum = np.transpose(datum)  # (part, engine)

            width = 1.0 / (len(engines) - 1)
            location = np.arange(len(engines))

            bottom = np.zeros(len(engines))
            for bar, label, color, hatch in zip(datum, labels, colors, hatches):
                ax.barh(
                    location,
                    bar,
                    left=bottom,
                    height=width,
                    label=label,
                    color=color,
                    hatch=hatch,
                    edgecolor="k",
                    alpha=1,
                )
                bottom += bar

            ax.set_ylim(-width, len(engines) - width)
            ax.set_yticks(location)
            ax.set_yticklabels(engines, fontsize=fontsize)
            ax.tick_params(axis="x", labelsize=fontsize - 2)
            ax.tick_params(top=False, right=False)
            ax.set_title(f"{benchmarks[r * ncols + c]}", fontsize=fontsize, y=-0.4)


def read_breakdown(fname):
    with open(fname) as f:
        for line in f.readlines():
            words = line.split()
            if len(words) > 0 and words[0] == "[NVTX]":
                if words[1] == "forward_decode_batch":
                    decoding = float(words[2])
                elif words[1] == "forward_prefill_batch":
                    prefill = float(words[2])
                elif words[1] == "prepare_for_fasttree":
                    preprocess = float(words[2])
                elif words[1] == "end_to_end":
                    e2e = float(words[2])
    overhead = e2e - sum([prefill, decoding, preprocess])
    breakdown = [prefill, decoding, preprocess, overhead]
    # breakdown = np.array(breakdown) / e2e * 100
    return breakdown


def normalize(datum):
    sglang_sum = np.sum(datum[0])
    return np.array(datum) / sglang_sum


def main(args):
    data = []
    for benchmark in args.benchmarks:
        datum = []
        for engine in ["sglang", "flashinfer", "fasttree"]:
            fname = os.path.join(
                args.input_dir, f"{args.model}-{engine}-{benchmark}.txt"
            )
            datum.append(read_breakdown(fname))
        norm = normalize(datum)
        data.append(norm)

    data = np.array(data)
    print(
        "Average Decoding Speedup over SGLang",
        np.mean((data[:, 0, 1] / data[:, 2, 1]).flatten()),
    )
    print(
        "Average Decoding Speedup over FlashInfer",
        np.mean((data[:, 1, 1] / data[:, 2, 1]).flatten()),
    )

    fig = plt.figure(figsize=(4 * args.ncols + 2, 2 * args.nrows + 2))
    plot_axes(fig, args.benchmark_labels, data, args.nrows, args.ncols, args.fontsize)

    fig.add_subplot(111, frameon=False)
    # fig.subplots_adjust(hspace=0.8)
    plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    # plt.xlabel("End-to-End Execution Time Breakdown", fontsize=args.fontsize, labelpad=32, x=0.48)
    bars, labels = fig.axes[0].get_legend_handles_labels()
    plt.legend(
        bars,
        labels,
        bbox_to_anchor=(0.45, 1.0),
        loc="lower center",
        ncol=4,
        handletextpad=1,
        fontsize=args.fontsize,
        columnspacing=1.0,
        frameon=False,
    )
    fig.tight_layout()
    # plt.show()
    fig.savefig(args.output, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="../breakdowns")
    parser.add_argument("--output", type=str, default="./outputs/breakdown.pdf")
    parser.add_argument("--model", type=str, default="Llama-2-7b-hf")
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        default=[
            "multi_level_system",
            "multi_few_shot",
            "multi_chain_reasoning",
            "multi_document",
        ],
    )
    parser.add_argument(
        "--benchmark_labels",
        type=str,
        nargs="+",
        default=[
            "Multi-Level System Prompt",
            "Multiple Few-Shot Learning",
            "Multi-Chain Reasoning",
            "Multi-Document QA",
        ],
    )
    parser.add_argument("--fontsize", type=int, default=18)
    parser.add_argument("--nrows", type=int, default=2)
    parser.add_argument("--ncols", type=int, default=2)
    args = parser.parse_args()
    main(args)
