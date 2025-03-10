import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["pdf.fonttype"] = 42


def plot_axes(fig, models, benchmarks, data_list, nrows=1, ncols=2, fontsize=22):
    labels = ["SGLang w/ Triton", "SGLang w/ FlashInfer", "FastTree (Ours)"]
    colors = ["#B4C7E7", "#FFE699", "#F8CBAD"]
    hatches = ["\\", "/", "-"]

    axes = fig.subplots(nrows, ncols, sharex="col", sharey="row")

    for r in range(nrows):
        for c in range(ncols):
            if nrows == 1 or ncols == 1:
                ax = axes[r * ncols + c]
            else:
                ax = axes[r, c]
            # SGLang, FlashInfer, FastTree
            data = data_list[r * ncols + c]

            width = 1.0 / (len(data) + 1)
            location = np.arange(len(benchmarks))

            for i, (bar, label, color, hatch) in enumerate(
                zip(data, labels, colors, hatches)
            ):
                ax.bar(
                    location + width * i,
                    bar,
                    width=width,
                    label=label,
                    color=color,
                    hatch=hatch,
                    edgecolor="k",
                    alpha=1,
                )

            ax.set_xticks(location + width * len(labels) / 2)
            ax.set_xticklabels(benchmarks, fontsize=fontsize)
            ax.set_xlim(-width, len(benchmarks) - width)
            ax.tick_params(axis="y", labelsize=fontsize)
            ax.tick_params(bottom=False, top=False, right=False)
            ax.set_title(f"{models[r * ncols + c]}", fontsize=fontsize + 2, y=-0.35)


def read_latency(fname, benchmarks):
    bench_to_latencies = {}
    with open(fname) as f:
        for line in f.readlines():
            bench, latency = line.strip().split()
            bench_to_latencies[bench] = float(latency)
    return [bench_to_latencies[bench] for bench in benchmarks]


def normalize(data):
    norm = []
    our = np.array(data[-1])
    for datum in data:
        norm.append(our / np.array(datum))
    return norm


def read_data(args):
    data_list = []
    for model in args.models:
        data = []
        for engine in ["sglang", "flashinfer", "fasttree"]:
            data.append(
                read_latency(
                    os.path.join(args.input_dir, f"{model}-{engine}.log"),
                    args.benchmarks,
                ),
            )
            norm = normalize(data)
        data_list.append(norm)

    data_list = np.array(data_list)
    print("Max speedup over FlashInfer", np.max(1 / data_list[:, 1, :]))
    print("Avg speedups on Llama over FlashInfer", np.mean(1 / data_list[0, 1, :]))
    print("Avg speedups on Mistral over FlashInfer", np.mean(1 / data_list[1, 1, :]))
    print("Avg speedups on Llama over SGLang", np.mean(1 / data_list[0, 0, :]))
    print("Avg speedups on Mistral over SGLang", np.mean(1 / data_list[1, 0, :]))

    return data_list


def main(args):
    data_list = read_data(args)

    fig = plt.figure(figsize=(4 * args.ncols + 2, 2 * args.nrows + 3.25))
    plot_axes(
        fig,
        args.model_labels,
        args.benchmark_labels,
        data_list,
        args.nrows,
        args.ncols,
        args.fontsize,
    )

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    # plt.xlabel("Benchmarks", fontsize=args.fontsize, labelpad=30)
    plt.ylabel(
        "Normalized Performance", fontsize=args.fontsize + 2, labelpad=25, y=0.55
    )
    bars, labels = fig.axes[0].get_legend_handles_labels()
    plt.legend(
        bars,
        labels,
        bbox_to_anchor=(0.47, 1.0),
        loc="lower center",
        ncol=2,
        handletextpad=1,
        fontsize=args.fontsize - 2,
        columnspacing=2.5,
        frameon=False,
    )
    fig.tight_layout()
    # plt.show()
    fig.savefig(args.output, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="../logs")
    parser.add_argument("--output", type=str, default="./outputs/e2e.pdf")
    parser.add_argument(
        "--models", type=str, nargs="+", default=["Llama-2-7b-hf", "Mistral-7B-v0.1"]
    )
    parser.add_argument(
        "--model_labels", type=str, nargs="+", default=["Llama-7B", "Mistral-7B"]
    )
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
        "--benchmark_labels", type=str, nargs="+", default=["A", "B", "C", "D"]
    )
    parser.add_argument("--fontsize", type=int, default=22)
    parser.add_argument("--nrows", type=int, default=1)
    parser.add_argument("--ncols", type=int, default=2)
    args = parser.parse_args()
    main(args)
