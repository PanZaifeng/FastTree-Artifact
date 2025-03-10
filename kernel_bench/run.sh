#!/bin/bash

script_path=$(realpath "$0")
kernel_bench_dir=$(dirname "$0")

docker build -f ${kernel_bench_dir}/Dockerfile -t fasttree_kernel:latest ${kernel_bench_dir}
docker run --rm -v ${kernel_bench_dir}:/kernel_bench --gpus all -it fasttree_kernel:latest sh -c \
    "cd /kernel_bench && ./run_benchmark.sh true \
        && python ./parse_output.py > parsed_output.md \
        && python ./plot_norm_perf.py"
