#!/bin/bash

echo "Please set HF_TOKEN environment variable to your Hugging Face API token."

HF_TOKEN=${HF_TOKEN:-$1}
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}

script_path=$(realpath "$0")
sglang_bench_dir=$(dirname "$0")

docker build -f ${sglang_bench_dir}/Dockerfile -t fasttree_sglang:latest ${sglang_bench_dir}
docker run --rm -e "HF_TOKEN=${HF_TOKEN}" \
    -v $HF_HOME:/root/.cache/huggingface -v ${sglang_bench_dir}:/sglang_bench \
    --gpus all -it fasttree_sglang:latest sh -c \
        "cd /sglang_bench && pip install -e . \
            && ./prepare_data.sh \
            && ./run_benchmark.sh true \
            && ./plot.sh"
