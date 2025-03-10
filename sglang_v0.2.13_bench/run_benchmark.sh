#!/bin/bash

FORCE_RUN=${1:-false}

cur_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
bench_dir=$cur_dir/benchmarks
log_dir=$bench_dir/logs
breakdown_dir=$bench_dir/breakdowns
data_dir=$bench_dir/data

mkdir -p "$log_dir"
mkdir -p "$breakdown_dir"

engines=("sglang" "flashinfer" "fasttree")
engine_extensions=("--disable-flashinfer" "" "")
engine_envs=("ENABLE_FASTTREE=0" "ENABLE_FASTTREE=0" "ENABLE_FASTTREE=1")
models=("meta-llama/Llama-2-7b-hf" "mistralai/Mistral-7B-v0.1")
benchmarks=("multi_level_system" "multi_few_shot" "multi_chain_reasoning" "multi_document")
scripts=("bench_multi_level_system.py" "bench_multi_few_shot.py" "bench_multi_chain_reasoning.py" "bench_multi_document.py")

for model in "${models[@]}"; do
  model_name=$(basename "$model")

  for ((i = 0; i < ${#engines[@]}; i++)); do
    engine="${engines[$i]}"
    engine_extension="${engine_extensions[$i]}"
    engine_env="${engine_envs[$i]}"

    log="$log_dir/${model_name}-${engine}.log"
    log_exists=false
    if [[ -f "$log" ]]; then
      log_exists=true
      if [[ "$FORCE_RUN" == "true" ]]; then
        rm "$log"
      fi
    fi

    for ((j = 0; j < ${#benchmarks[@]}; j++)); do
      benchmark="${benchmarks[$j]}"
      script="${scripts[$j]}"

      case "$benchmark" in
        "multi_level_system")
          data_path="$data_dir/test.jsonl"
          extra_args="--prompt-templ-path $data_dir/system_prompt.template"
          ;;
        "multi_few_shot" | "multi_chain_reasoning")
          data_path="$data_dir/test.jsonl"
          extra_args="--sys-prompt-path $data_dir/system_prompt.txt"
          ;;
        "multi_document")
          data_path="$data_dir/questions.jsonl"
          extra_args=""
          ;;
      esac

      if [[ "$log_exists" == false ]] || [ "$FORCE_RUN" == "true" ]; then
        echo -n "$benchmark " >> "$log"
        env $engine_env python "$bench_dir/$script" \
          --model-path "$model" \
          --data-path "$data_path" \
          $extra_args $engine_extension \
          | grep Latency | awk '{print $2}' >> "$log"
      fi

      if [[ "$model" == "meta-llama/Llama-2-7b-hf" ]]; then
        breakdown_prefix="$breakdown_dir/${model_name}-${engine}-${benchmark}"
        breakdown_rep="${breakdown_prefix}.nsys-rep"
        if [[ ! -f "$breakdown_rep" ]] || [ "$FORCE_RUN" == "true" ]; then
          env $engine_env ANNOTATE_BREAKDOWN=1 nsys profile \
            -o $breakdown_prefix -t cuda,nvtx -f true \
              python "$bench_dir/$script" \
                --model-path "$model" \
                --data-path "$data_path" \
                $extra_args $engine_extension
        fi

        breakdown_db="${breakdown_prefix}.sqlite"
        breakdown="${breakdown_prefix}.txt"
        nsys export -f true --type=sqlite --output=$breakdown_db $breakdown_rep
        python $bench_dir/parse_breakdown.py "$breakdown_db" > "$breakdown"
      fi
    done
  done
done