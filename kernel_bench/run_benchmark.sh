#!/bin/bash

FORCE_RUN=${1:-false}

nodes_contexts=(
  "1,2,64 8,256,32"
  "1,4,256 8,256,32"
  "1,4,8,256 8,256,256,32"
  "1,256 256,32"
  "1,1024 2048,32"
  "1,16,64 1024,256,32"
  "1,4,16,512 1024,256,128,32"
  "1,4,16,64,256,1024 256,8,256,64,32,256"
  "1,10 4000,400"
  "1,2,4,8,16,32,64,128,1024 16,16,16,16,16,16,16,16,16"
  "1,2,4,8,16,32,64,128,1024 256,128,64,16,16,16,16,16,16"
  "1,8,16,32,64,128,1024 256,128,64,16,16,16,16"
  "1,8,16,32,64,256,1024 256,128,64,16,16,16,16"
  "1,16,32,64,128,1024 256,128,64,16,16,16"
  "1,16,32,64,256,1024 256,128,64,16,16,16"
)

qk_heads=(
  "16 1"
  "32 8"
  "32 32"
)

mkdir -p trees
mkdir -p outputs
mkdir -p plans
for nc in "${nodes_contexts[@]}"; do
  tree=nc_${nc// /_}.txt
  IFS=' ' read -r nodes contexts <<< "$nc"
  if [ ! -f "trees/$tree" ] || [ "$FORCE_RUN" == "true" ]; then
    echo "[INFO] Generating tree file: trees/$tree"
    python tree_generation.py -n "$nodes" -l "$contexts" -o "trees/$tree"
  else
    echo "[SKIP] trees/$tree already exists"
  fi

  for qk in "${qk_heads[@]}"; do
    IFS=' ' read -r q_heads k_heads <<< "$qk"
    f=nc_${nc// /_}_qk_${qk// /_}.txt

    gqa_ratio=$((q_heads / k_heads))
    if [ $gqa_ratio == 1 ]; then
      q_tile_sizes="64,16"
    elif [ $gqa_ratio == 4 ]; then
      q_tile_sizes="16,4"
    elif [ $gqa_ratio == 16 ]; then
      q_tile_sizes="4,1"
    fi

    if [ ! -f "outputs/$f" ] || [ "$FORCE_RUN" == "true" ]; then
      rm -f outputs/$f
      echo "[INFO] Running benchmark.py -> outputs/$f"
      for choice in all; do
        python benchmark.py \
          --num_qo_heads $q_heads --num_kv_heads $k_heads \
          --q_tile_size_per_phase $q_tile_sizes \
          --load_file_path trees/$tree \
          --fasttree_heuristic_path plans/$f \
          --choice $choice | tee -a outputs/$f
      done

      for gamma in 10 -10; do
        if [ $gamma == 10 ]; then
          kv_split_sizes="81920,81920"
        else
          kv_split_sizes="1024,128"
        fi
        python benchmark.py \
            --num_qo_heads $q_heads --num_kv_heads $k_heads \
            --q_tile_size_per_phase $q_tile_sizes \
            --load_file_path trees/$tree \
            --choice fasttree \
            --alpha 0 --beta 0 --gamma $gamma \
            --kv_split_sizes $kv_split_sizes \
            --signature "fasttree $gamma" | tee -a outputs/$f
      done

      echo "[INFO] Running bench_cascade.py -> outputs/$f"
      # FlashInfer MultiCascadeAttn
      python bench_cascade.py \
        --node_num_per_level $nodes --node_seqlen_per_level $contexts \
        --num_qo_heads $q_heads \
        --num_kv_heads $k_heads | tee -a outputs/$f
    else
      echo "[SKIP] outputs/$f already exists"
    fi
  done
done