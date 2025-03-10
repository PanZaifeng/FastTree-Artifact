#!/bin/bash

cur_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
bench_dir=$cur_dir/benchmarks
plot_dir=$bench_dir/plots
output_dir=$cur_dir/outputs

mkdir -p $output_dir

python $plot_dir/e2e.py --input-dir $bench_dir/logs --output $output_dir/e2e.pdf
python $plot_dir/breakdown.py --input-dir $bench_dir/breakdowns --output $output_dir/breakdown.pdf
python $plot_dir/kern_breakdown.py --input-dir $bench_dir/breakdowns --output $output_dir/kern_breakdown.pdf