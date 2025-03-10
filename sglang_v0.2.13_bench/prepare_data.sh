#!/bin/bash

cur_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
data_dir=$cur_dir/benchmarks/data

gsm8k=$data_dir/test.jsonl
curl -o $gsm8k https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl

llama2=$data_dir/llama2.pdf
curl -o $llama2 https://arxiv.org/pdf/2307.09288

cd $data_dir

# This is adapted from https://github.com/0xeb/TheBigPromptLibrary/blob/main/SystemPrompts/Meta.ai/metaai_llama3-04182024.md
python -c "with open('system_prompt.template') as f, open('system_prompt.txt', 'w') as o: o.write(f.read().format(LOCATION='the United States', LANGUAGE='English'))"

python build_doc_dataset.py
