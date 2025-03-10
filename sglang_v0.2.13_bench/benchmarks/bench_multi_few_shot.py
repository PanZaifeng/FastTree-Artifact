import argparse
import time
import nvtx
import os
import multiprocessing as mp

import sglang as sgl
from sglang.utils import read_jsonl
from fasttree_sglang_plugin_v0_2_13 import load_fasttree_plugin


enable_fasttree = os.getenv("ENABLE_FASTTREE", "0") == "1"
annotate_breakdown = os.getenv("ANNOTATE_BREAKDOWN", "0") == "1"
if enable_fasttree or annotate_breakdown:
    load_fasttree_plugin(enable_fasttree, annotate_breakdown)

# Adapted from https://github.com/sgl-project/sglang/blob/v0.2.13/benchmark/gsm8k/bench_sglang.py


def get_one_example(lines, i, include_answer):
    ret = "Question: " + lines[i]["question"] + "\nAnswer:"
    if include_answer:
        ret += " " + lines[i]["answer"]
    return ret


def get_few_shot_examples(lines, offset, k):
    ret = ""
    for i in range(k):
        ret += get_one_example(lines, offset + i, True) + "\n\n"
    return ret


@sgl.function
def few_shot_with_sys(s, sys_prompt, few_shot_examples, question, max_tokens):
    s += sgl.system(sys_prompt)
    s += few_shot_examples
    s += "Question: " + question + "\n"
    s += "Answer:" + sgl.gen("answer", max_tokens=max_tokens)


def main(args):
    with open(args.sys_prompt_path, "r") as f:
        sys_prompt = f.read()
    lines = read_jsonl(args.data_path)

    arguments = []
    stride = args.num_shots + args.num_questions
    for i in range(args.num_example_branches):
        offset = stride * i
        few_shot_examples = get_few_shot_examples(lines, offset, args.num_shots)
        for j in range(offset + args.num_shots, stride * (i + 1)):
            arguments.append(
                {
                    "sys_prompt": sys_prompt,
                    "few_shot_examples": few_shot_examples,
                    "question": get_one_example(lines, j, False),
                    "max_tokens": args.max_tokens,
                }
            )

    # Select backend
    runtime = sgl.Runtime(
        model_path=args.model_path,
        context_length=20000,
        disable_cuda_graph=True,
        disable_flashinfer=args.disable_flashinfer,
    )
    sgl.set_default_backend(runtime)

    # Run requests
    tic = time.time()
    with nvtx.annotate("End-to-end"):
        states = few_shot_with_sys.run_batch(
            arguments,
            temperature=0,
            backend=runtime,
            num_threads=args.parallel,
            progress_bar=False,
        )
    latency = time.time() - tic

    print(f"Latency: {latency:.3f}")

    runtime.shutdown()


if __name__ == "__main__":
    if annotate_breakdown:
        mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", type=str, default="meta-llama/Llama-2-7b-chat-hf"
    )
    parser.add_argument("--data-path", type=str, default="data/test.jsonl")
    parser.add_argument("--sys-prompt-path", type=str, default="data/system_prompt.txt")
    parser.add_argument("--num-example-branches", type=int, default=8)
    parser.add_argument("--num-shots", type=int, default=20)
    parser.add_argument("--num-questions", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--parallel", type=int, default=64)
    parser.add_argument("--disable-flashinfer", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
