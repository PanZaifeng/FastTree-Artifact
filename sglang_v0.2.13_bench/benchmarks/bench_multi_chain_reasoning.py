import argparse
import time
import nvtx
import os
import multiprocessing as mp

from sglang.utils import read_jsonl
from fasttree_sglang_plugin_v0_2_13 import load_fasttree_plugin


enable_fasttree = os.getenv("ENABLE_FASTTREE", "0") == "1"
annotate_breakdown = os.getenv("ANNOTATE_BREAKDOWN", "0") == "1"
if enable_fasttree or annotate_breakdown:
    load_fasttree_plugin(enable_fasttree, annotate_breakdown)

# Adapted from https://github.com/sgl-project/sglang/blob/v0.2.13/benchmark/multi_chain_reasoning/bench_sglang.py

prompt_lib = [
    "Let us think step by step.",
    "Approach this methodically. Let's dissect the problem into smaller, more manageable parts.",
    "It's important to proceed step by step, ensuring accuracy at each stage.",
    "Take a deep breath and break this down.",
    "A little bit of arithmetic and a logical approach will help us quickly arrive at the solution to this problem.",
    "I am extremely good at math.",
]


def main(args):
    lines = list(read_jsonl(args.data_path))
    with open(args.sys_prompt_path, "r") as f:
        sys_prompt = f.read()

    questions = []
    for i in range(len(lines[: args.num_questions])):
        questions.append(lines[i]["question"])
    arguments = [
        {
            "sys_prompt": sys_prompt,
            "question": q,
            "max_tokens": args.max_tokens,
        }
        for q in questions
    ]

    num_chains = args.num_chains

    #####################################
    ######### SGL Program Begin #########
    #####################################

    import sglang as sgl

    @sgl.function
    def multi_chain_gsm8k(s, sys_prompt, question, max_tokens):
        s += sgl.system(sys_prompt)
        s += "Question: " + question + "\n"
        # s += "Answer: " + prompt_lib[0] + sgl.gen("answer", max_tokens=256, stop="Question",
        #    temperature=0)
        # return

        forks = s.fork(num_chains)
        for i in range(num_chains):
            forks[i] += (
                "Answer: "
                + prompt_lib[i % 6]
                + sgl.gen(
                    "chain", max_tokens=max_tokens, temperature=0.3, stop="Question"
                )
            )
        forks.join()

        s += "Answer: To answer this question, here are some possible solutions. "
        s += "After considering all of them, I will do a majority vote.\n\n"
        for i in range(num_chains):
            s += f"Solution {i+1}: " + forks[i]["chain"].strip() + "\n\n"
        s += "\nBy considering the above solutions and doing a majority vote, I think the final answer (a single integer number) is "
        s += sgl.gen("answer", max_tokens=max_tokens)

    #####################################
    ########## SGL Program End ##########
    #####################################

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
        states = multi_chain_gsm8k.run_batch(
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
    parser.add_argument("--num-chains", type=int, default=4)
    parser.add_argument("--num-questions", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--parallel", type=int, default=64)
    parser.add_argument("--disable-flashinfer", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
