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


@sgl.function
def text_qa_with_sys(s, sys_prompt, question, max_tokens):
    s += sgl.system(sys_prompt)
    s += "Q: " + question + "\n"
    s += "A:" + sgl.gen("answer", max_tokens=max_tokens)


def main(args):
    with open(args.prompt_templ_path, "r") as f:
        sys_prompt_templ = f.read()
    choices = [
        ("the United States", "English"),
        ("the United States", "Spanish"),
        ("Canada", "English"),
        ("China", "Chinese"),
    ]
    sys_prompts = [
        sys_prompt_templ.format(LOCATION=location, LANGUAGE=language)
        for location, language in choices
    ]

    lines = list(read_jsonl(args.data_path))[: args.num_questions]
    arguments = []
    for sys_prompt in sys_prompts:
        for line in lines:
            arguments.append(
                {
                    "sys_prompt": sys_prompt,
                    "question": line["question"],
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
        states = text_qa_with_sys.run_batch(
            arguments,
            temperature=0,
            backend=runtime,
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
    parser.add_argument(
        "--prompt-templ-path", type=str, default="data/system_prompt.template"
    )
    parser.add_argument("--num-questions", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--disable-flashinfer", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
