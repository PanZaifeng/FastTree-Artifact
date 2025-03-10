import argparse
import random
import time
import nvtx
import os
import multiprocessing as mp

import sglang as sgl
from sglang.utils import read_jsonl
from fasttree_sglang_plugin_v0_2_13 import load_fasttree_plugin

random.seed(0)

enable_fasttree = os.getenv("ENABLE_FASTTREE", "0") == "1"
annotate_breakdown = os.getenv("ANNOTATE_BREAKDOWN", "0") == "1"
if enable_fasttree or annotate_breakdown:
    load_fasttree_plugin(enable_fasttree, annotate_breakdown)

# Adapted from https://github.com/sgl-project/sglang/blob/v0.2.13/benchmark/multi_document_qa/bench_sglang.py


@sgl.function
def multi_document_qa(s, docs, question, max_tokens):
    s += sgl.user_begin()
    s += "Pleaes answer a question according to given documents.\n"
    s += "Documents begin.\n"

    for doc in docs:
        s += doc
    s += "\nDocuments end."
    s += (
        "\n\nBased on the above documents, please answer this question:\n"
        + question
        + "\nAnswer in three words or fewer."
    )
    s += sgl.user_end()
    s += sgl.assistant(sgl.gen("answer", max_tokens=max_tokens))


def rand_docs(input_docs, max_doc_id=16, num_docs=4):
    doc_ids = sorted(random.sample(range(max_doc_id + 1), num_docs))
    return [input_docs[doc_id] for doc_id in doc_ids]


def main(args):
    lines = list(read_jsonl(args.data_path))
    l = lines[0]
    arguments = []
    max_questions = len(l["questions"])
    for i in range(args.num_doc_branches):
        docs = rand_docs(l["documents"])
        for j in range(args.per_branch_questions):
            arguments.append(
                {
                    "docs": docs,
                    "question": f"Q{j}: " + l["questions"][j % max_questions],
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
        states = multi_document_qa.run_batch(
            arguments,
            backend=runtime,
            temperature=0,
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
    parser.add_argument("--data-path", type=str, default="data/questions.jsonl")
    parser.add_argument("--num-doc-branches", type=int, default=16)
    parser.add_argument("--per-branch-questions", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--parallel", type=int, default=64)
    parser.add_argument("--disable-flashinfer", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
