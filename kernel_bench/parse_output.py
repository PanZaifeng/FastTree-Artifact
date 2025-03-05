#!/usr/bin/env python3
import os
import re
from pathlib import Path

OUTPUT_DIR = "outputs"

##############################################################################
# Helper Functions
##############################################################################

def extract_raw_time(pattern: str, lines: list[str]) -> str:
    """
    From the given list of lines, finds the first line starting with 'pattern'
    and returns the last field of that line.
    Returns an empty string if not found.
    """
    for line in lines:
        line_strip = line.strip()
        if line_strip.startswith(pattern):
            parts = line_strip.split()
            return parts[-1] if parts else ""
    return ""

def compute_speedup(t_flash: str, t_other: str) -> str:
    """
    Computes speedup = flash-attn time / other time if valid.
    Returns an empty string if invalid or zero division occurs.
    """
    if not t_flash:
        return ""
    try:
        f_val = float(t_flash)
        if f_val == 0.0:
            return ""
    except ValueError:
        return ""

    if not t_other:
        return ""
    try:
        o_val = float(t_other)
        if o_val == 0.0:
            return ""
    except ValueError:
        return ""

    return str(f_val / o_val)

def format_two_sig(val: str) -> str:
    """
    Converts the string to float, then formats it to two significant digits.
    Returns empty string if the input is invalid or empty.
    """
    if not val:
        return ""
    try:
        num = float(val)
        return f"{num:.2g}"  # two significant digits
    except ValueError:
        return ""

def float_gt(a: str, b: str) -> bool:
    """
    Returns True if the float in string 'a' is greater than
    the float in string 'b', otherwise False.
    """
    try:
        return float(a) > float(b)
    except ValueError:
        return False

##############################################################################
# Main Logic
##############################################################################

def main():
    # Print the Markdown table header
    print("| node-num-per-level | node-context-per-level | Query Heads | KV Heads | Flash-Attn | SGLang-Triton | FlashInfer | DeFT | MultiCascade | FastTree (naive) | FastTree | FastTree (naive 1) |")
    print("| ------------------ | ---------------------- | ----------- | -------- | ---------- | ------------- | ---------- | ---- | ------------ | ---------------- | -------- | ------------------ |")

    rows = []

    out_dir = Path(OUTPUT_DIR)
    if not out_dir.is_dir():
        print(f"Warning: directory '{OUTPUT_DIR}' does not exist.")
        return

    # Iterate through all .txt files in the OUTPUT_DIR
    for file_path in out_dir.glob("*.txt"):
        filename = file_path.name

        # Regex pattern for filenames: nc_<node-nums>_<node-contexts>_qk_<qHeads>_<kHeads>.txt
        match = re.match(r"^nc_([^_]+)_([^_]+)_qk_([^_]+)_([^\.]+)\.txt$", filename)
        if not match:
            # Skip files that do not follow the naming convention
            continue

        node_nums_str = match.group(1)      # e.g., "1,4,16"
        node_contexts_str = match.group(2)  # e.g., "1024,256,32"
        q_heads_str = match.group(3)        # e.g., "32"
        k_heads_str = match.group(4)        # e.g., "8"

        # Read file content
        lines = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

        # Extract raw times
        raw_flash_attn      = extract_raw_time("flash attn time:", lines)
        raw_sglang_triton   = extract_raw_time("sglang triton time:", lines)
        raw_flashinfer      = extract_raw_time("sglang flashinfer time:", lines)
        raw_deft            = extract_raw_time("DeFT attn time:", lines)
        raw_multicascade    = extract_raw_time("FlashInfer MutliLevelCascadeAttn time:", lines)
        raw_fasttree_naive  = extract_raw_time("fasttree -10 time:", lines)
        raw_fasttree_naive1 = extract_raw_time("fasttree 10 time:", lines)
        raw_fasttree        = extract_raw_time("fasttree attn time:", lines)

        # Compute speedups relative to flash-attn
        speedup_flash_attn = compute_speedup(raw_flash_attn, raw_flash_attn)
        if not speedup_flash_attn:
            # If flash-attn time is invalid, set all to empty
            f_flash_attn = ""
            f_sglang_triton = ""
            f_flashinfer = ""
            f_deft = ""
            f_multicascade = ""
            f_fasttree_naive = ""
            f_fasttree_naive1 = ""
            f_fasttree = ""
        else:
            # Compute relative speedups
            speedup_sglang_triton   = compute_speedup(raw_flash_attn, raw_sglang_triton)
            speedup_flashinfer      = compute_speedup(raw_flash_attn, raw_flashinfer)
            speedup_deft            = compute_speedup(raw_flash_attn, raw_deft)
            speedup_multicascade    = compute_speedup(raw_flash_attn, raw_multicascade)
            speedup_fasttree_naive  = compute_speedup(raw_flash_attn, raw_fasttree_naive)
            speedup_fasttree_naive1 = compute_speedup(raw_flash_attn, raw_fasttree_naive1)
            speedup_fasttree        = compute_speedup(raw_flash_attn, raw_fasttree)

            # Format to two significant digits
            f_flash_attn        = format_two_sig(speedup_flash_attn)
            f_sglang_triton     = format_two_sig(speedup_sglang_triton)
            f_flashinfer        = format_two_sig(speedup_flashinfer)
            f_deft              = format_two_sig(speedup_deft)
            f_multicascade      = format_two_sig(speedup_multicascade)
            f_fasttree_naive    = format_two_sig(speedup_fasttree_naive)
            f_fasttree_naive1   = format_two_sig(speedup_fasttree_naive1)
            f_fasttree          = format_two_sig(speedup_fasttree)

        # Identify the maximum speedup among certain methods
        methods = [
            f_flash_attn,
            f_sglang_triton,
            f_flashinfer,
            f_deft,
            f_multicascade,
            f_fasttree
        ]
        max_idx = -1
        max_val = ""
        for i, sp_val in enumerate(methods):
            if not sp_val:
                continue
            if not max_val:
                max_val = sp_val
                max_idx = i
            else:
                if float_gt(sp_val, max_val):
                    max_val = sp_val
                    max_idx = i

        # Bold the best speedup in the Markdown table
        out_flash_attn      = f_flash_attn
        out_sglang_triton   = f_sglang_triton
        out_flashinfer      = f_flashinfer
        out_deft            = f_deft
        out_multicascade    = f_multicascade
        out_fasttree        = f_fasttree
        out_fasttree_naive  = f_fasttree_naive
        out_fasttree_naive1 = f_fasttree_naive1

        if max_idx == 0 and out_flash_attn:
            out_flash_attn = f"**{out_flash_attn}**"
        elif max_idx == 1 and out_sglang_triton:
            out_sglang_triton = f"**{out_sglang_triton}**"
        elif max_idx == 2 and out_flashinfer:
            out_flashinfer = f"**{out_flashinfer}**"
        elif max_idx == 3 and out_deft:
            out_deft = f"**{out_deft}**"
        elif max_idx == 4 and out_multicascade:
            out_multicascade = f"**{out_multicascade}**"
        elif max_idx == 5 and out_fasttree:
            out_fasttree = f"**{out_fasttree}**"

        # Prepare the final Markdown row
        markdown_line = (
            f"| {node_nums_str} | {node_contexts_str} | {q_heads_str} | {k_heads_str} "
            f"| {out_flash_attn} | {out_sglang_triton} | {out_flashinfer} "
            f"| {out_deft} | {out_multicascade} | {out_fasttree_naive} "
            f"| {out_fasttree} | {out_fasttree_naive1} |"
        )

        # Parse node-num-per-level and node-context-per-level for sorting
        node_nums_list = [int(x) for x in node_nums_str.split(',') if x.isdigit()]
        node_contexts_list = [int(x) for x in node_contexts_str.split(',') if x.isdigit()]

        # Number of elements
        n_len = len(node_nums_list)
        c_len = len(node_contexts_list)

        # Parse Q heads and K heads as integers
        def extract_digits(s: str) -> int:
            digits = "".join(ch for ch in s if ch.isdigit())
            return int(digits) if digits else 0

        q_int = extract_digits(q_heads_str)
        k_int = extract_digits(k_heads_str)

        # Sort key structure:
        #   1) -q_int (desc)
        #   2) -k_int (desc)
        #   3) n_len (asc)
        #   4) each element of node_nums_list (asc)
        #   5) c_len (asc)
        #   6) each element of node_contexts_list (asc)
        #
        # Because Python sorts in ascending order, we use negative for descending fields.
        sort_tuple = (
            -q_int,
            -k_int,
            n_len,
            *node_nums_list,      # all ascending
            c_len,
            *node_contexts_list,  # all ascending
            markdown_line
        )

        rows.append(sort_tuple)

    # Sort the rows by the composite key (all but the last element),
    # then print the final Markdown lines (which is the last element in each tuple).
    rows.sort(key=lambda x: x[:-1])  # everything except the final markdown_line

    for row in rows:
        print(row[-1])

if __name__ == "__main__":
    main()