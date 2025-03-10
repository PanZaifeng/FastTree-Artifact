import sqlite3
import pandas as pd
import argparse

def analyze_nsys_sqlite(input_db):
    # Connect to SQLite database
    conn = sqlite3.connect(input_db)

    # Load StringIds (for kernel name mapping)
    string_ids_df = pd.read_sql_query("SELECT * FROM StringIds", conn)

    # Load NVTX events
    nvtx_df = pd.read_sql_query("SELECT * FROM NVTX_EVENTS", conn)

    # Load CUDA kernel execution data
    kernels_df = pd.read_sql_query("SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL", conn)

    # Load Memory operations (memcpy and memset)
    memcpy_df = pd.read_sql_query("SELECT * FROM CUPTI_ACTIVITY_KIND_MEMCPY", conn)
    memset_df = pd.read_sql_query("SELECT * FROM CUPTI_ACTIVITY_KIND_MEMSET", conn)

    # Load ENUM table to map memory copy types
    memcpy_enum_df = pd.read_sql_query("SELECT * FROM ENUM_CUDA_MEMCPY_OPER", conn)

    conn.close()

    # Step 1: Map NVTX event names
    nvtx_df = nvtx_df.merge(string_ids_df, left_on="textId", right_on="id", how="left")
    nvtx_df.rename(columns={"value": "event_name"}, inplace=True)

    # Step 2: Find the "End-to-end" NVTX event
    end_to_end_event = nvtx_df[nvtx_df["event_name"] == "End-to-end"]

    if len(end_to_end_event) != 1:
        raise ValueError(f"Expected 1 'End-to-end' NVTX event, but found {len(end_to_end_event)}.")

    start_time = end_to_end_event["start"].values[0]
    end_time = end_to_end_event["end"].values[0]
    total_time = (end_time - start_time) / 1e9  # Convert to seconds

    # Step 3: Filter NVTX events within the End-to-end range
    filtered_nvtx = nvtx_df[
        (nvtx_df["start"] >= start_time) & (nvtx_df["end"] <= end_time)
    ]

    # Step 4: Compute NVTX event absolute times in seconds
    forward_decode_time = (filtered_nvtx[filtered_nvtx["event_name"] == "forward_decode_batch"]["end"].sum() -
                           filtered_nvtx[filtered_nvtx["event_name"] == "forward_decode_batch"]["start"].sum()) / 1e9

    forward_prefill_time = (filtered_nvtx[filtered_nvtx["event_name"] == "forward_prefill_batch"]["end"].sum() -
                            filtered_nvtx[filtered_nvtx["event_name"] == "forward_prefill_batch"]["start"].sum()) / 1e9

    fasttree_meta_time = (filtered_nvtx[filtered_nvtx["event_name"] == "prepare_for_fasttree_meta"]["end"].sum() -
                          filtered_nvtx[filtered_nvtx["event_name"] == "prepare_for_fasttree_meta"]["start"].sum()) / 1e9

    fasttree_step_time = (filtered_nvtx[filtered_nvtx["event_name"] == "prepare_for_fasttree_step"]["end"].sum() -
                          filtered_nvtx[filtered_nvtx["event_name"] == "prepare_for_fasttree_step"]["start"].sum()) / 1e9

    fasttree_prepare_time = fasttree_meta_time + fasttree_step_time

    # Print NVTX event times
    print(f"[NVTX] forward_decode_batch {forward_decode_time:.6f} s")
    print(f"[NVTX] forward_prefill_batch {forward_prefill_time:.6f} s")
    print(f"[NVTX] prepare_for_fasttree {fasttree_prepare_time:.6f} s")
    print(f"[NVTX] end_to_end {total_time:.6f} s")

    # Step 5: Map kernel names
    kernels_df = kernels_df.merge(string_ids_df, left_on="demangledName", right_on="id", how="left")
    kernels_df.rename(columns={"value": "kernel_name"}, inplace=True)

    # Step 6: Filter kernel events within the End-to-end range
    filtered_kernels = kernels_df[
        (kernels_df["start"] >= start_time) & (kernels_df["end"] <= end_time)
    ].copy()

    # Step 7: Compute kernel execution absolute times in seconds
    categories = {
        "AttnDecode": ['_fwd_kernel_stage', 'BatchDecode', '_fwd_fasttree_decode_stage'],
        "AttnPrefill": ['_fwd_kernel', 'BatchPrefill'],
        "GEMM": ['GemmWithFusedEpilogue', 'GemmUniversal', 'gemm', 'cutlass', 'gemv', 'matmul', 'dense'],
        "Memory": []  # Memory operations will be included here
    }

    # Compute kernel execution times
    filtered_kernels.loc[:, "duration"] = (filtered_kernels["end"] - filtered_kernels["start"]) / 1e9
    total_kernel_time = filtered_kernels["duration"].sum()

    # Initialize category time dictionary
    category_times = {key: 0 for key in categories.keys()}
    category_times["Others"] = 0

    # Classify kernels
    for index, row in filtered_kernels.iterrows():
        kernel_name = str(row["kernel_name"]) if pd.notna(row["kernel_name"]) else ""
        duration = row["duration"]

        categorized = False
        for category, keywords in categories.items():
            if any(keyword in kernel_name for keyword in keywords):
                category_times[category] += duration
                categorized = True
                break

        if not categorized:
            category_times["Others"] += duration

    # Compute memory operation times
    memcpy_df = memcpy_df[
        (memcpy_df["start"] >= start_time) & (memcpy_df["end"] <= end_time)
    ].copy()
    memcpy_df.loc[:, "duration"] = (memcpy_df["end"] - memcpy_df["start"]) / 1e9

    memset_df = memset_df[
        (memset_df["start"] >= start_time) & (memset_df["end"] <= end_time)
    ].copy()
    memset_df.loc[:, "duration"] = (memset_df["end"] - memset_df["start"]) / 1e9

    # Map copyKind to human-readable labels
    memcpy_df = memcpy_df.merge(memcpy_enum_df, left_on="copyKind", right_on="id", how="left")

    # Compute total memory execution time and add to "Memory" category
    category_times["Memory"] = memset_df["duration"].sum() + memcpy_df["duration"].sum()

    # Print Kernel execution times (including Memory as part of Kernel)
    for category, time in sorted(category_times.items(), key=lambda x: x[1], reverse=True):
        print(f"[Kernel] {category} {time:.6f} s")
    print(f"[Kernel] total {sum(category_times.values()):.6f} s")

# Command-line argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze NVTX, Kernel, and Memory execution time from nsys SQLite report.")
    parser.add_argument("input_db", type=str, help="Path to the input SQLite database file (nsys report).")
    args = parser.parse_args()

    # Run analysis
    analyze_nsys_sqlite(args.input_db)