import argparse
import ctypes
import torch
import triton

try:
    import flashinfer
except ImportError:
    raise ImportError("Please install flashinfer to use MultiLevelCascadeAttentionWrapper.")


_cudart = ctypes.CDLL("libcudart.so")


def cu_prof_start():
    """Start CUDA profiler."""
    ret = _cudart.cudaProfilerStart()
    if ret != 0:
        raise RuntimeError(f"cudaProfilerStart() returned {ret}")


def cu_prof_stop():
    """Stop CUDA profiler."""
    ret = _cudart.cudaProfilerStop()
    if ret != 0:
        raise RuntimeError(f"cudaProfilerStop() returned {ret}")


def build_bfs_kv_indices(node_pages_per_level, node_num_per_level):
    """
    Build BFS-based KV indices.
    """
    num_levels = len(node_pages_per_level)
    nodes = []
    for node_num in node_num_per_level:
        nodes.append([None] * node_num)

    current_index = [0]

    def dfs(level, pos):
        start = current_index[0]
        length = node_pages_per_level[level]
        end = start + length
        slice_for_node = list(range(start, end))
        nodes[level][pos] = slice_for_node
        current_index[0] = end

        if level < num_levels - 1:
            stride = node_num_per_level[level + 1] // node_num_per_level[level]
            for i in range(stride):
                dfs(level + 1, pos * stride + i)

    dfs(0, 0)

    result = []
    for lvl in range(num_levels):
        concat_for_level = []
        for pos in range(node_num_per_level[lvl]):
            concat_for_level.extend(nodes[lvl][pos])
        result.append(torch.tensor(concat_for_level, dtype=torch.int32, device="cuda:0"))

    return result


def main():
    parser = argparse.ArgumentParser(description="MultiLevelCascadeAttention Example")
    parser.add_argument(
        "--node_num_per_level",
        type=str,
        default="1,4,8",
        help="Comma-separated numbers of nodes per level (e.g. '1,4,8').",
    )
    parser.add_argument(
        "--node_seqlen_per_level",
        type=str,
        default="128,128,128",
        help="Comma-separated sequence lengths per level (e.g. '128,128,128').",
    )
    parser.add_argument(
        "--num_qo_heads",
        type=int,
        default=32,
        help="Number of QO heads.",
    )
    parser.add_argument(
        "--num_kv_heads",
        type=int,
        default=32,
        help="Number of KV heads.",
    )

    args = parser.parse_args()

    # Parse levels
    node_num_per_level = list(map(int, args.node_num_per_level.split(",")))
    node_seqlen_per_level = list(map(int, args.node_seqlen_per_level.split(",")))
    num_qo_heads = args.num_qo_heads
    num_kv_heads = args.num_kv_heads

    # Fixed hyperparameters (can be turned into arguments as needed)
    num_layers = 1
    head_dim = 128
    page_size = 1

    node_pages_per_level = [ns // page_size for ns in node_seqlen_per_level]
    pages_per_level = [
        node_pages * node_num
        for node_pages, node_num in zip(node_pages_per_level, node_num_per_level)
    ]
    total_num_pages = sum(pages_per_level)
    batch_size = node_num_per_level[-1]
    num_levels = len(node_num_per_level)

    # Allocate workspace (1GB as an example)
    workspace_buffer = torch.empty(1024 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")

    wrapper = flashinfer.MultiLevelCascadeAttentionWrapper(num_levels, workspace_buffer, "NHD")

    kv_page_indices_arr = build_bfs_kv_indices(node_pages_per_level, node_num_per_level)

    kv_page_indptr_arr = []
    for node_pages, node_num in zip(node_pages_per_level, node_num_per_level):
        kv_page_indptr_arr.append(
            torch.arange(node_num + 1, device="cuda:0", dtype=torch.int32) * node_pages
        )

    kv_last_page_len_arr = []
    for node_num in node_num_per_level:
        kv_last_page_len_arr.append(
            torch.full((node_num,), page_size, dtype=torch.int32, device="cuda:0")
        )

    kv_cache_at_layer = [
        torch.randn(
            total_num_pages, 2, page_size, num_kv_heads, head_dim,
            dtype=torch.float16, device="cuda:0"
        )
        for _ in range(num_layers)
    ]

    # Example Qo index pointer
    qo_indptr_arr = []
    for node_num in node_num_per_level:
        qo_indptr_arr.append(
            torch.linspace(0, batch_size, node_num + 1, dtype=torch.int32, device="cuda:0")
        )

    # print("qo_indptr_arr:", qo_indptr_arr)
    # print("kv_page_indices_arr:", kv_page_indices_arr)
    # print("kv_page_indptr_arr:", kv_page_indptr_arr)
    # print("kv_last_page_len_arr:", kv_last_page_len_arr)

    wrapper.plan(
        qo_indptr_arr,
        kv_page_indptr_arr,
        kv_page_indices_arr,
        kv_last_page_len_arr,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
    )

    q_arr = [
        torch.randn(batch_size, num_qo_heads, head_dim, dtype=torch.float16, device="cuda:0")
        for _ in range(num_layers)
    ]

    outputs = []
    cu_prof_start()
    for i in range(num_layers):
        q = q_arr[i]
        o = wrapper.run(q, kv_cache_at_layer[i])
        outputs.append(o)
    cu_prof_stop()

    t, _, _ = triton.testing.do_bench(
        lambda: wrapper.run(q_arr[0], kv_cache_at_layer[0]),
        quantiles=[0.5, 0.2, 0.8],
        warmup=10
    )
    print("FlashInfer MutliLevelCascadeAttn time:", t)


if __name__ == "__main__":
    main()