import torch
import argparse
import ctypes
import triton
from kv_tree_simple import retrive_from_file
from flash_attn_wrap import qkv_preparation, flash_attn_decode
from fasttree import fasttree_preparation, fasttree_decode, FastTreeParams
from sglang_triton import sglang_triton_preparation, sglang_triton_decode
from flashinfer_wrap import flashinfer_preparation, flashinfer_decode
from DeFT import DeFT_preparation, DeFT_decode

_cudart = ctypes.CDLL("libcudart.so")


def cu_prof_start():
    ret = _cudart.cudaProfilerStart()
    if ret != 0:
        raise Exception("cudaProfilerStart() returned %d" % ret)


def cu_prof_stop():
    ret = _cudart.cudaProfilerStop()
    if ret != 0:
        raise Exception("cudaProfilerStop() returned %d" % ret)


def main(args):
    load_file_path = args.load_file_path

    num_qo_heads = args.num_qo_heads
    num_kv_heads = args.num_kv_heads
    head_dim = args.head_dim
    dtype = torch.float16

    quantiles = [0.5, 0.2, 0.8]
    Q_TILE_SIZE_PER_PHASE = list(map(int, args.q_tile_size_per_phase.split(",")))
    KV_TILE_SIZE_PER_PHASE = list(map(int, args.kv_tile_size_per_phase.split(",")))
    KV_SPLIT_SIZES = list(map(int, args.kv_split_sizes.split(",")))
    para_threshs1 = list(map(int, args.para_threshs1.split(",")))
    para_threshs2 = list(map(int, args.para_threshs2.split(",")))

    params = FastTreeParams()
    params.set_values(args.alpha, args.beta, args.gamma)
    params.set_q_tile_sizes(Q_TILE_SIZE_PER_PHASE)
    params.set_kv_tile_sizes(KV_TILE_SIZE_PER_PHASE)
    params.set_kv_group_num(num_qo_heads // num_kv_heads)

    tree_info = retrive_from_file(load_file_path)

    Q, K_cache, V_cache, cache_seqlens, K_tree_tensor, V_tree_tensor, KV_ptrs = (
        qkv_preparation(tree_info, num_qo_heads, num_kv_heads, head_dim, "cuda", dtype)
    )
    batch_size = Q.shape[0]

    def run_and_bench(func, name):
        if args.signature:
            name = args.signature
        out = func()
        ms, _, _ = triton.testing.do_bench(func, quantiles=quantiles)
        if args.profile:
            cu_prof_start()
            func()
            cu_prof_stop()
        print(f"{name} time:", ms)
        return out

    out_flash_attn = None
    out_tree_attn = None
    out_sglang_triton_attn = None
    out_sglang_flashinfer_attn = None
    out_DeFT_attn = None

    # ============================
    # 1) flash_attn
    # ============================
    if args.choice in ["flash_attn", "all"]:
        flash_attn_func = lambda: flash_attn_decode(Q, K_cache, V_cache, cache_seqlens)
        out_flash_attn = run_and_bench(flash_attn_func, "flash attn")

        # 后处理
        out_flash_attn = out_flash_attn.view(
            -1, out_flash_attn.shape[-2], out_flash_attn.shape[-1]
        ).cpu()

    # ============================
    # 2) fasttree
    # ============================
    if args.choice in ["fasttree", "all"]:
        fasttree_aux, node_assignment = fasttree_preparation(
            tree_info,
            KV_ptrs,
            batch_size,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            KV_SPLIT_SIZES,
            para_threshs1,
            para_threshs2,
            params,
        )

        if args.fasttree_heuristic_path:
            with open(args.fasttree_heuristic_path, "w") as file:
                for i in range(len(tree_info)):
                    file.write(
                        f"Node {i}: parent {tree_info[i].parent}, assignment {node_assignment[i]}\n"
                    )

        sm_scale = 1.0 / (head_dim**0.5)
        out_tree_attn = torch.empty(
            batch_size, num_qo_heads, head_dim, device="cuda", dtype=dtype
        )

        def call_tree_attn():
            fasttree_decode(
                Q,
                K_tree_tensor,
                V_tree_tensor,
                out_tree_attn,
                *fasttree_aux,
                Q_TILE_SIZE_PER_PHASE,
                KV_TILE_SIZE_PER_PHASE,
                sm_scale,
            )
            return out_tree_attn

        out_tree_attn = run_and_bench(call_tree_attn, "fasttree attn")
        out_tree_attn = out_tree_attn.cpu()

    # ============================
    # 3) sglang_triton
    # ============================
    if args.choice in ["sglang_triton", "all"]:
        out_sglang_triton_attn = torch.empty(
            batch_size, num_qo_heads, head_dim, device="cuda", dtype=dtype
        )
        sglang_triton_aux, att_m = sglang_triton_preparation(
            K_cache, cache_seqlens, num_qo_heads
        )

        sm_scale = 1.0 / (head_dim**0.5)

        def call_sglang_triton():
            sglang_triton_decode(
                Q,
                K_cache.view(-1, K_cache.shape[-2], K_cache.shape[-1]),
                V_cache.view(-1, V_cache.shape[-2], V_cache.shape[-1]),
                out_sglang_triton_attn,
                *sglang_triton_aux,
                sm_scale,
                att_m=att_m,
            )
            return out_sglang_triton_attn

        out_sglang_triton_attn = run_and_bench(call_sglang_triton, "sglang triton")
        out_sglang_triton_attn = out_sglang_triton_attn.cpu()

    # ============================
    # 4) flashinfer
    # ============================
    if args.choice in ["sglang_flashinfer", "all"]:
        decode_wrapper = flashinfer_preparation(num_qo_heads, K_cache, cache_seqlens)
        sm_scale = 1.0 / (head_dim**0.5)

        def call_sglang_flashinfer():
            return flashinfer_decode(
                decode_wrapper,
                Q,
                K_cache.view(-1, K_cache.shape[-2], K_cache.shape[-1]),
                V_cache.view(-1, V_cache.shape[-2], V_cache.shape[-1]),
                sm_scale,
            )

        out_sglang_flashinfer_attn = run_and_bench(
            call_sglang_flashinfer, "sglang flashinfer"
        )
        out_sglang_flashinfer_attn = out_sglang_flashinfer_attn.cpu()

    # ============================
    # 5) DeFT
    # ============================
    if args.choice in ["DeFT", "all"]:
        subtree_len = 128
        mask_len = 64
        Q_TILE_SIZE = Q_TILE_SIZE_PER_PHASE[0]
        KV_TILE_SIZE = KV_TILE_SIZE_PER_PHASE[0]

        DeFT_aux = DeFT_preparation(
            tree_info, K_cache, subtree_len, mask_len, num_qo_heads, head_dim
        )
        out_DeFT_attn = torch.empty(
            batch_size, num_qo_heads, head_dim, device="cuda", dtype=dtype
        )
        sm_scale = 1.0 / (head_dim**0.5)

        def call_DeFT_attn():
            DeFT_decode(
                Q,
                K_cache.view(-1, K_cache.shape[-2], K_cache.shape[-1]),
                V_cache.view(-1, V_cache.shape[-2], V_cache.shape[-1]),
                out_DeFT_attn,
                *DeFT_aux,
                Q_TILE_SIZE,
                KV_TILE_SIZE,
                sm_scale,
                mask_len,
            )
            return out_DeFT_attn

        out_DeFT_attn = run_and_bench(call_DeFT_attn, "DeFT attn")
        out_DeFT_attn = out_DeFT_attn.cpu()

    # ============================
    # 6) Check results
    # ============================
    if out_flash_attn is not None and out_tree_attn is not None:
        if torch.allclose(out_flash_attn, out_tree_attn, rtol=1e-2, atol=1e-2):
            print("fasttree attn Check passed!")
        else:
            print("fasttree attn Check failed!")

    if out_flash_attn is not None and out_sglang_triton_attn is not None:
        if torch.allclose(out_flash_attn, out_sglang_triton_attn, rtol=1e-2, atol=1e-2):
            print("sglang triton attn Check passed!")
        else:
            print("sglang triton attn Check failed!")

    if out_flash_attn is not None and out_sglang_flashinfer_attn is not None:
        if torch.allclose(
            out_flash_attn, out_sglang_flashinfer_attn, rtol=1e-2, atol=1e-2
        ):
            print("sglang flashinfer attn Check passed!")
        else:
            print("sglang flashinfer attn Check failed!")

    if out_flash_attn is not None and out_DeFT_attn is not None:
        if torch.allclose(out_flash_attn, out_DeFT_attn, rtol=1e-2, atol=1e-2):
            print("DeFT attn Check passed!")
        else:
            print("DeFT attn Check failed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_file_path", type=str, required=True)
    parser.add_argument("--alpha", type=float, default=0.66)
    parser.add_argument("--beta", type=float, default=0.33)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--q_tile_size_per_phase", type=str, default="64,16")
    parser.add_argument("--kv_tile_size_per_phase", type=str, default="32,32")
    parser.add_argument("--kv_split_sizes", type=str, default="1024,128")
    parser.add_argument("--para_threshs1", type=str, default="132,528")
    parser.add_argument("--para_threshs2", type=str, default="132,132")
    parser.add_argument("--fasttree_heuristic_path", type=str, default=None)
    parser.add_argument("--num_qo_heads", type=int, default=32)
    parser.add_argument("--num_kv_heads", type=int, default=32)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--signature", type=str, default=None)
    parser.add_argument(
        "--choice",
        type=str,
        choices=[
            "flash_attn",
            "fasttree",
            "sglang_triton",
            "sglang_flashinfer",
            "DeFT",
            "all",
        ],
        default="all",
    )
    parser.add_argument("--profile", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
