import torch
import flash_attn


def qkv_preparation(
    tree_info, num_qo_heads, num_kv_heads, head_dim, device="cuda", dtype=torch.float16
):
    node_num = len(tree_info)
    num_requests = 0
    max_seqlen = 0
    K_tree_tensor = []
    V_tree_tensor = []
    KV_ptrs = [0]
    for n in range(node_num):
        seqlen = tree_info[n].seqlen
        K_tree_tensor.append(
            torch.randn(1, seqlen, num_kv_heads, head_dim, device=device, dtype=dtype)
        )
        V_tree_tensor.append(
            torch.randn(1, seqlen, num_kv_heads, head_dim, device=device, dtype=dtype)
        )
        KV_ptrs.append(KV_ptrs[-1] + seqlen)

        if tree_info[n].num_children == 0:
            node = n
            sum_seqlen = 0
            while node != -1:
                tree_info[node].requests.append(num_requests)
                sum_seqlen += tree_info[node].seqlen
                node = tree_info[node].parent
            if sum_seqlen > max_seqlen:
                max_seqlen = sum_seqlen
            num_requests += 1

    batch_size = num_requests
    K_cache = torch.empty(
        batch_size, max_seqlen, num_kv_heads, head_dim, device=device, dtype=dtype
    )
    V_cache = torch.empty(
        batch_size, max_seqlen, num_kv_heads, head_dim, device=device, dtype=dtype
    )
    cache_seqlens = [0 for _ in range(num_requests)]
    for n in range(node_num):
        if tree_info[n].num_children == 0:
            node = n
            chain_nodes = []
            while node != -1:
                chain_nodes.append(node)
                node = tree_info[node].parent

            K_temp_tensor = torch.empty(
                1, 0, num_kv_heads, head_dim, device=device, dtype=dtype
            )
            V_temp_tensor = torch.empty(
                1, 0, num_kv_heads, head_dim, device=device, dtype=dtype
            )
            for n in reversed(chain_nodes):
                K_temp_tensor = torch.cat((K_temp_tensor, K_tree_tensor[n]), dim=1)
                V_temp_tensor = torch.cat((V_temp_tensor, V_tree_tensor[n]), dim=1)

            cache_seqlens[tree_info[n].requests[0]] = K_temp_tensor.shape[1]
            K_cache[tree_info[n].requests[0], 0 : K_temp_tensor.shape[1]] = (
                K_temp_tensor
            )
            V_cache[tree_info[n].requests[0], 0 : V_temp_tensor.shape[1]] = (
                V_temp_tensor
            )

    Q = torch.randn(batch_size, num_qo_heads, head_dim, device=device, dtype=dtype)
    cache_seqlens = torch.tensor(cache_seqlens, dtype=torch.int32, device=device)

    K_tree_tensor = torch.cat(K_tree_tensor, dim=1).reshape(
        [-1, num_kv_heads, head_dim]
    )
    V_tree_tensor = torch.cat(V_tree_tensor, dim=1).reshape(
        [-1, num_kv_heads, head_dim]
    )

    return Q, K_cache, V_cache, cache_seqlens, K_tree_tensor, V_tree_tensor, KV_ptrs


def flash_attn_decode(Q, K_cache, V_cache, cache_seqlens):
    o = flash_attn.flash_attn_with_kvcache(
        torch.unsqueeze(Q, 1), K_cache, V_cache, cache_seqlens=cache_seqlens
    )
    return o
