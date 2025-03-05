import torch
from flashinfer import BatchDecodeWithPagedKVCacheWrapper


def flashinfer_preparation(num_qo_heads, K_cache, cache_seqlens):
    batch_size = K_cache.shape[0]
    max_seqlen = K_cache.shape[1]
    num_kv_heads = K_cache.shape[2]
    head_dim = K_cache.shape[3]
    kv_indptr = [0]
    kv_indices = []
    kv_last_page_len = [1 for _ in range(batch_size)]
    for i in range(batch_size):
        kv_indptr.append(kv_indptr[i] + cache_seqlens[i])
        for j in range(cache_seqlens[i]):
            kv_indices.append(i * max_seqlen + j)

    with torch.device("cuda"):
        kv_indptr = torch.tensor(kv_indptr, dtype=torch.int32)
        kv_indices = torch.tensor(kv_indices, dtype=torch.int32)
        kv_last_page_len = torch.tensor(kv_last_page_len, dtype=torch.int32)

    flashinfer_workspace_buffer = torch.empty(
        384 * 1024 * 1024, dtype=torch.uint8, device="cuda"
    )

    decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
        flashinfer_workspace_buffer, "NHD", use_tensor_cores=num_qo_heads > num_kv_heads
    )
    decode_wrapper.end_forward()
    decode_wrapper.begin_forward(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        1,
    )
    return decode_wrapper


def flashinfer_decode(decode_wrapper, Q, K_cache, V_cache, sm_scale):
    o = decode_wrapper.forward(
        Q, (K_cache, V_cache), sm_scale=sm_scale, logits_soft_cap=0
    )
    return o
