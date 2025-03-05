import torch

from sglang.srt.layers.decode_attention import decode_attention_fwd, REDUCE_TORCH_TYPE

sglang_triton_decode = decode_attention_fwd


def sglang_triton_preparation(K_cache, b_seq_len, num_qo_heads):
    batch_size = K_cache.shape[0]
    max_len_in_batch = K_cache.shape[1]
    b_req_idx = []
    b_start_loc = []
    total_num_tokens = 0

    req_to_token = [[] for _ in range(batch_size)]
    for i in range(batch_size):
        for j in range(max_len_in_batch):
            req_to_token[i].append(i * max_len_in_batch + j)

    for i in range(batch_size):
        b_req_idx.append(i)
        b_start_loc.append(total_num_tokens)
        total_num_tokens += int(b_seq_len[i])

    with torch.device("cuda"):
        req_to_token = torch.tensor(req_to_token)
        b_req_idx = torch.tensor(b_req_idx)
        b_start_loc = torch.tensor(b_start_loc)
        att_m = torch.empty((num_qo_heads, total_num_tokens), dtype=REDUCE_TORCH_TYPE)

    return (
        req_to_token,
        b_req_idx,
        b_start_loc,
        b_seq_len,
        max_len_in_batch,
        total_num_tokens,
    ), att_m
