import torch
import triton
import triton.language as tl
from kv_tree_simple import KVTreeNode


class KVTreeSubnode(KVTreeNode):
    def __init__(self):
        super().__init__()
        self.start = -1
        self.end = -1


def _generate_subnode(node, start, end):
    sn = KVTreeSubnode()
    sn.id = node.id
    sn.parent = node.parent
    sn.num_children = node.num_children
    sn.requests = node.requests
    sn.seqlen = end - start
    sn.start = start
    sn.end = end
    return sn


cur_length = 0


def _group_subtree(Tree_info, subtree, node_id, cur_pos, subtree_len):
    global cur_length
    if cur_length + Tree_info[node_id].seqlen - cur_pos >= subtree_len:
        sn = _generate_subnode(
            Tree_info[node_id], cur_pos, cur_pos + subtree_len - cur_length
        )
        subtree[len(subtree) - 1].append(sn)
        subtree.append([])

        cur_pos += subtree_len - cur_length
        cur_length = 0

        if cur_pos == Tree_info[node_id].seqlen:
            for node in Tree_info:
                if node.parent == node_id:
                    _group_subtree(Tree_info, subtree, node.id, 0, subtree_len)
        else:
            _group_subtree(Tree_info, subtree, node_id, cur_pos, subtree_len)
        return

    sn = _generate_subnode(Tree_info[node_id], cur_pos, Tree_info[node_id].seqlen)
    subtree[len(subtree) - 1].append(sn)
    cur_length += Tree_info[node_id].seqlen - cur_pos
    for node in Tree_info:
        if node.parent == node_id:
            _group_subtree(Tree_info, subtree, node.id, 0, subtree_len)


def DeFT_preparation(Tree_info, K_cache, subtree_len, mask_len, num_qo_heads, head_dim):
    batch_size = K_cache.shape[0]
    max_seqlen = K_cache.shape[1]
    node_num = len(Tree_info)

    total_token = 0
    for node in Tree_info:
        total_token += node.seqlen

    subtree_num = (total_token - 1) // subtree_len + 1

    subtree = [[]]
    _group_subtree(Tree_info, subtree, 0, 0, subtree_len)

    if subtree[len(subtree) - 1] == []:
        subtree.pop()
    if len(subtree) != subtree_num:
        assert 0, "Group subtree error!"

    vnode_to_kv_entries = [[] for _ in range(subtree_num)]
    vnode_to_q_entries = [[] for _ in range(subtree_num)]
    causalmask = [[] for _ in range(subtree_num)]

    for subtree_id, subt in enumerate(subtree):
        Qset = []
        now_kvlen = 0

        for sn in subt:
            Qset = Qset + sn.requests
            prefix_len = sn.start
            node = sn.parent
            while node != -1:
                prefix_len += Tree_info[node].seqlen
                node = Tree_info[node].parent

            for i in range(sn.end - sn.start):
                vnode_to_kv_entries[subtree_id].append(
                    prefix_len + i + sn.requests[0] * max_seqlen
                )
                now_kvlen = now_kvlen + 1

        Qset = list(set(Qset))
        Qset.sort()
        vnode_to_q_entries[subtree_id] = vnode_to_q_entries[subtree_id] + Qset

        # causalmask
        for mask_num in range((len(Qset) - 1) // mask_len + 1):
            for sn in subt:
                cur_mask = 0
                for req_num in range(min(mask_len, len(Qset) - mask_num * mask_len)):
                    req_id = mask_num * mask_len + req_num
                    if Qset[req_id] in sn.requests:
                        cur_mask = cur_mask | (1 << req_num)

                if mask_len == 64 and cur_mask & (1 << 63):
                    cur_mask -= 1 << 64

                for i in range(sn.end - sn.start):
                    causalmask[subtree_id].append(cur_mask)

    vnode_to_kv_offs = [0]
    vnode_to_kv_lens = []
    vnode_to_q_offs = [0]
    vnode_to_q_lens = []
    causalmask_offset = [0]
    for i in range(subtree_num):
        vnode_to_kv_lens.append(len(vnode_to_kv_entries[i]))
        if i != subtree_num - 1:
            vnode_to_kv_offs.append(vnode_to_kv_offs[i] + vnode_to_kv_lens[i])

        vnode_to_q_lens.append(len(vnode_to_q_entries[i]))
        if i != subtree_num - 1:
            vnode_to_q_offs.append(vnode_to_q_offs[i] + vnode_to_q_lens[i])

        if i != subtree_num - 1:
            causalmask_offset.append(causalmask_offset[i] + len(causalmask[i]))

    vnode_to_kv_entries = [item for sublist in vnode_to_kv_entries for item in sublist]
    vnode_to_q_entries = [item for sublist in vnode_to_q_entries for item in sublist]
    causalmask = [item for sublist in causalmask for item in sublist]

    req_to_vnode_entries = [[] for _ in range(batch_size)]
    req_to_vnode_offs = []
    req_to_vnode_lens = []

    for i, q in enumerate(vnode_to_q_entries):
        req_to_vnode_entries[q].append(i)

    last_off = 0
    for i in range(batch_size):
        req_to_vnode_offs.append(last_off)
        last_off = last_off + len(req_to_vnode_entries[i])
        req_to_vnode_lens.append(len(req_to_vnode_entries[i]))

    req_to_vnode_entries = [
        item for sublist in req_to_vnode_entries for item in sublist
    ]

    vnode_to_kv_entries = vnode_to_kv_entries + [-1] * 32
    req_to_vnode_entries = req_to_vnode_entries + [-1] * 32

    with torch.device("cuda"):
        vnode_to_kv_entries = torch.tensor(vnode_to_kv_entries, dtype=torch.int32)
        vnode_to_kv_offs = torch.tensor(vnode_to_kv_offs, dtype=torch.int32)
        vnode_to_kv_lens = torch.tensor(vnode_to_kv_lens, dtype=torch.int32)
        vnode_to_q_entries = torch.tensor(vnode_to_q_entries, dtype=torch.int32)
        vnode_to_q_offs = torch.tensor(vnode_to_q_offs, dtype=torch.int32)
        vnode_to_q_lens = torch.tensor(vnode_to_q_lens, dtype=torch.int32)
        causalmask = torch.tensor(causalmask, dtype=torch.int64)
        causalmask_offset = torch.tensor(causalmask_offset, dtype=torch.int32)
        req_to_vnode_entries = torch.tensor(req_to_vnode_entries, dtype=torch.int32)
        req_to_vnode_offs = torch.tensor(req_to_vnode_offs, dtype=torch.int32)
        req_to_vnode_lens = torch.tensor(req_to_vnode_lens, dtype=torch.int32)
        mid_o = torch.empty(
            (vnode_to_q_entries.numel(), num_qo_heads, head_dim), dtype=torch.float32
        )
        mid_lse = torch.empty(
            (vnode_to_q_entries.numel(), num_qo_heads), dtype=torch.float32
        )

    return (
        vnode_to_kv_entries,
        vnode_to_kv_offs,
        vnode_to_kv_lens,
        vnode_to_q_entries,
        vnode_to_q_offs,
        vnode_to_q_lens,
        causalmask,
        causalmask_offset,
        req_to_vnode_entries,
        req_to_vnode_offs,
        req_to_vnode_lens,
        mid_o,
        mid_lse,
    )


@triton.jit
def _fwd_DeFT_decode_stage1(
    Q,
    K,
    V,
    stride_q0,
    stride_q1,
    stride_k0,
    stride_k1,
    stride_v0,
    stride_v1,
    stride_o0,
    stride_o1,
    stride_lse0,
    vnode_to_kv_entries,
    vnode_to_kv_offs,
    vnode_to_kv_lens,
    vnode_to_q_entries,
    vnode_to_q_offs,
    vnode_to_q_lens,
    causalmask,
    causalmask_offset,
    sm_scale,
    MidO,
    MidLSE,
    head_dim: tl.constexpr,
    kv_group_num: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    KV_TILE_SIZE: tl.constexpr,
    mask_len: tl.constexpr,
):
    cur_subtree = tl.program_id(0)
    cur_kv_head = tl.program_id(1)
    Q_BLOCK_SIZE: tl.constexpr = Q_TILE_SIZE * kv_group_num

    offs_d = tl.arange(0, head_dim)
    offs_g = tl.arange(0, kv_group_num)

    cur_q_entry_start = tl.load(vnode_to_q_offs + cur_subtree)
    cur_q_len = tl.load(vnode_to_q_lens + cur_subtree)

    cur_kv_entry_start = tl.load(vnode_to_kv_offs + cur_subtree)
    cur_kv_len = tl.load(vnode_to_kv_lens + cur_subtree)
    cur_qo_head = cur_kv_head * kv_group_num

    cur_causalmask_off = tl.load(causalmask_offset + cur_subtree)

    offs_q_tile = tl.arange(0, Q_TILE_SIZE)
    for cur_Block_Qnode in range(tl.cdiv(cur_q_len, Q_TILE_SIZE)):
        offs_q_tile_new = cur_Block_Qnode * Q_TILE_SIZE + offs_q_tile
        q_entries = tl.load(
            vnode_to_q_entries + cur_q_entry_start + offs_q_tile_new,
            mask=offs_q_tile_new < cur_q_len,
            other=-1,
        )
        off_q = (
            q_entries[:, None, None] * stride_q0
            + (cur_qo_head + offs_g[None, :, None]) * stride_q1
            + offs_d[None, :]
        )
        q = tl.load(Q + off_q, mask=q_entries[:, None, None] >= 0, other=0.0)
        q = q.reshape([Q_BLOCK_SIZE, head_dim])

        acc = tl.zeros([Q_BLOCK_SIZE, head_dim], dtype=tl.float32)
        sum_exp = tl.zeros([Q_BLOCK_SIZE, 1], dtype=tl.float32)
        max_logics = tl.full([Q_BLOCK_SIZE, 1], float("-inf"), dtype=tl.float32)
        offs_kv_tile = tl.arange(0, KV_TILE_SIZE)
        for cur_kv_tile in range(tl.cdiv(cur_kv_len, KV_TILE_SIZE)):
            offs_kv_tile_new = cur_kv_tile * KV_TILE_SIZE + offs_kv_tile
            # kv_entries = tl.load(
            #     vnode_to_kv_entries + cur_kv_entry_start + offs_kv_tile_new,
            #     mask=offs_kv_tile_new < cur_kv_len,
            #     other=-1,
            # )
            kv_entries = tl.load(
                vnode_to_kv_entries + cur_kv_entry_start + offs_kv_tile_new
            ).cast(tl.int64)
            kv_entries = tl.where(offs_kv_tile_new < cur_kv_len, kv_entries, -1)
            off_k = (
                kv_entries[None, :] * stride_k0
                + cur_kv_head * stride_k1
                + offs_d[:, None]
            )
            k = tl.load(K + off_k, mask=kv_entries[None, :] >= 0, other=0.0)
            qk = tl.dot(q, k)
            qk *= sm_scale

            offs_causalmask = (
                cur_kv_len * ((cur_Block_Qnode * Q_TILE_SIZE) // mask_len)
                + cur_kv_tile * KV_TILE_SIZE
                + offs_kv_tile
            )
            cur_causalmask = tl.load(
                causalmask + cur_causalmask_off + offs_causalmask,
                mask=cur_kv_tile * KV_TILE_SIZE + offs_kv_tile < cur_kv_len,
                other=0,
            )
            row_index = (
                cur_Block_Qnode * Q_TILE_SIZE
                - cur_Block_Qnode * Q_TILE_SIZE // mask_len * mask_len
                + tl.arange(0, Q_TILE_SIZE)[:, None]
            )
            bitmask = (cur_causalmask >> row_index) & 1
            bitmask = bitmask[:, None, :].broadcast_to(
                [Q_TILE_SIZE, kv_group_num, KV_TILE_SIZE]
            )
            bitmask = bitmask.reshape([Q_BLOCK_SIZE, KV_TILE_SIZE])
            qk = tl.where(bitmask == 0, -1.0e6, qk)

            off_v = (
                kv_entries[:, None] * stride_v0
                + cur_kv_head * stride_v1
                + offs_d[None, :]
            )
            v = tl.load(V + off_v, mask=kv_entries[:, None] >= 0, other=0.0)

            cur_max_logics = tl.max(qk, axis=1)[:, None]
            new_max_logics = tl.maximum(cur_max_logics, max_logics)
            exp_logics = tl.exp(qk - new_max_logics)
            logic_scale = tl.exp(max_logics - new_max_logics)

            sum_exp = sum_exp * logic_scale + tl.sum(exp_logics, axis=1)[:, None]
            acc = acc * logic_scale + tl.dot(exp_logics.to(tl.float16), v)
            max_logics = new_max_logics

        off_mid_o = (
            (cur_q_entry_start + offs_q_tile_new[:, None, None]) * stride_o0
            + (cur_qo_head + offs_g[None, :, None]) * stride_o1
            + offs_d[None, None, :]
        )
        off_mid_lse = (
            (cur_q_entry_start + offs_q_tile_new[:, None]) * stride_lse0
            + cur_qo_head
            + offs_g[None, :]
        )

        mid_o = (acc / sum_exp).reshape([Q_TILE_SIZE, kv_group_num, head_dim])
        tl.store(
            MidO + off_mid_o,
            tl.cast(mid_o, dtype=tl.float16),
            mask=offs_q_tile_new[:, None, None] < cur_q_len,
        )

        mid_lse = (tl.log(sum_exp) + max_logics).reshape([Q_TILE_SIZE, kv_group_num])
        tl.store(
            MidLSE + off_mid_lse, mid_lse, mask=offs_q_tile_new[:, None] < cur_q_len
        )


def _DeFT_attn_stage_1(
    Q,
    K,
    V,
    vnode_to_kv_entries,
    vnode_to_kv_offs,
    vnode_to_kv_lens,
    vnode_to_q_entries,
    vnode_to_q_offs,
    vnode_to_q_lens,
    causalmask,
    causalmask_offset,
    sm_scale,
    MidO,
    MidLSE,
    subtree_num,
    head_num,
    head_dim,
    kv_head_num,
    Q_TILE_SIZE,
    KV_TILE_SIZE,
    mask_len,
):
    Lq, Lk = Q.shape[-1], K.shape[-1]
    assert Lq == Lk
    assert Lk in {16, 32, 64, 128, 256}

    grid = (
        subtree_num,
        kv_head_num,
    )

    kv_group_num = head_num // kv_head_num

    _fwd_DeFT_decode_stage1[grid](
        Q,
        K,
        V,
        Q.stride(0),
        Q.stride(1),
        K.stride(0),
        K.stride(1),
        V.stride(0),
        V.stride(1),
        MidO.stride(0),
        MidO.stride(1),
        MidLSE.stride(0),
        vnode_to_kv_entries,
        vnode_to_kv_offs,
        vnode_to_kv_lens,
        vnode_to_q_entries,
        vnode_to_q_offs,
        vnode_to_q_lens,
        causalmask,
        causalmask_offset,
        sm_scale,
        MidO,
        MidLSE,
        head_dim,
        kv_group_num,
        Q_TILE_SIZE,
        KV_TILE_SIZE,
        mask_len,
    )


@triton.jit
def _fwd_DeFT_decode_stage2(
    req_to_vnode_entries,
    req_to_vnode_offs,
    req_to_vnode_lens,
    MidO,
    MidLSE,
    O,
    stride_mid_o0,
    stride_mid_o1,
    stride_mid_lse0,
    stride_o0,
    stride_o1,
    head_dim: tl.constexpr,
    NODE_TILE_SIZE: tl.constexpr,
):
    cur_req = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_vnode_entry_start = tl.load(req_to_vnode_offs + cur_req)
    cur_vnode_len = tl.load(req_to_vnode_lens + cur_req)

    sum_exp = tl.zeros([1], dtype=tl.float32)
    max_lse = tl.full([1], float("-inf"), dtype=tl.float32)
    acc = tl.zeros([1, head_dim], dtype=tl.float32)

    offs_d = tl.arange(0, head_dim)
    offs_n = tl.arange(0, NODE_TILE_SIZE)
    for tile_start in range(0, cur_vnode_len, NODE_TILE_SIZE):
        offs_n_new = offs_n + tile_start
        # vnode_entries = tl.load(
        #     req_to_vnode_entries + cur_vnode_entry_start + offs_n_new,
        #     mask=offs_n_new < cur_vnode_len,
        #     other=-1,
        # )
        vnode_entries = tl.load(
            req_to_vnode_entries + cur_vnode_entry_start + offs_n_new
        )
        vnode_entries = tl.where(offs_n_new < cur_vnode_len, vnode_entries, -1)

        off_mid_o = (
            vnode_entries[:, None] * stride_mid_o0
            + cur_head * stride_mid_o1
            + offs_d[None, :]
        )
        mid_o = tl.load(MidO + off_mid_o, mask=vnode_entries[:, None] >= 0, other=0.0)

        off_mid_lse = vnode_entries * stride_mid_lse0 + cur_head
        mid_lse = tl.load(
            MidLSE + off_mid_lse, mask=vnode_entries >= 0, other=float("-inf")
        )[None, :]

        tile_max_lse = tl.max(mid_lse, axis=1)
        new_max_lse = tl.maximum(tile_max_lse, max_lse)

        exp_lse = tl.exp(mid_lse - new_max_lse)
        old_scale = tl.exp(max_lse - new_max_lse)
        sum_exp = sum_exp * old_scale + tl.sum(exp_lse, axis=1)
        acc = acc * old_scale + tl.sum(exp_lse[:, :, None] * mid_o[None, :, :], axis=1)
        max_lse = new_max_lse

    tl.store(
        O + cur_req * stride_o0 + cur_head * stride_o1 + offs_d,
        tl.cast(tl.ravel(acc / sum_exp), dtype=tl.float16),
    )


def _DeFT_decode_stage2(
    req_to_vnode_entries,
    req_to_vnode_offs,
    req_to_vnode_lens,
    MidO,
    MidLSE,
    O,
    batch_size,
    head_num,
    head_dim,
):
    NODE_TILE_SIZE = 4
    grid = (
        batch_size,
        head_num,
    )
    _fwd_DeFT_decode_stage2[grid](
        req_to_vnode_entries,
        req_to_vnode_offs,
        req_to_vnode_lens,
        MidO,
        MidLSE,
        O,
        MidO.stride(0),
        MidO.stride(1),
        MidLSE.stride(0),
        O.stride(0),
        O.stride(1),
        head_dim,
        NODE_TILE_SIZE,
    )


def DeFT_decode(
    q,
    k_buffer,
    v_buffer,
    o,
    vnode_to_kv_entries,
    vnode_to_kv_offs,
    vnode_to_kv_lens,
    vnode_to_q_entries,
    vnode_to_q_offs,
    vnode_to_q_lens,
    causalmask,
    causalmask_offset,
    req_to_vnode_entries,
    req_to_vnode_offs,
    req_to_vnode_lens,
    mid_o,
    mid_lse,
    Q_TILE_SIZE,
    KV_TILE_SIZE,
    sm_scale,
    mask_len,
    logit_cap=-1,
):
    subtree_num = vnode_to_q_offs.shape[0]
    head_num = q.shape[1]
    head_dim = q.shape[-1]
    kv_head_num = k_buffer.shape[1]

    _DeFT_attn_stage_1(
        q,
        k_buffer,
        v_buffer,
        vnode_to_kv_entries,
        vnode_to_kv_offs,
        vnode_to_kv_lens,
        vnode_to_q_entries,
        vnode_to_q_offs,
        vnode_to_q_lens,
        causalmask,
        causalmask_offset,
        sm_scale,
        mid_o,
        mid_lse,
        subtree_num,
        head_num,
        head_dim,
        kv_head_num,
        Q_TILE_SIZE,
        KV_TILE_SIZE,
        mask_len,
    )

    batch_size = q.shape[0]
    _DeFT_decode_stage2(
        req_to_vnode_entries,
        req_to_vnode_offs,
        req_to_vnode_lens,
        mid_o,
        mid_lse,
        o,
        batch_size,
        head_num,
        head_dim,
    )
