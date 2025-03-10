import torch
import queue
from collections import defaultdict
from typing import Dict, List, Tuple

from sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode
from sglang.srt.managers.schedule_batch import ScheduleBatch

from fasttree_sglang_plugin_v0_2_13.metadata import FastTreeMetadata


def prepare_for_fasttree_meta(
    batch: ScheduleBatch, fasttree_metadata: FastTreeMetadata
):
    assert isinstance(batch.tree_cache, RadixCache)
    root_node = batch.tree_cache.root_node

    node_to_reqs: Dict[TreeNode, List[int]] = defaultdict(list)
    for rid, req in enumerate(batch.reqs):
        node = root_node
        output_index = 0 - len(req.origin_input_ids)
        while output_index < len(req.output_ids):
            token_index = output_index + len(req.origin_input_ids)
            token = (
                req.output_ids[output_index]
                if output_index >= 0
                else req.origin_input_ids[token_index]
            )
            child = node.children.get(token)
            if child:
                node_to_reqs[child].append(rid)
                node = child
                output_index += len(child.key)
            else:
                break

    alpha = fasttree_metadata.alpha
    beta = fasttree_metadata.beta
    gamma = fasttree_metadata.gamma
    num_kv_heads = fasttree_metadata.num_kv_heads
    kv_group_num = fasttree_metadata.kv_group_num
    phase_q_tile_sizes = fasttree_metadata.TSQs
    phase_kv_tile_sizes = fasttree_metadata.TSKs
    phase_kv_split_sizes = [fasttree_metadata.kv_split_sizes[0] for _ in range(2)]

    def CpadQ(TS, N):
        return TS - ((N - 1) % TS + 1)

    def CpadK(TS, N):
        return max(0, TS - N)

    def Cmm(nQ, nK):
        phase = 0 if nQ > phase_q_tile_sizes[1] else 1
        TSQ = phase_q_tile_sizes[phase]
        TSK = phase_kv_tile_sizes[phase]
        return (
            alpha * CpadQ(TSQ, nQ) * kv_group_num * nK
            + beta * CpadK(TSK, nK) * nQ * kv_group_num
        )

    def Cred(nQl, lenl, lenv):
        return gamma * nQl

    def SplitQCost(nQcurr, nQl, lenv, lenl):
        return Cmm(nQcurr - nQl, lenv) + Cmm(nQl, lenl + lenv)

    def SplitKCost(nQcurr, nQl, lenl, lenv):
        return (
            Cmm(nQcurr, lenv)
            + Cmm(nQl, lenl)
            - Cmm(nQl, lenv + lenl)
            + Cred(nQl, lenl, lenv)
        )

    nodes_in_bfs_order: List[TreeNode] = []
    node_token_offsets: List[int] = []
    node_que: queue.Queue[Tuple[TreeNode, int]] = queue.Queue()
    node_que.put((root_node, 0))
    while not node_que.empty():
        node, offset = node_que.get()
        nodes_in_bfs_order.append(node)
        node_token_offsets.append(offset)
        child_offset = offset + len(node.value)
        for child in node.children.values():
            if child in node_to_reqs.keys():
                node_que.put((child, child_offset))

    def traverse():
        vnode_num_aggr_reqs = [len(node_to_reqs[node]) for node in nodes_in_bfs_order]
        vnode_seqlens = [len(node.value) for node in nodes_in_bfs_order]
        node_assignments = [0 for _ in vnode_seqlens]
        parallelisms = [0, 0]
        for i, node in enumerate(nodes_in_bfs_order):
            nQcurr = vnode_num_aggr_reqs[i]
            lenv = vnode_seqlens[i]
            l = i
            for child in enumerate(node.children.values()):
                if child not in node_to_reqs.keys():
                    continue
                l = l + 1
                nQl = len(node_to_reqs[child])
                lenl = vnode_seqlens[l]
                C0 = SplitKCost(nQcurr, nQl, lenl, lenv)
                C1 = SplitQCost(nQcurr, nQl, lenv, lenl)
                if C0 > C1:
                    node_assignments[l] = 1
                    nQcurr -= nQl
                    vnode_seqlens[l] = lenl + lenv
                else:
                    node_assignments[l] = 0

            vnode_num_aggr_reqs[i] = nQcurr
            if nQcurr > 0:
                phase = 0 if nQcurr > phase_q_tile_sizes[1] else 1
                q_vnode_count = (nQcurr - 1) // phase_q_tile_sizes[phase] + 1
                kv_vnode_count = (lenv - 1) // phase_kv_tile_sizes[phase] + 1
                parallelisms[phase] += q_vnode_count * kv_vnode_count
        parallelisms = [p * num_kv_heads for p in parallelisms]
        return node_assignments, vnode_num_aggr_reqs, vnode_seqlens, parallelisms

    for i in range(3):
        node_assignments, vnode_num_aggr_reqs, vnode_seqlens, parallelisms = traverse()
        if i == 0:
            para_threshs = fasttree_metadata.para_threshs1
            break_flag = True
            for phase in range(2):
                if (
                    parallelisms[phase] > 0
                    and parallelisms[phase] < para_threshs[phase]
                ):
                    phase_kv_split_sizes[phase] = phase_kv_split_sizes[1]
                    break_flag = False
            if break_flag:
                break
        elif i == 1:
            para_threshs = fasttree_metadata.para_threshs2
            if parallelisms[0] > 0 and parallelisms[0] < para_threshs[0]:
                phase_q_tile_sizes = [phase_q_tile_sizes[1] for _ in range(2)]
                phase_kv_tile_sizes = [phase_q_tile_sizes[1] for _ in range(2)]
            elif parallelisms[1] > 0 and parallelisms[1] < para_threshs[1]:
                phase_q_tile_sizes = [phase_q_tile_sizes[0] for _ in range(2)]
                phase_kv_tile_sizes = [phase_q_tile_sizes[0] for _ in range(2)]
            else:
                break

    fasttree_metadata.phase_q_tile_sizes = phase_q_tile_sizes
    fasttree_metadata.phase_kv_tile_sizes = phase_kv_tile_sizes

    vnode_to_kv_offs = []
    vnode_to_kv_lens = []
    vnode_to_q_entries = []
    vnode_to_q_offs = []
    vnode_to_q_lens = []
    req_to_vnode_entries = [[] for _ in range(len(batch.reqs))]
    req_to_vnode_offs = []
    req_to_vnode_lens = []
    req_indices = batch.req_pool_indices.cpu()
    req_to_token_stride = batch.req_to_token_pool.req_to_token.stride(0)

    for i, (node, num_aggr_reqs, seqlen, token_offset) in enumerate(
        zip(
            nodes_in_bfs_order,
            vnode_num_aggr_reqs,
            vnode_seqlens,
            node_token_offsets,
        )
    ):
        if num_aggr_reqs == 0:
            continue

        aggregated_nodes: List[TreeNode] = []
        l = i
        for child in node.children.values():
            if child not in node_to_reqs.keys():
                continue
            l = l + 1
            if node_assignments[l] == 0:
                aggregated_nodes.append(child)
        if len(aggregated_nodes) == 0:
            aggregated_nodes.append(node)
        first_req = node_to_reqs[aggregated_nodes[0]][0]

        kv_offset_start = (
            req_to_token_stride * int(req_indices[first_req]) + token_offset
        )

        phase = 0 if num_aggr_reqs > phase_q_tile_sizes[1] else 1
        q_split_size = phase_q_tile_sizes[phase]
        kv_split_size = phase_kv_split_sizes[phase]

        q_split_count = (num_aggr_reqs - 1) // q_split_size + 1
        kv_split_count = (seqlen - 1) // kv_split_size + 1
        for kv_split_id in range(kv_split_count):
            q_offset_start = len(vnode_to_q_entries)
            for n in aggregated_nodes:
                for req in node_to_reqs[n]:
                    vnode_to_q_entries.append(req)

            split_kv_off = kv_split_id * kv_split_size
            vnode_kv_len = min(split_kv_off + kv_split_size, seqlen) - split_kv_off

            for q_split_id in range(q_split_count):
                split_q_off = q_split_id * q_split_size
                vnode_q_len = (
                    min(split_q_off + q_split_size, num_aggr_reqs) - split_q_off
                )

                vnode_to_kv_offs.append(kv_offset_start + split_kv_off)
                vnode_to_kv_lens.append(vnode_kv_len)
                vnode_to_q_offs.append(q_offset_start + split_q_off)
                vnode_to_q_lens.append(vnode_q_len)

    kv_split_size = phase_kv_split_sizes[1]
    req_last_vnodes = [0 for _ in range(len(batch.reqs))]
    for rid, req in enumerate(batch.reqs):
        prefix_len = len(req.origin_input_ids)
        generated_len = len(req.output_ids) + 1
        kv_offset_start = req_to_token_stride * int(req_indices[rid]) + prefix_len
        for kv_split_id in range((generated_len - 1) // kv_split_size + 1):
            q_offset = len(vnode_to_q_entries)
            vnode_to_q_entries.append(rid)
            vnode_to_q_offs.append(q_offset)
            vnode_to_q_lens.append(1)

            split_kv_off = kv_split_id * kv_split_size
            vnode_kv_len = (
                min(split_kv_off + kv_split_size, generated_len) - split_kv_off
            )
            vnode_to_kv_offs.append(kv_offset_start + split_kv_off)
            vnode_to_kv_lens.append(vnode_kv_len)
        # the positions of these nodes are unchanged after reordering
        req_last_vnodes[rid] = len(vnode_to_kv_lens) - 1
    fasttree_metadata.req_last_vnodes = req_last_vnodes

    for i, req in enumerate(vnode_to_q_entries):
        req_to_vnode_entries[req].append(i)

    offset = 0
    for i in range(len(batch.reqs)):
        req_to_vnode_offs.append(offset)
        offset += len(req_to_vnode_entries[i])
        req_to_vnode_lens.append(len(req_to_vnode_entries[i]))

    req_to_vnode_entries = [
        item for sublist in req_to_vnode_entries for item in sublist
    ]

    threshold = phase_q_tile_sizes[1]
    above_indices = [i for i, val in enumerate(vnode_to_q_lens) if val > threshold]
    below_indices = [i for i, val in enumerate(vnode_to_q_lens) if val <= threshold]

    new_order = above_indices + below_indices
    fasttree_metadata.phase_node_nums = [len(above_indices), len(below_indices)]
    fasttree_metadata.phase_node_offsets = [0, len(above_indices)]

    vnode_to_q_lens = [vnode_to_q_lens[i] for i in new_order]
    vnode_to_q_offs = [vnode_to_q_offs[i] for i in new_order]
    vnode_to_kv_lens = [vnode_to_kv_lens[i] for i in new_order]
    vnode_to_kv_offs = [vnode_to_kv_offs[i] for i in new_order]

    def list_to_gpu_tensor(preallocated_gpu, cpu_list, dtype=torch.int32):
        cpu_tensor = torch.tensor(cpu_list, device="cpu", dtype=dtype)
        n = cpu_tensor.size(0)
        preallocated_gpu[:n].copy_(cpu_tensor, non_blocking=True)

    list_to_gpu_tensor(fasttree_metadata.vnode_to_q_entries, vnode_to_q_entries)
    list_to_gpu_tensor(fasttree_metadata.vnode_to_q_offs, vnode_to_q_offs)
    list_to_gpu_tensor(fasttree_metadata.vnode_to_q_lens, vnode_to_q_lens)
    list_to_gpu_tensor(fasttree_metadata.vnode_to_kv_offs, vnode_to_kv_offs)
    list_to_gpu_tensor(fasttree_metadata.vnode_to_kv_lens, vnode_to_kv_lens)
    list_to_gpu_tensor(fasttree_metadata.req_to_vnode_entries, req_to_vnode_entries)
    list_to_gpu_tensor(fasttree_metadata.req_to_vnode_offs, req_to_vnode_offs)
    list_to_gpu_tensor(fasttree_metadata.req_to_vnode_lens, req_to_vnode_lens)

    # Bind FastTree metadata to the batch
    batch.fasttree_metadata = fasttree_metadata


def prepare_for_fasttree_step(
    batch: ScheduleBatch, fasttree_metadata: FastTreeMetadata
):
    bs = len(batch.reqs)
    increment = torch.ones(bs, device=fasttree_metadata.device, dtype=torch.int32)
    fasttree_metadata.vnode_to_kv_lens[fasttree_metadata.req_last_vnodes] += increment
