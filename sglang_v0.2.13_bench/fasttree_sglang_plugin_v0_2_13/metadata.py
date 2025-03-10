import torch
from typing import Tuple, List


class FastTreeMetadata:
    def __init__(
        self,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        device: str = "cuda",
        alpha: float = 0.6,
        beta: float = 0.3,
        gamma: float = 0.1,
        TSQs: Tuple[int, int] = None,
        TSKs: Tuple[int, int] = (32, 32),
        kv_split_sizes: Tuple[int, int] = (1024, 128),
        para_threshs1: Tuple[int, int] = (132, 132),
        para_threshs2: Tuple[int, int] = (132, 528),
        max_vnodes: int = 16384,
        max_vnode_to_q_entries: int = 81920,
        max_batch_size: int = 8192,
    ):
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.kv_group_num = num_qo_heads // num_kv_heads
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.TSQs = TSQs
        if self.TSQs is None:
            self.TSQs = (64 // self.kv_group_num, 16 // self.kv_group_num)
        self.TSKs = TSKs
        self.kv_split_sizes = kv_split_sizes
        self.para_threshs1 = para_threshs1
        self.para_threshs2 = para_threshs2

        with torch.device(device):
            self.vnode_to_q_entries = torch.empty(
                max_vnode_to_q_entries, dtype=torch.int32
            )
            self.vnode_to_q_offs = torch.empty(max_vnodes, dtype=torch.int32)
            self.vnode_to_q_lens = torch.empty(max_vnodes, dtype=torch.int32)
            self.vnode_to_kv_offs = torch.empty(max_vnodes, dtype=torch.int32)
            self.vnode_to_kv_lens = torch.empty(max_vnodes, dtype=torch.int32)
            self.req_to_vnode_entries = torch.empty(
                max_vnode_to_q_entries, dtype=torch.int32
            )
            self.req_to_vnode_offs = torch.empty(max_batch_size, dtype=torch.int32)
            self.req_to_vnode_lens = torch.empty(max_batch_size, dtype=torch.int32)
            self.mid_o = torch.empty(
                (max_vnode_to_q_entries, num_qo_heads, head_dim), dtype=torch.float32
            )
            self.mid_lse = torch.empty(
                (max_vnode_to_q_entries, num_qo_heads), dtype=torch.float32
            )

        self.phase_node_nums: Tuple[int, int] = None
        self.phase_node_offsets: Tuple[int, int] = None
        self.phase_q_tile_sizes: Tuple[int, int] = None
        self.phase_kv_tile_sizes: Tuple[int, int] = None
        self.req_last_vnodes: List[int] = None

        print("FastTreeMetadata initialized")
