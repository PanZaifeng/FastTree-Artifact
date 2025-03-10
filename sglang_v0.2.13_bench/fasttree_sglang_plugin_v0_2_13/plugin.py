import torch
import nvtx
from types import MethodType

from fasttree_sglang_plugin_v0_2_13.attn_kernels import fasttree_decode
from fasttree_sglang_plugin_v0_2_13.metadata import FastTreeMetadata


def _replace_radix_attn():
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import InputMetadata

    old_radix_init = RadixAttention.__init__

    def decode_forward_fasttree(
        self: RadixAttention,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        input_metadata: InputMetadata,
    ):
        if self.qk_head_dim != self.v_head_dim:
            o = q.new_empty((q.shape[-1], self.tp_q_head_num * self.v_head_dim))
        else:
            o = torch.empty_like(q)
        self.store_kv_cache(k, v, input_metadata)

        fasttree_decode(
            q.view(-1, self.tp_q_head_num, self.qk_head_dim),
            input_metadata.token_to_kv_pool.get_key_buffer(self.layer_id),
            input_metadata.token_to_kv_pool.get_value_buffer(self.layer_id),
            o.view(-1, self.tp_q_head_num, self.v_head_dim),
            input_metadata.req_to_token_pool.req_to_token,
            input_metadata.fasttree.vnode_to_kv_offs,
            input_metadata.fasttree.vnode_to_kv_lens,
            input_metadata.fasttree.vnode_to_q_entries,
            input_metadata.fasttree.vnode_to_q_offs,
            input_metadata.fasttree.vnode_to_q_lens,
            input_metadata.fasttree.req_to_vnode_entries,
            input_metadata.fasttree.req_to_vnode_offs,
            input_metadata.fasttree.req_to_vnode_lens,
            input_metadata.fasttree.mid_o,
            input_metadata.fasttree.mid_lse,
            input_metadata.fasttree.phase_node_nums,
            input_metadata.fasttree.phase_node_offsets,
            input_metadata.fasttree.phase_q_tile_sizes,
            input_metadata.fasttree.phase_kv_tile_sizes,
            sm_scale=self.scaling,
            logit_cap=self.logit_cap,
        )
        return o

    def new_radix_init(self: RadixAttention, *args, **kwargs):
        old_radix_init(self, *args, **kwargs)
        # ============================
        # FastTree Modification Begin
        # ============================
        self.decode_forward = MethodType(decode_forward_fasttree, self)
        # ============================
        # FastTree Modification End
        # ============================

    RadixAttention.__init__ = new_radix_init


def _replace_tp_server_fwd(enable_fasttree=True, annotate_breakdown=False):
    from sglang.srt.managers.tp_worker import ModelTpServer, global_config
    import fasttree_sglang_plugin_v0_2_13.schedule as fasttree_schedule

    if annotate_breakdown:

        def annotate_function(module, func_name):
            original_func = getattr(module, func_name)

            def new_func(*args, **kwargs):
                torch.cuda.synchronize()
                with nvtx.annotate(func_name):
                    result = original_func(*args, **kwargs)
                torch.cuda.synchronize()
                return result

            setattr(module, func_name, new_func)

        annotate_function(ModelTpServer, "forward_prefill_batch")
        annotate_function(ModelTpServer, "forward_decode_batch")
        if enable_fasttree:
            annotate_function(fasttree_schedule, "prepare_for_fasttree_meta")
            annotate_function(fasttree_schedule, "prepare_for_fasttree_step")

    # Adapted from ModelTpServer.forward_step
    @torch.inference_mode()
    def new_tp_server_forward_step(self: ModelTpServer):
        new_batch = self.get_new_prefill_batch()

        if new_batch is not None:
            # Run a new prefill batch
            self.forward_prefill_batch(new_batch)

            if not new_batch.is_empty():
                if self.running_batch is None:
                    self.running_batch = new_batch
                else:
                    self.running_batch.merge(new_batch)
        else:
            # Run a decode batch
            if self.running_batch is not None:

                # ============================
                # FastTree Modification Begin
                # ============================
                if enable_fasttree:
                    fasttree_schedule.prepare_for_fasttree_meta(
                        self.running_batch, self.fasttree_metadata
                    )
                    init_bs = len(self.running_batch.reqs)
                # ============================
                # FastTree Modification End
                # ============================

                # Run a few decode batches continuously for reducing overhead
                for _ in range(global_config.num_continue_decode_steps):
                    self.num_generated_tokens += len(self.running_batch.reqs)
                    self.forward_decode_batch(self.running_batch)

                    # ============================
                    # FastTree Modification Begin
                    # ============================
                    if enable_fasttree:
                        if len(self.running_batch.reqs) < init_bs:
                            break

                        if _ < global_config.num_continue_decode_steps - 1:
                            fasttree_schedule.prepare_for_fasttree_step(
                                self.running_batch, self.fasttree_metadata
                            )
                    # ============================
                    # FastTree Modification End
                    # ============================

                    # Print stats
                    if self.tp_rank == 0 and self.decode_forward_ct % 40 == 0:
                        self.print_decode_stats()

                    if self.running_batch.is_empty():
                        self.running_batch = None
                        break

                    if self.out_pyobjs and self.running_batch.has_stream():
                        break
            else:
                self.check_memory()
                self.new_token_ratio = global_config.init_new_token_ratio

    ModelTpServer.forward_step = new_tp_server_forward_step


def _replace_fwd_batch_info():
    from sglang.srt.model_executor.forward_batch_info import InputMetadata, ForwardMode
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.managers.schedule_batch import ScheduleBatch

    old_meta_from_batch = InputMetadata.from_schedule_batch.__func__

    def new_meta_from_batch(
        cls: type[InputMetadata],
        model_runner: ModelRunner,
        batch: ScheduleBatch,
        forward_mode: ForwardMode,
    ):
        ret = old_meta_from_batch(cls, model_runner, batch, forward_mode)
        # ============================
        # FastTree Modification Begin
        # ============================
        ret.fasttree = getattr(batch, "fasttree_metadata", None)
        # ============================
        # FastTree Modification End
        # ============================
        return ret

    InputMetadata.from_schedule_batch = classmethod(new_meta_from_batch)


def _delayed_load_fasttree_plugin(enable_fasttree=True, annotate_breakdown=False):
    _replace_tp_server_fwd(enable_fasttree, annotate_breakdown)
    if enable_fasttree:
        _replace_fwd_batch_info()
        _replace_radix_attn()


def _load_plugin_and_replace_tp_server_init(
    enable_fasttree=True, annotate_breakdown=False
):
    from sglang.srt.managers.tp_worker import ModelTpServer, ServerArgs

    old_tp_server_init = ModelTpServer.__init__

    def new_tp_server_init(
        self: ModelTpServer,
        gpu_id: int,
        tp_rank: int,
        server_args: ServerArgs,
        nccl_port: int,
        model_overide_args: dict,
    ):
        # Apply the plugin for each TP server
        # We cannot apply it too early due to the CUDA re-init issues in MP
        _delayed_load_fasttree_plugin(enable_fasttree, annotate_breakdown)

        # Currently, FastTree is not compatible with CUDA Graph
        if enable_fasttree:
            server_args.disable_cuda_graph = True

        old_tp_server_init(
            self, gpu_id, tp_rank, server_args, nccl_port, model_overide_args
        )

        # ============================
        # FastTree Modification Begin
        # ============================
        if enable_fasttree:
            self.fasttree_metadata = FastTreeMetadata(
                self.model_config.num_attention_heads,
                self.model_config.num_key_value_heads,
                self.model_config.head_dim,
                torch.device("cuda", self.gpu_id),
            )
        # ============================
        # FastTree Modification End
        # ============================

    ModelTpServer.__init__ = new_tp_server_init


def load_fasttree_plugin(enable_fasttree=True, annotate_breakdown=False):
    if not enable_fasttree and not annotate_breakdown:
        print("[Warning] FastTree plugin is not enabled")
        return
    _load_plugin_and_replace_tp_server_init(enable_fasttree, annotate_breakdown)
