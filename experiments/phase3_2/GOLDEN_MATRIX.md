# Phase 3 golden execution matrix

- tasks run: 60
- input generation PASS: 54
- official golden PASS: 54
- blocked: 6

## Passing tasks

| Task | Case | Seconds |
|---|---|---|
| add_rms_norm_dynamic_quant | 17 | 0.002 |
| aggregate_hidden | 1 | 0.005 |
| aggregate_hidden_grad | 18 | 0.001 |
| arg_max | 13 | 0.013 |
| attention_residual | 2 | 0.001 |
| categorical_embedding_bag | 6 | 0.012 |
| compressed_sparse_attention_core | 5 | 0.001 |
| conv_2d | 12 | 0.001 |
| conv_3d_backprop_filter | 14 | 0.001 |
| cross_entropy_loss | 7 | 0.002 |
| cummin | 13 | 0.005 |
| depthwise_conv_2d | 12 | 0.001 |
| dequant_swiglu_quant | 5 | 0.001 |
| dilation_2d | 14 | 0.001 |
| dynamic_mx_quant | 17 | 0.004 |
| dynamic_quant | 13 | 0.002 |
| embedding_hash_table_lookup_or_insert | 12 | 0.001 |
| engram_gate_fusion | 14 | 0.012 |
| exp | 12 | 0.003 |
| fft_conv | 13 | 0.013 |
| fused_causal_conv1d | 17 | 0.022 |
| gcd | 11 | 0.006 |
| gqa | 18 | 0.041 |
| grid_sampler_3d | 14 | 0.001 |
| grouped_matmul | 19 | 0.001 |
| grouped_matmul_swiglu_quant | 12 | 0.001 |
| image_resize_normalize | 2 | 0.001 |
| kimi_delta_attention | 5 | 0.001 |
| kv_cache_append | 10 | 0.001 |
| lact_ttt_chunk | 16 | 0.032 |
| mhc_post_grad | 5 | 0.001 |
| mhc_sinkhorn | 18 | 0.001 |
| mla | 7 | 0.156 |
| mla_prolog | 4 | 0.079 |
| moe_gating_top_k_softmax | 14 | 0.001 |
| nms | 16 | 0.003 |
| q4_0_dequant_matmul | 6 | 0.001 |
| quant_matmul | 18 | 0.001 |
| roi_align | 14 | 0.018 |
| scatter_block_update | 13 | 0.001 |
| selective_scan | 1 | 0.021 |
| sparse_flash_attention | 15 | 0.322 |
| sparse_lightning_indexer_grad_kl_loss_enhance | 2 | 0.026 |
| strided_slice | 16 | 0.001 |
| tensor_program_vm | 16 | 0.001 |
| top_k | 13 | 0.006 |
| transpose | 14 | 0.002 |
| unique | 13 | 0.036 |
| unsorted_segment_sum | 7 | 0.002 |
| weight_quant_batch_matmul | 4 | 0.005 |
| rmsnorm | 16 | 0.001 |
| swiglu | 13 | 0.0 |
| softmax | 10 | 0.002 |
| gather | 13 | 0.002 |

## Blocked tasks

| Task | Case | Reason |
|---|---|---|
| flash_attention_score_grad_enhance | 9 | RuntimeError: permute(sparse_coo): number of dimensions in the tensor input does not match the length of the d |
| linear_cross_entropy | None | no case fits the memory budget |
| lstm | 11 | TypeError: unsupported operand type(s) for *: 'int' and 'NoneType' |
| moe_re_routing | 20 | AssertionError: Sum of expert_token_num_per_rank (170) must equal A (64) |
| scatter | 14 | TypeError: scatter_() received an invalid combination of arguments - got (Tensor, Tensor, NoneType), but expec |
| apply_rotary_pos_emb | 14 | RuntimeError: The size of tensor a (17) must match the size of tensor b (13) at non-singleton dimension 2 |

Every run used one official case, executed on CPU with the official golden.
No case shape, dtype, range, or attribute was altered.
