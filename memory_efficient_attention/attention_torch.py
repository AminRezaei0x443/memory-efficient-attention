import torch
from torch.utils.checkpoint import checkpoint
from .utils import dynamic_slice, map_pt, scan
import math


def _query_chunk_attention(query, key, value, mask, bias, key_chunk_size=4096, return_attentions=False):
    num_kv, num_heads, k_features = key.shape[-3:]
    v_features = value.shape[-1]
    num_q = query.shape[-3]
    key_chunk_size = min(key_chunk_size, num_kv)
    query = query / math.sqrt(k_features)

    def summarize_chunk(query, key, value, mask, bias):
        attn_weights = torch.einsum('...qhd,...khd->...qhk', query, key)
        if bias is not None:
            bias = torch.einsum('...hqk->...qhk', bias)
            attn_weights = attn_weights + bias
        if mask is not None:
            big_neg = torch.finfo(attn_weights.dtype).min
            big_neg = torch.tensor(big_neg, dtype=torch.float32)
            mask = torch.einsum('...hqk->...qhk', mask)
            attn_weights = torch.where(mask, attn_weights, big_neg)
        max_score, _ = torch.max(attn_weights, -1, keepdim=True)
        max_score = max_score.detach()
        exp_weights = torch.exp(attn_weights - max_score)
        exp_values = torch.einsum('...vhf,...qhv->...qhf', value, exp_weights)
        max_score = torch.einsum('...qhk->...qh', max_score)
        return exp_values, exp_weights, exp_weights.sum(dim=-1), max_score

    def chunk_scanner(chunk_idx):
        key_chunk = dynamic_slice(key, tuple([0] * (key.ndim - 3)) + (chunk_idx, 0, 0),
                                  tuple(key.shape[:-3]) + (key_chunk_size, num_heads, k_features))
        value_chunk = dynamic_slice(value, tuple([0] * (value.ndim - 3)) + (chunk_idx, 0, 0),
                                    tuple(value.shape[:-3]) + (key_chunk_size, num_heads, v_features))
        if bias is not None:
            bias_chunk = dynamic_slice(bias, tuple([0] * (bias.ndim - 3)) + (0, 0, chunk_idx),
                                       tuple(bias.shape[:-3]) + (num_heads, num_q, key_chunk_size))
        else:
            bias_chunk = None
        if mask is not None:
            mask_chunk = dynamic_slice(mask, tuple([0] * (mask.ndim - 3)) + (0, 0, chunk_idx),
                                       tuple(mask.shape[:-3]) + (num_heads, num_q, key_chunk_size))
        else:
            mask_chunk = None
        return checkpoint(summarize_chunk, query, key_chunk, value_chunk, mask_chunk, bias_chunk)

    chunk_values, chunk_attentions, chunk_weights, chunk_max = map_pt(
        chunk_scanner, xs=torch.arange(0, num_kv, key_chunk_size))

    global_max, _ = torch.max(chunk_max, 0, keepdim=True)
    max_diffs = torch.exp(chunk_max - global_max)
    chunk_values *= torch.unsqueeze(max_diffs, -1)
    chunk_weights *= max_diffs

    all_values = chunk_values.sum(dim=0)
    all_weights = torch.unsqueeze(chunk_weights, -1).sum(dim=0)
    if return_attentions:
        all_attentions = torch.cat(tuple(chunk_attentions), dim=-1) / all_weights
    else:
        all_attentions = None
    return all_values / all_weights, all_attentions


def efficient_dot_product_attention(query, key, value,
                                    mask=None, bias=None,
                                    query_chunk_size=1024,
                                    key_chunk_size=4096,
                                    return_attentions=False):
    """Computes efficient dot-product attention given query, key, and value.
      This is efficient version of attention presented in
      https://arxiv.org/abs/2112.05682v2 which comes with O(sqrt(n)) memory requirements.
      Note: query, key, value needn't have any batch dimensions.
      Args:
        query: queries for calculating attention with shape of
          `[batch..., q_length, num_heads, qk_depth_per_head]`.
        key: keys for calculating attention with shape of
          `[batch..., kv_length, num_heads, qk_depth_per_head]`.
        value: values to be used in attention with shape of
          `[batch..., kv_length, num_heads, v_depth_per_head]`.
        bias: bias for the attention weights. This should be broadcastable to the
          shape `[batch..., num_heads, q_length, kv_length]`.
          This can be used for incorporating causal masks, padding masks,
          proximity bias, etc.
        mask: mask for the attention weights. This should be broadcastable to the
          shape `[batch..., num_heads, q_length, kv_length]`.
          This can be used for incorporating causal masks.
          Attention weights are masked out if their corresponding mask value
          is `False`.
        query_chunk_size: int: query chunks size
        key_chunk_size: int: key chunks size
        return_attentions: If specified, a tuple of (output, weights) will be returned.
      Returns:
        Output of shape `[batch..., q_length, num_heads, v_depth_per_head]`.
      """
    num_q, num_heads, q_features = query.shape[-3:]
    num_kv = key.shape[-3]

    def chunk_scanner(chunk_idx, _):
        print(chunk_idx)
        query_chunk = dynamic_slice(query, tuple([0] * (query.ndim - 3)) + (chunk_idx, 0, 0),
                                    tuple(query.shape[:-3]) + (min(query_chunk_size, num_q), num_heads, q_features))
        if mask is not None:
            mask_chunk = dynamic_slice(mask, tuple([0] * (mask.ndim - 3)) + (0, chunk_idx, 0),
                                       tuple(mask.shape[:-3]) + (num_heads, min(query_chunk_size, num_q), num_kv))
        else:
            mask_chunk = None
        if bias is not None:
            bias_chunk = dynamic_slice(bias, tuple([0] * (bias.ndim - 3)) + (0, chunk_idx, 0),
                                       tuple(bias.shape[:-3]) + (num_heads, min(query_chunk_size, num_q), num_kv))
        else:
            bias_chunk = None
        out, attn_chunk = _query_chunk_attention(query_chunk, key, value, mask_chunk, bias_chunk, key_chunk_size=key_chunk_size, return_attentions=return_attentions)
        if return_attentions:
            out = (out, attn_chunk)
        return (chunk_idx + query_chunk_size, out)

    _, res = scan(chunk_scanner, init=0, xs=None, length=math.ceil(num_q / query_chunk_size))
    if return_attentions:
        return torch.cat(list(res[0]), dim=-3), torch.cat(list(res[1]), dim=-3)
    else:
        return torch.cat(list(res), dim=-3)
