"""Latent attention and a jointly trained DeepSeek-style sparse indexer.

The FM uses unordered, bidirectional attribute sets: no causal mask or positional
RoPE is introduced. Keys and values share a normalized low-rank representation.
The indexer follows weighted ReLU dot-product scoring and detached-attention KL
supervision. One model trace/optimizer trains both paths; there is no staged fit.
"""
from __future__ import annotations

from contextvars import ContextVar
from functools import wraps

import jax
import jax.numpy as jnp
import numpyro

_context = ContextVar("strategize_attention_trace", default=None)


from strategize_transformer import rms_norm, run_layers


def masked_softmax(scores, mask):
    """Finite zero output and zero gradients for an entirely masked query."""
    scores = scores.astype(jnp.float32)
    masked = jnp.where(mask, scores, jnp.float32(-1e30))
    exp = jnp.where(mask, jnp.exp(masked - jnp.max(masked, -1, keepdims=True)), 0.)
    return exp / jnp.maximum(jnp.sum(exp, -1, keepdims=True), 1e-30)


def indexer_scores(x, query_latent, params, cfg):
    """The indexer receives gradients through its own weights only."""
    x, query_latent = jax.lax.stop_gradient(x), jax.lax.stop_gradient(query_latent)
    heads, width = int(cfg["indexer_heads"]), int(cfg["indexer_dim"])
    q = (query_latent @ params["W_index_q"].astype(query_latent.dtype)).reshape(*x.shape[:-1], heads, width)
    k = (x @ params["W_index_k"].astype(x.dtype)).astype(jnp.float32)
    k = (k - k.mean(-1, keepdims=True)) * jax.lax.rsqrt(k.var(-1, keepdims=True) + 1e-6)
    k = k * params["LN_index_k"] + params["b_index_k"]
    weights = (x @ params["W_index_w"].astype(x.dtype)).astype(jnp.float32) / (heads ** .5)
    logits = jnp.einsum("bqhd,bkd->bqhk", q.astype(jnp.float32), k) / (width ** .5)
    return jnp.sum(jax.nn.relu(logits) * weights[..., None], axis=2)


def _indexer_kl(probabilities, scores, mask, query_weight):
    target = jax.lax.stop_gradient(jnp.mean(probabilities, axis=2))
    predicted = masked_softmax(scores, mask)
    terms = jnp.where(target > 0, target * (jnp.log(jnp.maximum(target, 1e-30)) -
                                          jnp.log(jnp.maximum(predicted, 1e-30))), 0.)
    return jnp.sum(jnp.sum(terms, -1) * query_weight), jnp.sum(query_weight)


def mla_attention(x, mask, params, cfg, n_heads, head_dim, *, collect_loss=False,
                  row_weight=None):
    """Return attention output, summed indexer KL, and its valid-query weight.

    Absorbed keys and shared latent values avoid gathering a separate KV tensor
    per head. Q/K head normalization matches the expanded computation exactly.
    Boundary ties include all tied keys via a dense fallback, preserving set
    permutation equivariance rather than selecting by arbitrary token position.
    """
    p, cfg = dict(params), dict(cfg)
    n_heads, head_dim = int(n_heads), int(head_dim)
    valid = jnp.ones(x.shape[:2], bool) if mask is None else jnp.asarray(mask) > 0
    x = jnp.where(valid[..., None], x, 0)
    qlow = rms_norm(x @ p["W_q"].astype(x.dtype), p["RMS_q_latent"])
    latent = rms_norm(x @ p["W_k"].astype(x.dtype), p["RMS_kv_latent"])
    q = (qlow @ p["W_q_up"].astype(x.dtype)).reshape(*x.shape[:2], n_heads, head_dim)
    ku, vu = jnp.split(p["W_v"], 2, axis=-1)
    ku, vu = (w.reshape(latent.shape[-1], n_heads, head_dim) for w in (ku, vu))
    if "RMS_q" in p:
        q = rms_norm(q, p["RMS_q"])
    inverse_key_norm = jnp.ones((*x.shape[:2], n_heads), jnp.float32)
    if "RMS_k" in p:
        key = jnp.einsum("bkc,chd->bkhd", latent, ku.astype(latent.dtype)).astype(jnp.float32)
        inverse_key_norm = jax.lax.rsqrt(jnp.mean(key * key, -1) + 1e-6)
        ku = ku * p["RMS_k"]
    absorbed_q = jnp.einsum("bqhd,chd->bqhc", q.astype(jnp.float32), ku.astype(jnp.float32))
    query_weight = valid.astype(jnp.float32)
    if row_weight is not None:
        query_weight = query_weight * jnp.asarray(row_weight, jnp.float32)[:, None]
    dsa = cfg["architecture"] == "mla_dsa"
    scores = indexer_scores(x, qlow, p, cfg) if dsa else None
    pair_valid = valid[:, :, None] & valid[:, None, :]
    length, top_k = x.shape[1], min(int(cfg["top_k"]), x.shape[1])

    def finish(prob, values):
        context = jnp.einsum("bqhc,chd->bqhd", values, vu.astype(jnp.float32)).reshape(x.shape)
        out = context.astype(x.dtype) @ p["W_o"].astype(x.dtype)
        return jnp.where(valid[..., None], out, 0)

    def dense(selected_mask):
        logits = jnp.einsum("bqhc,bkc->bqhk", absorbed_q, latent.astype(jnp.float32))
        logits = logits * inverse_key_norm.transpose(0, 2, 1)[:, None, :, :] / (head_dim ** .5)
        probs = masked_softmax(logits, selected_mask[:, :, None, :])
        values = jnp.einsum("bqhk,bkc->bqhc", probs, latent.astype(jnp.float32))
        loss, count = _indexer_kl(probs, scores, selected_mask, query_weight) if collect_loss and dsa else (jnp.float32(0), jnp.float32(0))
        return finish(probs, values), loss, count

    if not dsa or top_k >= length:
        return dense(pair_valid)
    masked_scores = jnp.where(pair_valid, scores, jnp.float32(-1e30))
    top_values, top_indices = jax.lax.top_k(masked_scores, top_k + 1)
    indices = top_indices[..., :top_k]
    boundary_tie = ((top_values[..., top_k - 1] == top_values[..., top_k]) &
                    (top_values[..., top_k] > -1e29) & valid)

    def sparse(_):
        batch = jnp.arange(x.shape[0])[:, None, None]
        chosen_latent = latent[batch, indices].astype(jnp.float32)
        chosen_norm = inverse_key_norm[batch, indices].transpose(0, 1, 3, 2)
        selected_mask = jnp.take_along_axis(pair_valid, indices, axis=-1)
        logits = jnp.einsum("bqhc,bqkc->bqhk", absorbed_q, chosen_latent)
        logits = logits * chosen_norm / (head_dim ** .5)
        probs = masked_softmax(logits, selected_mask[:, :, None, :])
        values = jnp.einsum("bqhk,bqkc->bqhc", probs, chosen_latent)
        selected_scores = jnp.take_along_axis(scores, indices, axis=-1)
        loss, count = _indexer_kl(probs, selected_scores, selected_mask, query_weight) if collect_loss else (jnp.float32(0), jnp.float32(0))
        return finish(probs, values), loss, count

    # DeepSeek uses masked MHA for short contexts too. Here the fallback also
    # makes exact index-score ties independent of arbitrary schema ordering.
    def tied(_):
        return dense(pair_valid & (scores >= top_values[..., top_k - 1, None]))
    return jax.lax.cond(jnp.any(boundary_tie), tied, sparse, operand=None)


class AttentionContext:
    def __init__(self, runtime=None):
        self.numerator = jnp.float32(0)
        self.denominator = jnp.float32(0)
        self.row_weight = None
        self.runtime = runtime

    def add(self, loss, count):
        self.numerator = self.numerator + loss
        self.denominator = self.denominator + count


def set_branch(branch, scale, rows):
    ctx = _context.get()
    if ctx is None:
        return
    rows = int(rows)
    weight = jnp.ones((rows,), jnp.float32) if scale is None else jnp.broadcast_to(jnp.asarray(scale, jnp.float32), (rows,))
    if ctx.runtime is not None and ctx.runtime.in_svi_shard:
        ctx.runtime.set_branch(str(branch))
        n, local_n = ctx.runtime._branch_rows()
        if rows != local_n:
            raise ValueError("Attention branch does not match distributed observation layout")
        weight = weight * (jnp.arange(rows) + jax.lax.axis_index("data") * local_n < n)
    ctx.row_weight = jax.lax.stop_gradient(jnp.maximum(weight, 0))


def wrap_model(model, n_observations, loss_weight, runtime=None):
    """One auxiliary factor in the existing model/optimizer, outside its plates."""
    @wraps(model)
    def wrapped(*args, **kwargs):
        if runtime is not None and runtime.in_svi_shard:
            raise ValueError("MLA/DSA joint auxiliary objective is currently qualified for one device; disable data parallelism.")
        ctx = AttentionContext(runtime)
        token = _context.set(ctx)
        try:
            result = model(*args, **kwargs)
            mean_kl = ctx.numerator / jnp.maximum(ctx.denominator, 1)
            numpyro.factor("_strategize_dsa_indexer", -float(n_observations) * float(loss_weight) * mean_kl)
            return result
        finally:
            _context.reset(token)
    return wrapped


def unwrap_model(model):
    return model.__wrapped__


def transformer_scan(tokens, mask, params, attention_cfg, ffn_cfg, bias, n_heads, head_dim,
                     loop_cfg=None, return_details=False):
    """MLA/DSA with the shared dense/MoE and recurrent-depth executor."""
    cfg = dict(attention_cfg)
    if mask is not None:
        tokens = jnp.where(jnp.asarray(mask)[..., None] > 0, tokens, 0)
    ctx = _context.get()
    collect = ctx is not None and cfg["architecture"] == "mla_dsa"
    row_weight = None
    if ctx is not None and ctx.row_weight is not None:
        rows = ctx.row_weight.shape[0]
        if tokens.shape[0] % rows:
            raise ValueError("Attention rows must repeat complete observation layouts")
        row_weight = jnp.tile(ctx.row_weight, tokens.shape[0] // rows)

    def attention(x, layer):
        z = rms_norm(x, layer["RMS_attn"])
        a, loss, count = mla_attention(z, mask, layer, cfg, n_heads, head_dim,
                                      collect_loss=collect, row_weight=row_weight)
        h = x + layer["alpha_attn"].astype(x.dtype) * a
        return h, rms_norm(h, layer["RMS_ff"]), layer["alpha_ff"], loss, count

    result = run_layers(tokens, mask, params, ffn_cfg, bias, attention, loop_cfg,
                        return_details=return_details)
    if return_details:
        output, loss, count = result["output"], result["loss"], result["count"]
    else:
        output, loss, count = result
    if collect:
        ctx.add(loss, count)
    return result if return_details else output
