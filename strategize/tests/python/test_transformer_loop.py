"""Recurrent-depth oracles, shared gradients, masks and trace-local statistics."""
import contextlib
import io
import re
from types import SimpleNamespace

import jax
from jax.ad_checkpoint import print_saved_residuals
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
import numpy as np
import numpyro.handlers as handlers
import pytest

from test_latent_attention import a, moe, fixture, oracle, norm_ref, close, CFG
from test_efficiency import transformer_fixture
from test_transformer_moe import expert_kernel  # noqa: F401  (pytest fixture)
import strategize_transformer as tr


def test_checkpoint_identity_covers_the_recurrent_executor(monkeypatch):
    import strategize_distributed as distributed
    runtime = distributed.Runtime(dict(enabled=False))
    original = distributed._sha
    identity = runtime.metadata()['runtime_sha256']
    monkeypatch.setattr(distributed, '_sha', lambda p:
        'changed' if p.name == 'strategize_transformer.py' else original(p))
    assert runtime.metadata()['runtime_sha256'] != identity


def dense_fixture(depth=4):
    x, mask, p = fixture(length=6)
    rng = np.random.default_rng(45)
    p.update(RMS_attn=jnp.ones(16), RMS_ff=jnp.ones(16),
             alpha_attn=jnp.array(.2), alpha_ff=jnp.array(.15),
             W_ff1=jnp.array(rng.normal(size=(16, 32)) * .1, dtype=jnp.float32),
             W_ff2=jnp.array(rng.normal(size=(16, 16)) * .1, dtype=jnp.float32))
    p = {k + '_layers': jnp.stack([v * (.97 ** i) for i in range(depth)]) for k, v in p.items()}
    return x, mask, dict(p, RMS_final=jnp.ones(16))


def reference(x, mask, params, loop):
    depth = params['W_q_layers'].shape[0]
    def layers(z, start, stop):
        for i in range(start, stop):
            p = {k[:-7]: v[i] for k, v in params.items() if k.endswith('_layers')}
            z = z + p['alpha_attn'] * oracle(norm_ref(z, p['RMS_attn']), mask, p, CFG)
            gate, v = jnp.split(norm_ref(z, p['RMS_ff']) @ p['W_ff1'], 2, -1)
            z = z + p['alpha_ff'] * ((jax.nn.silu(gate) * v) @ p['W_ff2'])
        return z
    z = jnp.where(mask[..., None] > 0, x, 0)
    if loop.get('enabled', False):
        pre, end = loop['prelude_layers'], depth - loop['coda_layers']
        embedded = layers(z, 0, pre)
        embedded = jnp.where(mask[..., None] > 0, embedded, 0)
        z = jnp.zeros_like(embedded)
        keep = loop.get('backprop_iterations', 0) if loop.get('training', False) else 0
        iterations = int(loop.get('active_iterations', loop['iterations']))
        for i in range(iterations):
            if keep and i == iterations - keep:
                z = jax.lax.stop_gradient(z)
            if loop.get('reinjection') == 'rms_gated':
                injected = embedded if i == 0 else norm_ref(
                    params.get('loop_state_gain', 1.) * z +
                    params.get('loop_input_gain', 1.) * embedded,
                    jnp.ones(embedded.shape[-1]))
            else:
                injected = (embedded + z) / jnp.sqrt(2.)
            z = layers(injected, pre, end)
        z = layers(z, end, depth)
    else:
        z = layers(z, 0, depth)
    return norm_ref(z, params['RMS_final'])


@pytest.mark.parametrize('iterations,pre,coda,backprop', [(1,0,0,0), (2,1,1,0), (4,1,0,2)])
def test_scanned_shared_weights_and_all_gradients_match_explicit_recurrence(iterations, pre, coda, backprop):
    x, m, p = dense_fixture()
    loop = dict(enabled=True, iterations=iterations, prelude_layers=pre, coda_layers=coda,
                backprop_iterations=backprop, training=True)
    fn = lambda xx, pp: a.transformer_scan(xx, m, pp, CFG, None, None, 4, 4, loop)
    ref = lambda xx, pp: reference(xx, m, pp, loop)
    close(jax.jit(fn)(x,p), ref(x,p))
    target = jnp.linspace(-.5,.5,x.size).reshape(x.shape)
    grad = lambda f: jax.jit(jax.grad(lambda xx,pp:jnp.square(f(xx,pp)-target).sum(),(0,1)))(x,p)
    close(grad(fn), grad(ref), rtol=3e-3, atol=5e-4)
    # R changes neither the input parameter tree nor its leaf sizes.
    deeper = dict(loop, iterations=iterations+1)
    y = a.transformer_scan(x,m,p,CFG,None,None,4,4,deeper)
    assert not np.allclose(y,fn(x,p))


@pytest.mark.parametrize('reinjection', ['scaled_add', 'rms_gated'])
@pytest.mark.parametrize('backprop', [0, 2])
@pytest.mark.parametrize('active', [1, 2, 3, 4])
def test_sampled_depth_values_and_gradients_match_explicit_recurrence(active, backprop, reinjection):
    x, m, p = dense_fixture(4)
    p.update(loop_state_gain=jnp.array(.8), loop_input_gain=jnp.array(1.2))
    loop = dict(enabled=True, iterations=2, max_iterations=4, prelude_layers=1, coda_layers=1,
                reinjection=reinjection, backprop_iterations=backprop, training=True)
    # The sampled depth is traced: one program serves every R.
    fn = lambda xx, pp, r: a.transformer_scan(xx, m, pp, CFG, None, None, 4, 4, dict(loop, active_iterations=r))
    ref = lambda xx, pp, r: reference(xx, m, pp, dict(loop, active_iterations=active))
    r = jnp.int32(active)
    close(jax.jit(fn)(x, p, r), ref(x, p, r))
    target = jnp.linspace(-.5, .5, x.size).reshape(x.shape)
    grad = lambda f: jax.jit(jax.grad(lambda xx, pp, rr: jnp.square(f(xx, pp, rr) - target).sum(), (0, 1)))(x, p, r)
    close(grad(fn), grad(ref), rtol=3e-3, atol=5e-4)


def test_masks_permutation_batching_and_inference_keeps_full_derivatives():
    x,m,p = dense_fixture(3)
    loop = dict(enabled=True,iterations=4,prelude_layers=1,coda_layers=1,backprop_iterations=1)
    fn = lambda xx,mm:a.transformer_scan(xx,mm,p,CFG,None,None,4,4,loop)
    y = jax.jit(fn)(x,m)
    close(fn(jnp.where(m[...,None]>0,x,jnp.nan),m),y)
    order = np.array([4,2,0,5,1,3]); inv = np.argsort(order)
    close(fn(x[:,order],m[:,order])[:,inv],y)
    close(jnp.concatenate([fn(x[i:i+1],m[i:i+1]) for i in range(2)]),y)
    close(fn(jnp.full_like(x,jnp.nan),jnp.zeros_like(m)),jnp.zeros_like(x))
    objective = lambda xx: (fn(xx,m) * x).sum()
    full = lambda xx: (reference(xx,m,p,dict(loop,backprop_iterations=0)) * x).sum()
    close(jax.grad(objective)(x),jax.grad(full)(x),rtol=2e-3,atol=3e-4)


def test_sampled_depth_gated_reinjection_and_activation_diagnostics():
    x, m, p = dense_fixture(4)
    p.update(loop_state_gain=jnp.array(.8), loop_input_gain=jnp.array(1.2))
    loop = dict(enabled=True, iterations=2, active_iterations=jnp.int32(2),
                max_iterations=4, prelude_layers=1, coda_layers=1,
                reinjection='rms_gated', jacobian_power_iterations=1)
    details = jax.jit(lambda xx: a.transformer_scan(
        xx, m, p, CFG, None, None, 4, 4, loop, True))(x)
    close(details['output'], reference(x, m, p, loop), rtol=2e-3, atol=3e-4)
    assert int(details['active_iterations']) == 2
    assert details['states'].shape[0] == 4
    np.testing.assert_allclose(
        details['states'][2:], jnp.broadcast_to(details['states'][1], details['states'][2:].shape))
    assert np.isfinite(details['state_l2_mean'][:2]).all()
    assert np.isnan(details['state_l2_mean'][2:]).all()
    assert np.isnan(details['state_cosine_previous'][0])
    assert np.isfinite(details['state_cosine_previous'][1])
    assert float(details['core_jacobian_spectral_norm']) > 0


def test_auxiliary_and_moe_statistics_count_each_shared_layer_use_without_tracer_leaks(expert_kernel):
    # The existing MHA fixture has one dense layer followed by two MoE layers.
    _, x, p = transformer_fixture('float32', True)
    cfg = dict(n_routed_experts=4,n_experts_per_tok=2,n_shared_experts=1,moe_d_ff=8,
               first_k_dense=1,n_moe_layers=2,routed_scaling_factor=1.,
               capacity_factor=1.5,router_bias_rate=.001,compute_dtype='float32')
    loop = dict(enabled=True,iterations=3,prelude_layers=1,coda_layers=1)
    def attention(q,k,v,mask,*_):
        return jax.nn.dot_product_attention(q,k,v,mask=(mask>0)[:,None,None,:],implementation='xla')
    def model(xx):
        moe.set_branch('single',jnp.array([1.,0.,1.]),3)
        return moe.transformer_scan(xx,jnp.ones(xx.shape[:2]),p,cfg,None,2,4,attention,'xla','auto',8,loop)
    wrapped = moe.wrap_model(model,cfg)
    @jax.jit
    def stats(xx):
        return handlers.trace(wrapped).get_trace(xx)[moe.STATE]['value']['stats']
    s = stats(x); again = stats(x*.9)
    assert np.isfinite(again).all() and moe._context.get() is None
    close(s[:,-3],jnp.array([3*2*9,2*9],dtype=jnp.float32))
    close(s[:,:4].sum(-1),2*s[:,-3])
    close(s[:,4:8].sum(-1),2*s[:,-3])
    inference = moe.transformer_scan(
        x, jnp.ones(x.shape[:2]), p, cfg, jnp.zeros((2, 4)), 2, 4,
        attention, 'xla', 'auto', 8, loop, True)
    assert np.isfinite(inference['moe_load_entropy']).all()
    assert np.isnan(inference['moe_route_change_fraction'][0])
    assert np.isfinite(inference['moe_route_change_fraction'][1:]).all()
    xx, m, pp = dense_fixture(4)
    def latent_model(xx):
        a.set_branch('single',jnp.array([1.,0.]),2)
        return a.transformer_scan(xx,m,pp,CFG,None,None,4,4,loop)
    @jax.jit
    def aux(xx):
        ctx = a.AttentionContext()
        token = a._context.set(ctx)
        try:
            latent_model(xx)
            return ctx.numerator,ctx.denominator
        finally:
            a._context.reset(token)
    loss,count = aux(xx)
    assert np.isfinite(loss) and float(loss)>0
    close(count,m[0].sum()*(1+2*3+1))
    assert a._context.get() is None


def recurrent_moe():
    """Dense prelude, two shared MoE core layers and an MoE coda; R=3 of at most 4."""
    d, depth, experts = 8, 4, 4
    keys = iter(jax.random.split(jax.random.PRNGKey(37), 16))
    rand = lambda shape: .1 * jax.random.normal(next(keys), shape)
    p = {name + '_layers': rand((depth, d, d)) for name in ('W_q', 'W_k', 'W_v', 'W_o')}
    p.update({name + '_layers': jnp.ones((depth, d)) for name in ('RMS_attn', 'RMS_ff')})
    p.update({name + '_layers': jnp.full((depth,), .1) for name in ('alpha_attn', 'alpha_ff')})
    p.update(W_ff1_layers=rand((1, d, 2 * d)), W_ff2_layers=rand((1, d, d)),
             W_moe_router_layers=rand((depth - 1, d, experts)),
             W_moe_expert1_layers=rand((depth - 1, experts, d, 2 * d)),
             W_moe_expert2_layers=rand((depth - 1, experts, d, d)),
             W_moe_shared1_layers=rand((depth - 1, d, 2 * d)),
             W_moe_shared2_layers=rand((depth - 1, d, d)), RMS_final=jnp.ones(d))
    cfg = dict(n_routed_experts=experts, n_experts_per_tok=2, n_shared_experts=1, moe_d_ff=d,
               first_k_dense=1, n_moe_layers=depth - 1, routed_scaling_factor=1., capacity_factor=1.5,
               router_bias_rate=.001, compute_dtype='float32', activation_checkpointing=True)
    loop = dict(enabled=True, iterations=2, max_iterations=4, active_iterations=jnp.int32(3),
                prelude_layers=1, coda_layers=1, training=True)
    bias = jnp.zeros((depth - 1, experts))
    def attention(q, k, v, mask, *_):
        return jax.nn.dot_product_attention(q, k, v, mask=(mask > 0)[:, None, None, :], implementation='xla')
    def run(x, p, bias, loop):
        return moe.transformer_scan(x, jnp.ones(x.shape[:-1]), p, cfg, bias, 2, d // 2,
                                    attention, 'xla', 'auto', 8, loop)
    def loss(x, p, runtime=None):
        token = moe._context.set(moe.TraceContext(cfg, bias, runtime))
        try:
            return jnp.square(run(x, p, None, loop) - .3).sum()
        finally:
            moe._context.reset(token)
    return rand((4, 9, d)), p, loss, lambda x, p: run(x, p, bias, dict(loop, training=False))


def test_iteration_remat_saves_no_weight_stacks_and_adds_no_recompute(expert_kernel, monkeypatch):
    """The sampled-depth cond used to make the scan save a copy of the core weights per iteration."""
    x, p, loss, predict = recurrent_moe()
    grad = jax.grad(loss, (0, 1))
    def measure(iteration_remat):
        monkeypatch.setattr(tr, '_ITERATION_REMAT', iteration_remat)
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            print_saved_residuals(loss, x, p)
        # [iterations, core layers, *expert or attention weight]
        stacks = re.findall(r'f32\[4,2,4,8,16\]|f32\[4,2,8,8\]', stream.getvalue())
        dots = str(jax.make_jaxpr(grad)(x, p)).count('dot_general[')
        return stacks, dots, jax.jit(grad)(x, p), jax.jit(predict)(x, p)
    old, new = measure(False), measure(True)
    assert old[0] and not new[0], new[0]
    assert new[1] == old[1]  # boundary names leave the sublayer recomputation unchanged
    close(new[2], old[2], rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(new[3], old[3])


def test_iteration_remat_keeps_routing_collectives_out_of_backward(expert_kernel, monkeypatch):
    if len(jax.devices()) < 2:
        pytest.skip('requires simulated devices')
    x, p, loss, _ = recurrent_moe()
    mesh = Mesh(np.array(jax.devices()[:2]), ('data',))
    def compiled(iteration_remat):
        monkeypatch.setattr(tr, '_ITERATION_REMAT', iteration_remat)
        def local(x, p):
            runtime = SimpleNamespace(in_svi_shard=True, device_count=2, collective_chain=jnp.float32(0))
            return jax.lax.psum(loss(x, p, runtime), 'data')
        fn = jax.jit(jax.shard_map(jax.grad(local, (0, 1)), mesh=mesh, in_specs=(P('data'), P()),
                                   out_specs=(P('data'), P()), check_vma=False))
        hlo = fn.lower(x, p).compile().as_text()
        return len(re.findall(r' all-gather(?:-start)?\(', hlo)), fn(x, p)
    (old, old_grads), (new, new_grads) = compiled(False), compiled(True)
    # Saved routing decisions: the capacity all_gather runs once per routed layer use.
    assert old and new == old
    close(new_grads, old_grads, rtol=1e-6, atol=1e-7)
