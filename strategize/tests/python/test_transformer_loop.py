"""Recurrent-depth oracles, shared gradients, masks and trace-local statistics."""
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.handlers as handlers
import pytest

from test_latent_attention import a, moe, fixture, oracle, norm_ref, close, CFG
from test_efficiency import transformer_fixture


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


def test_auxiliary_and_moe_statistics_count_each_shared_layer_use_without_tracer_leaks():
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
