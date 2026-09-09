"""Run with JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=2."""
import importlib.util
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, TraceMeanField_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import optax_to_numpyro
import optax
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "inst/python"))
from strategize_distributed import Config, Runtime, collective_execution_schedule


def model(x, y, weights, x_single, y_single, weights_single, unused=None):
    w = numpyro.sample("w", dist.Normal(jnp.zeros(4), jnp.ones(4)).to_event(1))
    numpyro.factor("pair", jnp.sum(weights * dist.Bernoulli(logits=x @ w).log_prob(y)))
    numpyro.factor("single", jnp.sum(weights_single * dist.Normal(x_single @ w, 1).log_prob(y_single)))


def fixture(n=5, m=3, normalize=False):
    optimizer = optax.chain(optax.clip_by_global_norm(1), optax.adam(1e-3))
    if normalize:
        from strategize_optim import normalized_optimizer
        optimizer = normalized_optimizer(optax.adam(1e-3), objective_scale=1/(n+m), clip_global_norm=1)
    svi = SVI(model, AutoNormal(model), optax_to_numpyro(optimizer), TraceMeanField_ELBO())
    args = dict(x=np.arange(n*4, dtype=np.float32).reshape(n,4)/20,
                y=(np.arange(n)%2).astype(np.float32), weights=np.arange(n,dtype=np.float32),
                x_single=np.arange(m*4,dtype=np.float32).reshape(m,4)/16,
                y_single=np.arange(m,dtype=np.float32)/3, weights_single=np.array([0]+[2]*(m-1),np.float32),
                unused=None)
    state = svi.init(jax.random.PRNGKey(7), **args)
    return svi, state, args


def close(a,b):
    assert jax.tree.structure(a) == jax.tree.structure(b)
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):
        np.testing.assert_allclose(np.asarray(x),np.asarray(y),rtol=1e-5,atol=1e-6)


@pytest.fixture
def runtime():
    if len(jax.devices()) < 2:
        pytest.skip("requires two simulated CPU devices")
    return Runtime(dict(enabled=True))


@pytest.mark.parametrize("n,m", [(4,4),(5,3),(1,7)])
def test_weighted_global_elbo_update_and_scans(runtime,n,m):
    svi,state,args=fixture(n,m)
    batch=runtime.place_batch(args)
    replicated=runtime.replicate(state)
    expected=jax.jit(lambda s,a:svi.stable_update(s,**a))(state,args)
    actual=runtime.update(svi,replicated,batch)
    close(actual,expected)
    for steps in (3,1):
        chunks=runtime.stack_batches([batch]*steps)
        actual=runtime.update(svi,replicated,chunks,scan=True)
        def scan(s):
            return jax.lax.scan(lambda carry,_:svi.stable_update(carry,**args),s,None,length=steps)
        close(actual,jax.jit(scan)(state))


@pytest.mark.parametrize("normalize", [False, True])
def test_initialization_and_full_state_topology_change(runtime,tmp_path,normalize):
    svi,state,args=fixture(normalize=normalize)
    batch=runtime.place_batch(args)
    initialized=runtime.init_svi(svi,jax.random.PRNGKey(7),batch)
    close(initialized,state)
    state,_=runtime.update(svi,initialized,batch)
    generation=runtime.save_checkpoint(tmp_path,"latest",state,[0,1,2,255])
    loaded=runtime.load_checkpoint_payload(tmp_path)
    assert loaded["generation"] == generation
    np.testing.assert_array_equal(loaded["payload"],[0,1,2,255])
    single=Runtime(dict(enabled=False))
    restored=single.restore_state(tmp_path,state,generation)
    expected=runtime.update(svi,state,batch)
    close(jax.jit(lambda s:svi.stable_update(s,**args))(restored),expected)
    # A new single-device checkpoint can also be placed on the larger mesh.
    generation=single.save_checkpoint(tmp_path,"latest",restored,[7])
    close(runtime.restore_state(tmp_path,state,generation),state)


def test_checkpoint_commit_and_corruption(runtime,tmp_path):
    svi,state,args=fixture()
    runtime.place_batch(args)
    for i in range(3):
        runtime.save_checkpoint(tmp_path,"latest",state,[i])
    root=tmp_path/"full-state"
    assert len([p for p in root.iterdir() if p.is_dir()]) == 2
    (root/"partial").mkdir()
    assert runtime.load_checkpoint_payload(tmp_path)["payload"].tolist() == [2]
    pointer=json.loads((root/"latest.json").read_text())
    (root/pointer["generation"]/"metadata.rds").write_bytes(b"corrupt")
    with pytest.raises(RuntimeError,match="checksum"):
        runtime.load_checkpoint_payload(tmp_path)


@pytest.mark.parametrize("config", [dict(enabled=True,num_processes=2),dict(process_id=1),
                                   dict(enabled="on"),dict(num_processes=1.5),dict(unknown=1)])
def test_bad_configuration(config):
    with pytest.raises((ValueError,TypeError)):
        Config.parse(config)


def test_run_lock(tmp_path):
    first,second=Runtime(),Runtime()
    first.acquire_lock(tmp_path)
    try:
        with pytest.raises(RuntimeError,match="Another trainer"):
            second.acquire_lock(tmp_path)
    finally:
        first.release_locks()
    second.acquire_lock(tmp_path)
    second.release_locks()


def test_primary_inference_roundtrip(tmp_path):
    import orbax.checkpoint as ocp
    runtime = Runtime()
    tree = {"group": {"weight": np.arange(6, dtype=np.float32).reshape(2, 3),
                      "count": np.array(7, dtype=np.int32)}}
    path = tmp_path / "inference"
    runtime.save_inference_tree(path, tree)
    abstract = jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), tree)
    restore_args = jax.tree.map(lambda a: ocp.RestoreArgs(restore_type=np.ndarray, dtype=a.dtype), tree)
    restored = runtime.load_inference_tree(path, abstract, restore_args)
    for actual, expected in zip(jax.tree.leaves(restored), jax.tree.leaves(tree)):
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == expected.dtype
    runtime.rank = 1
    with pytest.raises(RuntimeError, match="Only rank zero"):
        runtime.load_inference_tree(path, abstract, restore_args)


@pytest.mark.parametrize("n,m", [(4,4),(5,3),(1,7)])
@pytest.mark.parametrize("normalize", [False, True])
def test_explicit_svi_preserves_weighted_loss_dropout_and_global_routing(runtime,n,m,normalize):
    def forward(X_left, Y_obs, obs_scale, X_single, Y_single_obs, obs_scale_single):
        w = numpyro.sample("w", dist.Normal(jnp.zeros(4), jnp.ones(4)).to_event(1))
        for branch, x, y, scale in (("pair",X_left,Y_obs,obs_scale),
                                     ("single",X_single,Y_single_obs,obs_scale_single)):
            runtime.set_branch(branch)
            key = numpyro.prng_key()
            u = runtime.schema_uniform(key,x.shape[0],4,jnp.float32) if runtime.in_svi_shard else jax.random.uniform(key,x.shape)
            masked = x * (u > .2)
            routes = jax.nn.softmax(jnp.stack((masked @ w, -(masked @ w)),axis=-1),axis=-1)[:,None,:]
            mask = jnp.ones(routes.shape[:2])
            mean = runtime.route_mean(routes,mask) if runtime.in_svi_shard else jnp.mean(routes,axis=(0,1))
            numpyro.factor(branch+"_balance", -3*jnp.sum((mean-.5)**2))
            numpyro.factor(branch, jnp.sum(scale*dist.Normal(masked@w,1).log_prob(y)))
    _,_,a=fixture(n,m)
    args=dict(X_left=a['x'],Y_obs=a['y'],obs_scale=a['weights'],
              X_single=a['x_single'],Y_single_obs=a['y_single'],obs_scale_single=a['weights_single'])
    optimizer = optax.chain(optax.clip_by_global_norm(1), optax.adam(1e-3))
    if normalize:
        from strategize_optim import normalized_optimizer
        optimizer = normalized_optimizer(optax.adam(1e-3), objective_scale=1/(n+m), clip_global_norm=1)
    svi=SVI(forward,AutoNormal(forward),optax_to_numpyro(optimizer),TraceMeanField_ELBO())
    state=svi.init(jax.random.PRNGKey(7),**args)
    expected=jax.jit(lambda s,a:svi.stable_update(s,**a))(state,args)
    runtime.configure_svi_sharding()
    batch=runtime.place_batch(args)
    from numpyro.infer.svi import _make_loss_fn
    _, loss_key = jax.random.split(state.rng_key)
    loss_fn = _make_loss_fn(svi.loss, loss_key, svi.constrain_fn, svi.model,
                           svi.guide, (), args, svi.static_kwargs, mutable_state=None)
    expected_grads = jax.jit(jax.grad(lambda p: loss_fn(p)[0]))(svi.optim.get_params(state.optim_state))
    replicated = runtime.replicate(state)
    actual_grads = runtime._compile(svi, batch, replicated, gradients=True)(replicated, batch)
    close(actual_grads, expected_grads)
    actual=runtime.update(svi,runtime.replicate(state),batch)
    close(actual,expected)
    expected_scan=jax.jit(lambda s:jax.lax.scan(lambda s,_:svi.stable_update(s,**args),s,None,length=3))(state)
    actual_scan=runtime.update(svi,runtime.replicate(state),runtime.stack_batches([batch]*3),scan=True)
    close(actual_scan,expected_scan)
