"""Independent attention oracle, gradients, masks, indexer training, scan and ELBO."""
from pathlib import Path
import sys
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.handlers as handlers
import optax
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'inst/python'))
import strategize_attention as a
import strategize_moe as moe

CFG = dict(architecture='mla_dsa', q_rank=8, kv_rank=6, indexer_heads=2,
           indexer_dim=4, top_k=3, indexer_loss_weight=.01)


def fixture(seed=1, length=9, norm=True):
    rng=np.random.default_rng(seed)
    shapes=dict(W_q=(16,8), W_k=(16,6), W_v=(6,32), W_o=(16,16), W_q_up=(8,16),
                W_index_q=(8,8), W_index_k=(16,4), W_index_w=(16,2))
    p={k:jnp.array(rng.normal(size=s)/np.sqrt(s[0]),dtype=jnp.float32) for k,s in shapes.items()}
    p.update(RMS_q_latent=jnp.ones(8), RMS_kv_latent=jnp.ones(6), LN_index_k=jnp.ones(4), b_index_k=jnp.zeros(4))
    if norm:p.update(RMS_q=jnp.array([.8,1.,1.2,1.1]),RMS_k=jnp.array([1.1,.8,1.,1.2]))
    x=jnp.array(rng.normal(size=(2,length,16)),dtype=jnp.float32)
    mask=jnp.ones((2,length)).at[0,-2:].set(0)
    return x,mask,p


def norm_ref(x,g):
    return x / jnp.sqrt(jnp.mean(x*x,axis=-1,keepdims=True)+1e-6)*g


def oracle(x,mask,p,cfg):
    """Expand ordinary Q/K/V per head; never use absorbed attention/gather."""
    x=jnp.where(mask[...,None]>0,x,0)
    ql=norm_ref(x@p['W_q'],p['RMS_q_latent'])
    kv=norm_ref(x@p['W_k'],p['RMS_kv_latent'])
    q=(ql@p['W_q_up']).reshape(*x.shape[:2],4,4)
    k,v=jnp.split(kv@p['W_v'],2,axis=-1)
    k,v=k.reshape(q.shape),v.reshape(q.shape)
    if 'RMS_q' in p:q=norm_ref(q,p['RMS_q'])
    if 'RMS_k' in p:k=norm_ref(k,p['RMS_k'])
    select=(mask[:,:,None]>0)&(mask[:,None,:]>0)
    if cfg['architecture']=='mla_dsa' and cfg['top_k']<x.shape[1]:
        # Independent lightning-indexer expression and order-statistic mask.
        iq=(jax.lax.stop_gradient(ql)@p['W_index_q']).reshape(*x.shape[:2],2,4)
        ik=jax.lax.stop_gradient(x)@p['W_index_k']
        ik=(ik-ik.mean(-1,keepdims=True))/jnp.sqrt(ik.var(-1,keepdims=True)+1e-6)
        ik=ik*p['LN_index_k']+p['b_index_k']
        iw=jax.lax.stop_gradient(x)@p['W_index_w']/np.sqrt(2)
        scores=sum(iw[:,:,h,None]*jax.nn.relu(iq[:,:,h]@ik.transpose(0,2,1)/2) for h in range(2))
        threshold=jnp.sort(jnp.where(select,scores,-1e30),axis=-1)[...,-cfg['top_k'],None]
        select=select & (scores>=threshold)
    scores=jnp.einsum('bqhd,bkhd->bqhk',q,k)/2
    scores=jnp.where(select[:,:,None,:],scores,-1e30)
    probs=jax.nn.softmax(scores,-1)*select[:,:,None,:]
    y=jnp.einsum('bqhk,bkhd->bqhd',probs,v).reshape(x.shape)@p['W_o']
    return jnp.where(mask[...,None]>0,y,0)


def close(x,y,rtol=3e-4,atol=5e-5):
    for xx,yy in zip(jax.tree.leaves(x),jax.tree.leaves(y)):
        assert np.isfinite(xx).all() and np.isfinite(yy).all()
        np.testing.assert_allclose(xx,yy,rtol=rtol,atol=atol)


@pytest.mark.parametrize('architecture,top_k',[('mla',3),('mla_dsa',3),('mla_dsa',20)])
@pytest.mark.parametrize('norm',[False,True])
def test_expanded_forward_and_all_derivatives(architecture,top_k,norm):
    x,m,p=fixture(norm=norm); cfg=dict(CFG,architecture=architecture,top_k=top_k)
    fn=lambda xx,pp:a.mla_attention(xx,m,pp,cfg,4,4)[0]
    ref=lambda xx,pp:oracle(xx,m,pp,cfg)
    close(jax.jit(fn)(x,p),ref(x,p))
    close(jax.jit(jax.grad(lambda xx,pp:(fn(xx,pp)**2).sum(),(0,1)))(x,p),
          jax.grad(lambda xx,pp:(ref(xx,pp)**2).sum(),(0,1))(x,p),rtol=2e-3,atol=3e-4)


@pytest.mark.parametrize('ties',[False,True])
def test_set_permutation_masked_garbage_padding_and_empty_rows(ties):
    x,m,p=fixture()
    if ties:p['W_index_w']=jnp.zeros_like(p['W_index_w'])
    fn=jax.jit(lambda xx,mm:a.mla_attention(xx,mm,p,CFG,4,4,collect_loss=True))
    y,loss,n=fn(x,m)
    order=np.random.default_rng(4).permutation(x.shape[1]); inverse=np.argsort(order)
    yp,lp,np_=fn(x[:,order],m[:,order]); close(y,yp[:,inverse]);close(loss,lp);close(n,np_)
    bad=jnp.where(m[...,None]>0,x,jnp.nan);close(fn(bad,m),(y,loss,n))
    pad=jnp.pad(x,((0,0),(0,3),(0,0)),constant_values=jnp.nan);mp=jnp.pad(m,((0,0),(0,3)))
    yy,ll,nn=fn(pad,mp);close(yy[:,:x.shape[1]],y);close(ll,loss);close(nn,n)
    empty=jnp.zeros_like(m); yy,ll,nn=fn(jnp.full_like(x,jnp.nan),empty)
    close(yy,jnp.zeros_like(x));close(ll,0.);close(nn,0.)
    gg=jax.grad(lambda xx:a.mla_attention(xx,empty,p,CFG,4,4,collect_loss=True)[0].sum())(x)
    close(gg,jnp.zeros_like(x))


def test_indexer_receives_learning_signal_but_auxiliary_does_not_change_trunk():
    x,m,p=fixture();cfg=dict(CFG,top_k=99)
    loss=lambda pp:a.mla_attention(x,m,pp,cfg,4,4,collect_loss=True)[1]/m.sum()
    g=jax.grad(loss)(p)
    for key in p:
        if 'index' in key: assert float(jnp.linalg.norm(g[key]))>0,key
        else: close(g[key],jnp.zeros_like(g[key]))
    close(jax.grad(lambda xx:a.mla_attention(xx,m,p,cfg,4,4,collect_loss=True)[1])(x),jnp.zeros_like(x))
    optimizer=optax.adam(.01);state=optimizer.init(p); initial=float(loss(p))
    @jax.jit
    def update(p,state):
        gg=jax.grad(loss)(p);u,state=optimizer.update(gg,state,p)
        return optax.apply_updates(p,u),state
    for _ in range(80):p,state=update(p,state)
    assert float(loss(p))<initial*.7


def test_scan_and_trace_local_auxiliary_equal_loop_and_do_not_leak_tracers():
    x,m,p=fixture(); p.update(RMS_attn=jnp.ones(16),RMS_ff=jnp.ones(16),alpha_attn=jnp.array(.2),alpha_ff=jnp.array(.1),
        W_ff1=jnp.ones((16,32))*.03,W_ff2=jnp.eye(16)*.2)
    stacked={k+'_layers':jnp.stack([v,v*.97]) for k,v in p.items()};stacked['RMS_final']=jnp.ones(16)
    def model(xx):
        a.set_branch('single',jnp.array([1.,0.]),2)
        return a.transformer_scan(xx,m,stacked,CFG,None,None,4,4)
    wrapped=a.wrap_model(model,100,.01)
    trace=handlers.trace(wrapped).get_trace(x)
    assert set(trace)=={'_strategize_dsa_indexer'}
    z=jnp.where(m[...,None]>0,x,0);los=0;num=0
    for layer in range(2):
        pp={k:v[layer] for k,v in stacked.items() if k.endswith('_layers')}
        pp={k[:-7]:v for k,v in pp.items()}
        y,l,n=a.mla_attention(a.rms_norm(z,pp['RMS_attn']),m,pp,CFG,4,4,collect_loss=True,row_weight=jnp.array([1.,0.]))
        z=z+pp['alpha_attn']*y
        z=z+pp['alpha_ff']*moe.swiglu(a.rms_norm(z,pp['RMS_ff']),pp['W_ff1'],pp['W_ff2']);los+=l;num+=n
    close(model(x),a.rms_norm(z,stacked['RMS_final']))
    close(model(jnp.where(m[...,None]>0,x,jnp.nan)),model(x))
    close(trace['_strategize_dsa_indexer']['fn'].log_factor,-los/num)
    fn=jax.jit(lambda xx:handlers.trace(wrapped).get_trace(xx)['_strategize_dsa_indexer']['fn'].log_factor)
    assert np.isfinite(fn(x));assert np.isfinite(fn(x*1.1));assert a._context.get() is None


def test_bfloat16_outputs_and_gradients_are_finite_and_close():
    x,m,p=fixture(length=40)
    y=a.mla_attention(x,m,p,CFG,4,4)[0]
    f=jax.jit(lambda pp:a.mla_attention(x.astype(jnp.bfloat16),m,pp,CFG,4,4,collect_loss=True))
    yy,ll,nn=f(p);assert np.isfinite(yy).all() and np.isfinite(ll)
    assert np.sqrt(np.mean((np.asarray(yy,dtype=float)-np.asarray(y))**2))<.1
    g=jax.jit(jax.grad(lambda pp:sum(jnp.sum(v.astype(jnp.float32)) for v in f(pp)[:2])))(p)
    assert all(np.isfinite(v).all() for v in jax.tree.leaves(g))
