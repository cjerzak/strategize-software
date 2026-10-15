import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "inst/python"))
from strategize_accounting import parameter_inventory


@pytest.mark.parametrize("moe", [None, dict(n_routed_experts=8, n_experts_per_tok=2)])
def test_inventory_matches_actual_predictor_tree_excluding_aliases(moe):
    def factory():
        w = numpyro.param("w", jnp.zeros((4, 8)))
        thresholds = numpyro.sample("thresholds", dist.Normal(0., 1.).expand((3,)))
        params = dict(W_q_layers=w, ordinal_threshold_raw=thresholds,
                      ordinal_thresholds=jnp.cumsum(thresholds),
                      log_calibration_scale=jnp.float32(0), calibration_scale=jnp.float32(1),
                      transformer_moe_router_bias=jnp.zeros((2,8)), sigma=None)
        if moe:
            params["W_moe_expert1_layers"] = jnp.zeros((2,8,4,12))
            params["W_moe_expert2_layers"] = jnp.zeros((2,8,6,4))
            params["W_moe_router_layers"] = jnp.zeros((2,4,8))
            params["W_moe_shared1_layers"] = jnp.zeros((2,4,12))
        return params
    actual = numpyro.handlers.seed(factory, rng_seed=0)()
    inventory = parameter_inventory(factory, moe)
    counted = inventory["parameter_shapes"]
    assert inventory["stored_model_parameters"] == sum(actual[name].size for name in counted)
    assert all(tuple(actual[name].shape) == tuple(shape) for name, shape in counted.items())
    inactive = 0 if moe is None else .75 * (2*8*4*12 + 2*8*6*4)
    assert inventory["active_model_parameters"] == inventory["stored_model_parameters"] - inactive
    assert "ordinal_thresholds" not in counted
    assert "transformer_moe_router_bias" not in counted


def test_huge_inventory_uses_abstract_arrays_without_advancing_rng():
    import random
    import numpy as np
    import pickle
    before = pickle.dumps((random.getstate(), np.random.get_state()))
    def factory():
        return {"w": numpyro.param("w", jnp.zeros((65536,65536))),
                "b": numpyro.sample("b", dist.Normal(0.,1.).expand((8,)))}
    result = parameter_inventory(factory)
    assert result["stored_model_parameters"] == 2**32 + 8
    assert result["parameter_shapes"]["w"] == [65536,65536]
    assert before == pickle.dumps((random.getstate(), np.random.get_state()))
