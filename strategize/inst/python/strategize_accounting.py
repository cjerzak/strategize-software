"""Shape-only predictor accounting, independent of optimizer/guide storage."""

import math

import jax
from numpyro.handlers import seed


def parameter_inventory(factory, moe=None):
    # A private seed traces sample sites without consuming the training RNG.
    # eval_shape never materializes the (potentially very large) weight arrays.
    shapes = jax.eval_shape(seed(factory, rng_seed=0))
    aliases = {"ordinal_thresholds", "pairwise_bernoulli_logit_scale", "calibration_scale",
               "transformer_moe_router_bias"}
    if "ordinal_threshold_raw" not in shapes:
        aliases.discard("ordinal_thresholds")
    shapes = {name: tuple(value.shape) for name, value in shapes.items()
              if value is not None and name not in aliases}
    sizes = {name: math.prod(shape) for name, shape in shapes.items()}
    stored = sum(sizes.values())
    active = float(stored)
    if moe is not None:
        ratio = float(moe["n_experts_per_tok"]) / int(moe["n_routed_experts"])
        active -= (1 - ratio) * sum(size for name, size in sizes.items()
                                   if name.startswith(("W_moe_expert1_", "W_moe_expert2_")))
    # R integers are 32-bit. Return counts as doubles, not narrowed PyInts.
    return {"parameter_shapes": {name: list(shape) for name, shape in shapes.items()},
            "stored_model_parameters": float(stored), "active_model_parameters": active}
