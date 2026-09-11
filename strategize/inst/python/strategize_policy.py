"""Compiled policy iterations and bounded-memory full-trace derivatives.

The objective is supplied by R. Parameters, PRNG keys and optimizer state are
explicit arguments; no fitted-model or cross-validation state is global here.
"""
import jax
import jax.numpy as jnp
import numpy as np


class GLMPairs:
    """Factor the existing difference-of-profile-features GLM exactly."""

    def __init__(self, main_indices, inter_indices, left_indices, right_indices,
                 binomial=True, ast_prop=0.5, dag_prop=0.5, strength=1.0):
        self.main = jnp.asarray(main_indices, dtype=jnp.int32).reshape(-1)
        self.inter = jnp.asarray(inter_indices, dtype=jnp.int32).reshape(-1)
        self.left = jnp.asarray(left_indices, dtype=jnp.int32).reshape(-1)
        self.right = jnp.asarray(right_indices, dtype=jnp.int32).reshape(-1)
        self.binomial = bool(binomial)
        self.ast_prop, self.dag_prop = ast_prop, dag_prop
        self.strength = strength

    def scores(self, profiles, coefficients):
        x = profiles.reshape(profiles.shape[0], -1)
        coefs = jnp.stack([c.reshape(-1) for c in coefficients], axis=1)
        scores = x @ coefs[self.main]
        if self.inter.size:
            scores = scores + (x[:, self.left] * x[:, self.right]) @ coefs[self.inter]
        return scores

    def link(self, x):
        return jax.nn.sigmoid(x) if self.binomial else x

    def population(self, left, right, intercept_ast, intercept_dag):
        delta = left - right
        return (self.ast_prop * self.link(jnp.reshape(intercept_ast, ()) + delta[..., 0])
                + self.dag_prop * self.link(jnp.reshape(intercept_dag, ()) + delta[..., 1]))

    def primary(self, left, right, intercept):
        return self.link(self.strength * (jnp.reshape(intercept, ()) + left - right))


def chunked_jacrev(fun, args, chunk_size=16):
    """Differentiate once, then apply the pullback to small output-row batches.

    Return host matrices: no dense output identity or batched full Jacobian is
    resident on the device. ``fun`` accepts a tuple of the differentiated inputs.
    """
    value, pullback = jax.vjp(fun, args)
    n = value.size
    width = min(int(chunk_size), n)
    rows = [np.empty((n, x.size), dtype=np.dtype(x.dtype)) for x in args]
    apply = jax.jit(jax.vmap(pullback))
    for start in range(0, n, width):
        indices = start + jnp.arange(width)
        basis = jax.nn.one_hot(indices, n, dtype=value.dtype)
        grads = apply(basis.reshape((width,) + value.shape))[0]
        count = min(width, n - start)
        for dst, grad in zip(rows, grads):
            dst[start:start + count] = np.asarray(grad).reshape(width, -1)[:count]
    return rows


class PolicyLoop:
    """Same updates and key splitting as getQPiStar_gd's R reference loop."""

    def __init__(self, grad_ast, grad_dag, steps, adversarial=False,
                 optimism="extragrad", optimism_coef=1.0, remat=True,
                 trace=False, optimizer=None, rain_eta=0.001):
        self.ga, self.gb = grad_ast, grad_dag
        self.steps = int(steps)
        self.adversarial = bool(adversarial)
        self.optimism = str(optimism)
        self.coef = float(optimism_coef)
        self.trace = bool(trace)
        self.optimizer = optimizer
        self.rain_eta = float(rain_eta)
        self.remat = bool(remat)
        self.run = jax.jit(self._run, static_argnames=("history",))

    def _run(self, a, b, key, objective_args, schedule, history=True):
        dtype = a.dtype
        zero = jnp.asarray(0., dtype=dtype)
        nan = jnp.asarray(jnp.nan, dtype=dtype)
        initial = dict(a=a, b=b, key=key, acc_a=zero, acc_b=zero,
                       prev_a=jnp.zeros_like(a), prev_b=jnp.zeros_like(b),
                       sum_a=jnp.zeros_like(a), sum_b=jnp.zeros_like(b),
                       weight_a=zero, weight_b=zero,
                       anchor_a=jnp.zeros_like(a), anchor_b=jnp.zeros_like(b),
                       anchor_weight=zero, stage_a=a, stage_b=b,
                       selected_a=a, selected_b=b)
        if self.optimizer is not None:
            initial["opt_a"] = self.optimizer.init(a)
            initial["opt_b"] = self.optimizer.init(b)

        def gradient(s, player, x, y, split=True):
            if split:
                # R indexes this Python array with [[1L]], i.e. Python index 1.
                s["key"] = jax.random.split(s["key"])[1]
            sign = jnp.asarray(1. if player == "a" else -1., dtype=dtype)
            result = (self.ga if player == "a" else self.gb)(
                x, y, *objective_args, sign, s["key"])
            return jnp.reshape(result[0], ()), result[1]

        def rate(s, player, grad, i):
            # Keep the reference's norm-then-square initialization and its
            # stop_gradient on the adaptive learning rate (also during SEs).
            acc = jnp.where(i == 0, jnp.maximum(jnp.asarray(.1, dtype),
                           10 * jnp.linalg.norm(grad) ** 2), s["acc_" + player])
            acc = jax.lax.stop_gradient(acc + jnp.sum(grad ** 2))
            s["acc_" + player] = acc
            return jnp.reciprocal(jnp.sqrt(acc))

        def step(state, xs):
            i, stage_start, stage_end, stage_lambda, stage_id, choose = xs
            s = dict(state)
            a0, b0 = s["a"], s["b"]
            pred_a, pred_b = a0, b0
            la, lb, na, nb, gamma_a, gamma_b = nan, nan, nan, nan, nan, nan
            anchor_norm_a, anchor_norm_b = zero, zero
            rain = self.optimism == "rain"
            extra = self.optimism in ("extragrad", "smp")

            if rain:
                s["stage_a"] = jnp.where(stage_start, a0, s["stage_a"])
                s["stage_b"] = jnp.where(stage_start, b0, s["stage_b"])
                denom = jnp.where(s["anchor_weight"] > 0, s["anchor_weight"], 1.)
                bar_a, bar_b = s["anchor_a"] / denom, s["anchor_b"] / denom
                s["key"] = jax.random.split(s["key"])[1]
                if self.adversarial:
                    lb, gb = gradient(s, "b", a0, b0, False)
                    gb = gb - s["anchor_weight"] * (b0 - bar_b)
                la, ga = gradient(s, "a", a0, b0, False)
                ga = ga - s["anchor_weight"] * (a0 - bar_a)
                eta = jnp.asarray(self.rain_eta, dtype)
                pred_a = a0 + eta * ga
                if self.adversarial:
                    pred_b = b0 + eta * gb
                s["selected_a"] = jnp.where(choose, pred_a, s["selected_a"])
                s["selected_b"] = jnp.where(choose, pred_b, s["selected_b"])
                s["key"] = jax.random.split(s["key"])[1]
                if self.adversarial:
                    _, gb = gradient(s, "b", pred_a, pred_b, False)
                    gb = gb - s["anchor_weight"] * (pred_b - bar_b)
                    s["b"] = b0 + eta * gb
                    nb = jnp.linalg.norm(gb)
                    s["acc_b"] = jnp.reciprocal(eta)
                _, ga = gradient(s, "a", pred_a, pred_b, False)
                ga = ga - s["anchor_weight"] * (pred_a - bar_a)
                s["a"] = a0 + eta * ga
                na = jnp.linalg.norm(ga)
                s["acc_a"] = jnp.reciprocal(eta)
                anchor_norm_a = jnp.where(s["anchor_weight"] > 0,
                                          jnp.linalg.norm(s["a"] - bar_a), zero)
                anchor_norm_b = jnp.where(s["anchor_weight"] > 0,
                                          jnp.linalg.norm(s["b"] - bar_b), zero)
            elif extra and self.adversarial:
                lb, gb = gradient(s, "b", a0, b0)
                la, ga = gradient(s, "a", a0, b0)
                gamma_b, gamma_a = rate(s, "b", gb, i), rate(s, "a", ga, i)
                pred_b, pred_a = b0 + gamma_b * gb, a0 + gamma_a * ga
                _, gb = gradient(s, "b", pred_a, pred_b)
                _, ga = gradient(s, "a", pred_a, pred_b)
                s["b"], s["a"] = b0 + gamma_b * gb, a0 + gamma_a * ga
                nb, na = jnp.linalg.norm(gb), jnp.linalg.norm(ga)
            else:
                # Alternating updates: B moves before A in none/OGDA/Optax.
                for player in (("b", "a") if self.adversarial else ("a",)):
                    loss, grad = gradient(s, player, s["a"], s["b"])
                    applied = grad
                    if self.optimism == "ogda":
                        applied = jnp.where(i == 0, grad,
                                           grad + self.coef * (grad - s["prev_" + player]))
                    if self.optimizer is not None:
                        updates, opt = self.optimizer.update(grad, s["opt_" + player], s[player])
                        s[player] = s[player] + updates
                        s["opt_" + player] = opt
                        gamma = nan
                    else:
                        gamma = rate(s, player, grad, i)
                        if extra:
                            pred_a = s["a"] + gamma * grad
                            _, applied = gradient(s, "a", pred_a, s["b"])
                        s[player] = s[player] + gamma * applied
                    s["prev_" + player] = grad
                    if player == "a":
                        la, na, gamma_a = loss, jnp.linalg.norm(applied), gamma
                    else:
                        lb, nb, gamma_b = loss, jnp.linalg.norm(applied), gamma

            if self.optimism == "smp":
                s["sum_a"] = s["sum_a"] + gamma_a * pred_a
                s["weight_a"] = s["weight_a"] + gamma_a
                if self.adversarial:
                    s["sum_b"] = s["sum_b"] + gamma_b * pred_b
                    s["weight_b"] = s["weight_b"] + gamma_b

            diagnostics = None
            if history:
                diagnostics = dict(loss_ast=la, loss_dag=lb, grad_ast=na, grad_dag=nb,
                    inv_lr_ast=s["acc_a"] if self.optimizer is None else nan,
                    inv_lr_dag=s["acc_b"] if self.adversarial and self.optimizer is None else nan,
                    gamma_ast=gamma_a, gamma_dag=gamma_b)
                if rain:
                    diagnostics.update(rain_lambda=stage_lambda, rain_lambda_sum=s["anchor_weight"],
                        rain_stage_idx=stage_id, rain_anchor_bar_norm_ast=anchor_norm_a,
                        rain_anchor_bar_norm_dag=anchor_norm_b)
                if self.trace:
                    diagnostics["trace"] = dict(start_ast=a0, start_dag=b0,
                                                pred_ast=pred_a, pred_dag=pred_b)
            if rain:
                # stage_end is 1 for last output, 2 for uniform-half output.
                uniform = stage_end == 2
                s["a"] = jnp.where(uniform, s["selected_a"], s["a"])
                if self.adversarial:
                    s["b"] = jnp.where(uniform, s["selected_b"], s["b"])
                weight = jnp.where(stage_end > 0, stage_lambda, zero)
                s["anchor_a"] = s["anchor_a"] + weight * s["stage_a"]
                s["anchor_b"] = s["anchor_b"] + weight * s["stage_b"]
                s["anchor_weight"] = s["anchor_weight"] + weight
            return s, diagnostics

        body = jax.checkpoint(step, prevent_cse=False) if self.remat else step
        state, diagnostics = jax.lax.scan(body, initial, schedule)
        if self.optimism == "smp" and self.steps:
            state["a"] = state["sum_a"] / state["weight_a"]
            if self.adversarial:
                state["b"] = state["sum_b"] / state["weight_b"]
        return dict(a=state["a"], b=state["b"], key=state["key"], history=diagnostics,
                    weight_a=state["weight_a"], weight_b=state["weight_b"])
