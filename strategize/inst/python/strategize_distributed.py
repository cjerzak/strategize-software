"""Optional global-array SVI data parallelism and portable full-state checkpoints.

No backend access at import time. The mathematical model and NumPyro optimizer
are unchanged. Operational conventions follow sailfast-software's distributed
trainer and UltrascaleTrainingSkill-JAX (Austin et al., Scaling Book).
"""
from __future__ import annotations

import dataclasses
import contextlib
import hashlib
import gzip
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import shutil
import time
import uuid

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P


@dataclasses.dataclass(frozen=True)
class Config:
    enabled: bool = False
    num_processes: int = 1
    process_id: int = 0
    coordinator_address: str | None = None
    local_device_ids: tuple[int, ...] | None = None
    initialization_timeout: int = 120
    heartbeat_timeout_seconds: int = 60
    shutdown_timeout_seconds: int = 15

    @classmethod
    def parse(cls, value=None):
        if value is None:
            value = json.loads(os.environ.get("STRATEGIZE_DATA_PARALLEL", "{}"))
        if isinstance(value, bool):
            value = {"enabled": value}
        value = dict(value or {})
        unknown = set(value) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"Unknown data_parallel fields: {sorted(unknown)}")
        for key in ("num_processes", "process_id", "initialization_timeout",
                    "heartbeat_timeout_seconds", "shutdown_timeout_seconds"):
            if key in value:
                n = value[key]
                if isinstance(n, bool) or int(n) != n:
                    raise ValueError(f"{key} must be an integer")
                value[key] = int(n)
        ids = value.get("local_device_ids")
        if ids is not None:
            ids = [ids] if isinstance(ids, (int, float)) else list(ids)
            if not ids or any(int(n) != n or n < 0 for n in ids):
                raise ValueError("local_device_ids must contain nonnegative integers")
            value["local_device_ids"] = tuple(map(int, ids))
        cfg = cls(**value)
        if not isinstance(cfg.enabled, bool):
            raise ValueError("data_parallel.enabled must be TRUE or FALSE")
        if cfg.num_processes < 1 or not 0 <= cfg.process_id < cfg.num_processes:
            raise ValueError("Require num_processes >= 1 and 0 <= process_id < num_processes")
        if min(cfg.initialization_timeout, cfg.heartbeat_timeout_seconds,
               cfg.shutdown_timeout_seconds) < 1:
            raise ValueError("Distributed timeouts must be positive")
        if cfg.num_processes > 1:
            if not cfg.enabled or not cfg.coordinator_address:
                raise ValueError("Multiple processes require enabled data parallelism and a coordinator")
            if cfg.local_device_ids is None:
                cfg = dataclasses.replace(cfg, local_device_ids=(0,))
            if len(cfg.local_device_ids) != 1:
                raise ValueError("Multi-host SVI requires one GPU per process")
        return cfg


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with tmp.open("w") as stream:
        stream.write(_json(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


class Runtime:
    def __init__(self, config=None):
        self.config = Config.parse(config)
        self.enabled = self.config.enabled
        self.rank = self.config.process_id
        self.count = self.config.num_processes
        self.mesh = None
        self.shapes = {}
        self.initial_batch = None
        self.executables = {}
        self.compile_count = 0
        self.timings = []
        self.trace_active = False
        self.profiles = []
        self.locks = {}
        self.manual_svi = False
        self.in_svi_shard = False
        self.branch = "all"
        self.collective_chain = None
        if self.enabled:
            if self.count > 1:
                if jax.distributed.is_initialized():
                    raise RuntimeError("JAX was already initialized outside strategize")
                kwargs = dataclasses.asdict(self.config)
                kwargs.pop("enabled")
                jax.distributed.initialize(**kwargs)
            devices = sorted(jax.devices(), key=lambda d: (d.process_index, d.id))
            if self.count > 1 and (len(jax.local_devices()) != 1 or len(devices) != self.count):
                raise ValueError("Multi-host SVI requires exactly one GPU per process")
            if self.count == 1 and self.config.local_device_ids is not None:
                devices = [jax.local_devices()[i] for i in self.config.local_device_ids]
            # make_mesh's automatic topology construction rejects this GPU pair.
            self.mesh = Mesh(np.array(devices), ("data",), axis_types=(AxisType.Auto,))
        self.rep = None if self.mesh is None else NamedSharding(self.mesh, P())

    @property
    def primary(self):
        return self.rank == 0

    @property
    def device_count(self):
        return 1 if self.mesh is None else self.mesh.size

    def gather_json(self, value):
        if self.count == 1:
            return [value]
        from jax.experimental import multihost_utils as mh
        payload = _json(value).encode()
        lengths = np.asarray(mh.process_allgather(np.array([len(payload)], np.int32))).reshape(-1)
        buf = np.zeros(int(lengths.max()), np.uint8)
        buf[:len(payload)] = np.frombuffer(payload, np.uint8)
        rows = np.asarray(mh.process_allgather(buf)).reshape(len(lengths), -1)
        return [json.loads(row[:int(n)].tobytes()) for row, n in zip(rows, lengths)]

    def agree_status(self, error=None, label="distributed operation"):
        errors = self.gather_json(None if error is None or error == "" else str(error))
        if any(e is not None for e in errors):
            raise RuntimeError(f"{label} failed: " + "; ".join(
                f"rank {i}: {e}" for i, e in enumerate(errors) if e is not None))

    def coordinated(self, fn, label):
        result, error = None, None
        try:
            result = fn()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        self.agree_status(error, label)
        return result

    def require_equal(self, value, label):
        values = self.gather_json(value)
        if any(v != values[0] for v in values[1:]):
            raise RuntimeError(f"Workers disagree on {label}: {values}")

    def broadcast_array(self, value):
        """Small variable-length one-dimensional integer host data."""
        value = np.asarray(value, dtype=np.int32).reshape(-1)
        if self.count == 1:
            return value
        from jax.experimental import multihost_utils as mh
        n = int(mh.broadcast_one_to_all(np.array(value.size, np.int32), self.primary))
        buf = value if self.primary else np.zeros(n, np.int32)
        return np.asarray(mh.broadcast_one_to_all(buf, self.primary))

    @staticmethod
    def _raw_bytes(value):
        if isinstance(value, (bytes, bytearray, memoryview)):
            return np.frombuffer(value, dtype=np.uint8)
        # Accept the old integer-vector transport for existing callers.
        return np.asarray(value, dtype=np.uint8).reshape(-1)

    def broadcast_bytes(self, value):
        """R raw vectors cross reticulate as bytearrays, without int32 expansion."""
        value = self._raw_bytes(value)
        if self.count > 1:
            from jax.experimental import multihost_utils as mh
            n = int(mh.broadcast_one_to_all(np.array(value.size, np.int32), self.primary))
            value = value if self.primary else np.zeros(n, np.uint8)
            value = np.asarray(mh.broadcast_one_to_all(value, self.primary))
        return bytearray(value)

    def primary_json(self, fn, label):
        result = self.coordinated(lambda: fn() if self.primary else None, label)
        return self.gather_json(result)[0]

    def local_tree(self, tree):
        def local(a):
            if not isinstance(a, jax.Array):
                return a
            if not a.is_fully_replicated:
                raise ValueError("Expected replicated training state")
            return a.addressable_data(0)
        return jax.tree.map(local, tree)

    def host_copy(self, tree):
        # An owned copy matters on CPU too: np.asarray may alias a JAX buffer
        # that the next update donates. No GPU state survives in this snapshot.
        return jax.tree.map(lambda a: np.array(a, copy=True), self.local_tree(tree))

    def device_copy(self, tree):
        return jax.tree.map(jnp.asarray, tree)

    def owned_state(self, tree):
        """Detach initialization/warm-start aliases before enabling donation."""
        return jax.tree.map(lambda a: a.copy(), tree)

    def replicate(self, tree, broadcast=False):
        if not self.enabled:
            return tree
        def put(a):
            a = np.asarray(a.addressable_data(0) if isinstance(a, jax.Array) else a)
            if broadcast and self.count > 1:
                from jax.experimental import multihost_utils as mh
                a = np.asarray(mh.broadcast_one_to_all(a, self.primary))
            return jax.make_array_from_process_local_data(self.rep, a, global_shape=a.shape)
        return jax.tree.map(put, tree)

    def place_batch(self, args):
        """Host batch -> process-owned shards; preserve logical shapes before padding.

        Trimming alignment padding *inside* jit preserves NumPyro plate sizes,
        random draws and auxiliary losses, including odd-sized mixed branches.
        The compiler may communicate an indivisible intermediate; profiling
        records that cost rather than silently changing the objective.
        """
        args = {k: np.asarray(v) if v is not None else None for k, v in dict(args).items()}
        self.shapes = {k: tuple(v.shape) for k, v in args.items() if v is not None}
        self.initial_batch = args
        if not self.enabled:
            return {k: jnp.asarray(v) if v is not None else None for k, v in args.items()}
        def put(a):
            if a is None:
                return None
            if a.ndim == 0 or a.size == 0:
                return jax.make_array_from_process_local_data(self.rep, a, global_shape=a.shape)
            padding = (-a.shape[0]) % self.device_count
            if padding:
                a = np.pad(a, [(0, padding)] + [(0, 0)] * (a.ndim - 1))
            shape = a.shape
            local_n = shape[0] // self.count
            local = a[self.rank * local_n:(self.rank + 1) * local_n]
            return jax.make_array_from_process_local_data(
                NamedSharding(self.mesh, P("data")), local, global_shape=shape)
        return {k: put(v) for k, v in args.items()}

    def init_svi(self, svi, key, args, init_params=None):
        started = time.monotonic()
        if os.environ.get("STRATEGIZE_DP_PROFILE_DIR"):
            print(f"SVI rank {self.rank}: initializing the local guide prototype", flush=True)
        if not self.enabled:
            kwargs = dict(args)
        else:
            # Only parameter/global latent shapes are supported; two observed
            # rows suffice for the guide prototype without a global eager pass.
            kwargs = {k: jnp.asarray(v[:2] if v.ndim else v) if v is not None else None
                      for k, v in self.initial_batch.items()}
        if init_params is not None:
            kwargs["init_params"] = init_params
        state = self.coordinated(lambda: svi.init(key, **kwargs), "SVI initialization")
        if os.environ.get("STRATEGIZE_DP_PROFILE_DIR"):
            print(f"SVI rank {self.rank}: broadcasting initialized state after {time.monotonic()-started:.1f}s", flush=True)
        state = self.replicate(state, broadcast=True)
        if os.environ.get("STRATEGIZE_DP_PROFILE_DIR"):
            print(f"SVI rank {self.rank}: initialized state ready after {time.monotonic()-started:.1f}s", flush=True)
        return state

    def owned_positions(self, logical_n):
        """One-based positions, including alignment slots, for the R loader."""
        n = int(logical_n)
        padded = n + (-n) % self.device_count
        local_n = padded // self.count
        return np.arange(self.rank * local_n + 1, (self.rank + 1) * local_n + 1, dtype=np.int32)

    def place_local_batch(self, args, logical_rows):
        args = dict(args)
        rows = dict(logical_rows)
        def prepare():
            host, shapes = {}, {}
            for k, value in args.items():
                if value is None:
                    host[k] = None
                    continue
                a = np.asarray(value)
                n = int(rows["all"] if "all" in rows else
                        rows["single"] if "single" in k else rows["pair"])
                logical = (n,) + a.shape[1:] if a.ndim else ()
                padded = n + (-n) % self.device_count
                if a.ndim and a.shape[0] != padded // self.count:
                    raise ValueError(f"Wrong local ownership for {k}: {a.shape}, global rows {n}")
                host[k], shapes[k] = a, logical
            return host, shapes
        host, self.shapes = self.coordinated(prepare, "materialize local batch")
        self.initial_batch = host
        result = {}
        for k, a in host.items():
            if a is None:
                result[k] = None
                continue
            shape = self.shapes[k]
            global_shape = (shape[0] + (-shape[0]) % self.device_count,) + shape[1:] if shape else ()
            if not a.ndim or a.size == 0:
                local = np.zeros(global_shape, dtype=a.dtype) if a.size == 0 else a
                result[k] = jax.make_array_from_process_local_data(self.rep, local, global_shape=global_shape)
            else:
                result[k] = jax.make_array_from_process_local_data(
                    NamedSharding(self.mesh, P("data")), a, global_shape=global_shape)
        return result

    def stack_batches(self, batches):
        batches = [dict(b) for b in batches]
        return {k: jnp.stack([b[k] for b in batches]) for k in batches[0]
                if batches[0][k] is not None}

    def _trim(self, args, shapes):
        return {k: v[:shapes[k][0]] if v is not None and shapes[k] else v for k, v in args.items()}

    def _argument_shardings(self, args, scan):
        return {k: None if v is None else self.rep if v.size == 0 else NamedSharding(
            self.mesh, P(None, "data") if scan and self.shapes[k] else
            P("data") if self.shapes[k] else P()) for k, v in args.items()}

    def configure_svi_sharding(self):
        self.manual_svi = True

    def set_branch(self, branch):
        if self.in_svi_shard:
            self.branch = branch

    def _branch_rows(self):
        key = "X_single" if self.branch == "single" else "X_left"
        if key not in self.shapes:
            key = "X"
        n = self.shapes[key][0]
        return n, (n + (-n) % self.device_count) // self.device_count

    def schema_uniform(self, key, n_batch, n_units, dtype):
        n, local_n = self._branch_rows()
        if int(n_batch) != local_n:
            raise ValueError("Unexpected schema dropout row layout")
        # Generate the same logical random array as the single-device model.
        full = jax.random.uniform(key, (n, int(n_units)), dtype=dtype)
        full = jnp.pad(full, ((0, local_n * self.device_count - n), (0, 0)))
        return jax.lax.dynamic_slice_in_dim(full, jax.lax.axis_index("data") * local_n, local_n)

    def route_mean(self, weights, mask):
        n, local_n = self._branch_rows()
        repeats = weights.shape[0] // local_n
        if weights.shape[0] % local_n:
            raise ValueError("Unexpected MoE row layout")
        valid = jnp.arange(local_n) + jax.lax.axis_index("data") * local_n < n
        mask = mask * jnp.tile(valid, repeats)[:, None]
        stats = jnp.concatenate((jnp.sum(weights * mask[..., None], axis=(0, 1)), jnp.sum(mask)[None]))
        # The auxiliary objective and its cotangent are identical on every
        # replica. Its backward sum is exactly R * cotangent, requiring no
        # independently scheduled backward collective.
        @jax.custom_vjp
        def total(x):
            return jax.lax.psum(x, "data")
        total.defvjp(lambda x: (jax.lax.psum(x, "data"), None),
                     lambda _, g: (g * self.device_count,))
        # Keep a real forward dependency: an unused tuple component can be
        # pruned, allowing heterogeneous GPU compilers to reorder these sums.
        # For finite routing statistics this is exactly the identity. The
        # predicate has no tangent, so it adds no backward communication.
        stats = jax.lax.optimization_barrier(jnp.where(
            jnp.isfinite(self.collective_chain), stats, jnp.full_like(stats, jnp.nan)))
        stats = total(stats)
        self.collective_chain = jnp.sum(stats)
        return stats[:-1] / jnp.maximum(stats[-1], 1.)

    def _manual_function(self, svi, scan, gradients):
        from jax.flatten_util import ravel_pytree
        from numpyro.infer.svi import _make_loss_fn, SVIState
        def step(state, args):
            moe_config = getattr(svi, "moe_config", None)
            if state.mutable_state is not None and moe_config is None:
                raise ValueError("Data parallel SVI requires a stateless model/guide")
            self.in_svi_shard, self.branch = True, "all"
            self.collective_chain = jnp.array(0., dtype=jnp.float32)
            try:
                rng, step_rng = jax.random.split(state.rng_key)
                # Mean the replica objectives: KL once, sum weighted likelihood
                # across the global batch. Priors and regularizers keep their
                # existing scale; only observation likelihood scales get R.
                scaled = {k: v * self.device_count if k in ("obs_scale", "obs_scale_single") and v is not None else v
                          for k, v in args.items()}
                loss_fn = _make_loss_fn(svi.loss, step_rng, svi.constrain_fn, svi.model,
                                       svi.guide, (), scaled, svi.static_kwargs,
                                       mutable_state=state.mutable_state)
                (loss, mutable), grads = jax.value_and_grad(loss_fn, has_aux=True)(svi.optim.get_params(state.optim_state))
                flat, unravel = ravel_pytree(grads)
                packed = jnp.concatenate((flat, loss.reshape(1)))
                if moe_config is not None:
                    from strategize_moe import _depend
                    packed = _depend(packed, self.collective_chain)
                packed = jax.lax.psum(packed, "data") / self.device_count
                grads, loss = unravel(packed[:-1]), packed[-1]
                if gradients:
                    return grads
                finite = jnp.isfinite(packed).all()
                optim = jax.lax.cond(finite, lambda _: svi.optim.update(grads, state.optim_state, value=loss),
                                     lambda _: state.optim_state, None)
                if moe_config is not None:
                    from strategize_moe import commit_state
                    finite = finite & jnp.all(jnp.stack([jnp.isfinite(a).all() for a in jax.tree.leaves((optim, mutable))]))
                    optim = jax.tree.map(lambda a, b: jnp.where(finite, a, b), optim, state.optim_state)
                    mutable = commit_state(state.mutable_state, mutable, finite, moe_config)
                return SVIState(optim, mutable, rng), jnp.where(finite, loss, jnp.nan)
            finally:
                self.in_svi_shard = False
                self.collective_chain = None
        return (lambda state, args: jax.lax.scan(step, state, args)) if scan else step

    def _compile(self, svi, args, state, scan=False, gradients=False, donate=False):
        shapes = dict(self.shapes)
        key = (id(svi), scan, gradients, donate, tuple(
            (k, shapes.get(k), None if v is None else (v.shape, str(v.dtype)))
            for k, v in args.items()))
        if key in self.executables:
            return self.executables[key]
        def body(s, a):
            return svi.stable_update(s, **self._trim(a, shapes))
        if gradients:
            from numpyro.infer.svi import _make_loss_fn
            def fn(s, a):
                _, rng = jax.random.split(s.rng_key)
                loss_fn = _make_loss_fn(svi.loss, rng, svi.constrain_fn, svi.model,
                                       svi.guide, (), self._trim(a, shapes), svi.static_kwargs,
                                       mutable_state=s.mutable_state)
                (_, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(svi.optim.get_params(s.optim_state))
                return grads
        elif scan:
            def fn(s, a):
                return jax.lax.scan(body, s, a)
        else:
            fn = body
        if self.manual_svi:
            specs = {k: None if v is None else s.spec for k, (v, s) in
                     ((k, (args[k], s)) for k, s in self._argument_shardings(args, scan).items())}
            fn = jax.shard_map(self._manual_function(svi, scan, gradients), mesh=self.mesh,
                               in_specs=(P(), specs), out_specs=P() if gradients else (P(), P()),
                               check_vma=False)
        compile_start = time.monotonic()
        if os.environ.get("STRATEGIZE_DP_PROFILE_DIR"):
            print(f"SVI rank {self.rank}: compiling {'gradients' if gradients else 'scan' if scan else 'update'}", flush=True)
        compiled = self.coordinated(lambda: jax.jit(
            fn, in_shardings=(self.rep, self._argument_shardings(args, scan)),
            out_shardings=self.rep if gradients else (self.rep, self.rep),
            donate_argnums=(0,) if donate and not gradients else ()
        ).lower(state, args).compile(), "compile SVI")
        hlo = compiled.as_text()
        schedule = collective_execution_schedule(hlo)
        profile_dir = os.environ.get("STRATEGIZE_DP_PROFILE_DIR")
        if profile_dir:
            out = Path(profile_dir) / f"rank-{self.rank}"
            out.mkdir(parents=True, exist_ok=True)
            with gzip.open(out / f"executable-{self.compile_count}.hlo.gz", "wt") as stream:
                stream.write(hlo)
            _atomic_json(out / f"collectives-{self.compile_count}.json", schedule)
            print(f"SVI rank {self.rank}: compilation finished after {time.monotonic()-compile_start:.1f}s; checking collectives", flush=True)
        if self.count > 1:
            self.require_equal(hashlib.sha256(_json(schedule).encode()).hexdigest(), "compiled collective order")
        memory = compiled.memory_analysis()
        cost = compiled.cost_analysis() or {}
        self.profiles.append({"compile_seconds": time.monotonic() - compile_start,
            "scan": scan, "gradients": gradients,
            "logical_shapes": shapes,
            "input_shapes": {k: None if v is None else list(v.shape) for k,v in args.items()},
            "state_bytes_per_replica": sum(int(a.size * a.dtype.itemsize) for a in jax.tree.leaves(state)),
            "parameter_count": sum(int(a.size) for a in jax.tree.leaves(svi.optim.get_params(state.optim_state))),
            "cost_analysis": {k: float(cost[k]) for k in ("flops", "transcendentals", "bytes accessed") if k in cost},
            "memory": None if memory is None else {k: int(getattr(memory, k)) for k in
                       ("argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes", "alias_size_in_bytes")}})
        self.executables[key] = compiled
        self.compile_count += 1
        return compiled

    def update(self, svi, state, args, scan=False, donate=False):
        args = dict(args)
        fn = self._compile(svi, args, state, scan=scan, donate=donate)
        self.begin_update()
        start = time.monotonic()
        try:
            result = fn(state, args)
            jax.block_until_ready(result)
            self.record_local_timing(time.monotonic() - start)
            finite = np.isfinite(np.asarray(self.local_tree(result[1]))).all()
            self.agree_status(None if finite else "nonfinite loss", "SVI update")
            if len(self.timings) == 1:
                self.check_replicas(result[1], "first losses")
                if os.environ.get("STRATEGIZE_DP_VERIFY_REPLICAS") == "1":
                    self.check_replicas(result[0], "first optimizer state")
            return result
        except Exception as exc:
            # Post-update validation can fail after donation has consumed the
            # input too. Never let the R scan fallback retry that old state.
            if donate:
                raise RuntimeError(f"Donated SVI update failed; its input state cannot be retried: {exc}") from exc
            raise

    def begin_update(self):
        chunks = os.environ.get("STRATEGIZE_DP_TRACE_CHUNKS", "")
        if not chunks:
            return
        first, _ = map(int, chunks.split(":"))
        if len(self.timings) + 1 == first:
            root = Path(os.environ["STRATEGIZE_DP_PROFILE_DIR"]) / f"rank-{self.rank}" / "trace"
            jax.profiler.start_trace(str(root), create_perfetto_trace=True)
            self.trace_active = True

    def record_local_timing(self, seconds):
        self.timings.append(float(seconds))
        if self.trace_active:
            _, last = map(int, os.environ["STRATEGIZE_DP_TRACE_CHUNKS"].split(":"))
            if len(self.timings) == last:
                jax.profiler.stop_trace()
                self.trace_active = False

    def gradients(self, svi, state, args):
        grads = self._compile(svi, dict(args), state, gradients=True)(state, dict(args))
        leaves = [np.asarray(a) for a in jax.tree.leaves(self.local_tree(grads))]
        n = sum(a.size for a in leaves)
        sq = sum(float(np.sum(np.square(np.where(np.isfinite(a), a, 0).astype(np.float64)))) for a in leaves)
        return {"grad_l2": sq ** .5, "grad_rms": (sq / max(n, 1)) ** .5,
                "grad_max_abs": max((float(np.max(np.abs(np.where(np.isfinite(a), a, 0)))) for a in leaves if a.size), default=0.),
                "grad_n_nonfinite": sum(int(np.sum(~np.isfinite(a))) for a in leaves), "grad_n_elements": n}

    def check_replicas(self, tree, label="replicas"):
        # Qualification can inspect a large optimizer state. Never expand it
        # into Python floats/JSON or stage the whole state in host memory.
        leaves, structure = jax.tree.flatten(self.local_tree(tree))
        self.require_equal((str(structure), [(a.shape, str(a.dtype)) for a in leaves]),
                           label + " structure")
        if self.count > 1:
            from jax.experimental import multihost_utils as mh
        finite, equal = True, True
        for leaf in leaves:
            flat = leaf.reshape(-1)
            chunk_size = max(1, 1024**2 // leaf.dtype.itemsize)
            for start in range(0, flat.size, chunk_size):
                chunk = np.asarray(flat[start:start + chunk_size])
                finite = bool(np.isfinite(chunk).all()) and finite
                if self.count > 1:
                    reference = np.asarray(mh.broadcast_one_to_all(chunk, self.primary))
                    equal = bool(np.allclose(chunk, reference, rtol=1e-5, atol=1e-6)) and equal
        self.agree_status("nonfinite state" if not finite else
                          "copies differ across workers" if not equal else None, label)

    def cache_info(self):
        return {"size": len(self.executables), "compile_count": self.compile_count}

    def clear_cache(self):
        self.executables.clear()

    def metadata(self):
        versions = {p: importlib.metadata.version(p) for p in
                    ("jax", "jaxlib", "numpyro", "optax", "equinox", "orbax-checkpoint")}
        versions.update(numpy=np.__version__, python=platform.python_version())
        return {"enabled": self.enabled, "process_id": self.rank, "process_count": self.count,
                "devices": [str(d) + ":" + d.device_kind for d in jax.devices()],
                "versions": versions, "xla_flags": os.environ.get("XLA_FLAGS", ""),
                "precision": str(jax.config.jax_default_matmul_precision),
                "prng": str(jax.config.jax_default_prng_impl), "x64": bool(jax.config.jax_enable_x64),
                "device_memory": {str(d.id): d.memory_stats() for d in jax.local_devices()},
                "runtime_sha256": hashlib.sha256("".join(
                    _sha(Path(__file__).with_name(name)) for name in
                    ("strategize_distributed.py", "strategize_moe.py", "strategize_optim.py")
                ).encode()).hexdigest()}

    def acquire_lock(self, path):
        def lock():
            import fcntl
            path_obj = Path(path)
            path_obj.mkdir(parents=True, exist_ok=True)
            stream = (path_obj / ".training.lock").open("a")
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                stream.close()
                raise RuntimeError(f"Another trainer owns {path}")
            self.locks[str(path)] = stream
        self.coordinated(lambda: lock() if self.primary else None, "training lock")

    def release_locks(self):
        for stream in self.locks.values():
            stream.close()
        self.locks.clear()

    def _checkpointer(self):
        import orbax.checkpoint as ocp
        options = ocp.options.MultiprocessingOptions(primary_host=0, active_processes={0})
        return ocp.Checkpointer(ocp.PyTreeCheckpointHandler(
            multiprocessing_options=options), multiprocessing_options=options)

    def save_checkpoint(self, path, kind, state, payload):
        if kind not in ("latest", "best"):
            raise ValueError("Checkpoint kind must be latest or best")
        def save():
            root = Path(path) / "full-state"
            root.mkdir(parents=True, exist_ok=True)
            generation = uuid.uuid4().hex
            directory = root / generation
            directory.mkdir()
            leaves, tree = jax.tree.flatten(self.local_tree(state))
            arrays = {f"leaf_{i:06d}": np.asarray(a) for i, a in enumerate(leaves)}
            with self._checkpointer() as writer:
                writer.save(str(directory / "arrays"), arrays)
            with (directory / "metadata.rds").open("wb") as stream:
                stream.write(self._raw_bytes(payload))
            meta = {"schema": "strategize-full-svi-v1", "treedef": str(tree),
                    "leaves": {k: {"shape": list(a.shape), "dtype": a.dtype.str} for k, a in arrays.items()},
                    "versions": self.metadata()["versions"]}
            _atomic_json(directory / "state.json", meta)
            checksums = {str(p.relative_to(directory)): _sha(p) for p in directory.rglob("*") if p.is_file()}
            # Publish a durable generation, including Orbax's array files.
            # Closing a writer alone does not order data before the pointer.
            for relative in checksums:
                with (directory / relative).open("rb") as stream:
                    os.fsync(stream.fileno())
            for subdir in sorted((p for p in directory.rglob("*") if p.is_dir()),
                                 key=lambda p: len(p.parts), reverse=True):
                fd = os.open(subdir, os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            _atomic_json(directory / "COMMITTED.json", {"files": checksums})
            old = json.loads((root / f"{kind}.json").read_text()) if (root / f"{kind}.json").exists() else {}
            _atomic_json(root / f"{kind}.json", {"generation": generation, "previous": old.get("generation")})
            retained = set()
            for name in ("latest", "best"):
                p = root / f"{name}.json"
                if p.exists():
                    pointer = json.loads(p.read_text())
                    retained.add(pointer["generation"])
                    if name == "latest" and pointer.get("previous"):
                        retained.add(pointer["previous"])
            for p in root.iterdir():
                if p.is_dir() and (p / "COMMITTED.json").exists() and p.name not in retained:
                    shutil.rmtree(p)
            return generation
        return self.primary_json(save, "save full SVI checkpoint")

    def _generation(self, path, kind="latest", generation=None):
        root = Path(path) / "full-state"
        if generation is None:
            pointer = root / f"{kind}.json"
            if not pointer.exists():
                return None
            generation = json.loads(pointer.read_text())["generation"]
        if len(generation) != 32 or any(c not in "0123456789abcdef" for c in generation):
            raise ValueError("Invalid checkpoint generation")
        directory = root / generation
        committed = json.loads((directory / "COMMITTED.json").read_text())
        for relative, expected in committed["files"].items():
            p = Path(relative)
            if p.is_absolute() or ".." in p.parts or _sha(directory / p) != expected:
                raise ValueError(f"Checkpoint checksum mismatch: {relative}")
        return directory

    def load_checkpoint_payload(self, path, kind="latest"):
        directory = self.coordinated(lambda: self._generation(path, kind) if self.primary else None,
                                     "read full SVI checkpoint")
        generation = self.gather_json(None if directory is None else directory.name)[0]
        if generation is None:
            return None
        raw = np.frombuffer((directory / "metadata.rds").read_bytes(), np.uint8) if self.primary else np.zeros(0, np.uint8)
        return {"generation": generation, "payload": self.broadcast_bytes(raw)}

    def restore_state(self, path, template, generation):
        def read():
            directory = self._generation(path, generation=generation)
            meta = json.loads((directory / "state.json").read_text())
            if meta["versions"] != self.metadata()["versions"]:
                raise ValueError("Full-state recovery requires matching package versions")
            leaves, structure = jax.tree.flatten(self.local_tree(template))
            if str(structure) != meta["treedef"]:
                raise ValueError("Checkpoint optimizer/SVI structure differs")
            with self._checkpointer() as reader:
                arrays = reader.restore(str(directory / "arrays"))
            restored = []
            for i, leaf in enumerate(leaves):
                key = f"leaf_{i:06d}"
                a = np.asarray(arrays[key])
                if a.shape != np.shape(leaf) or a.dtype != np.asarray(leaf).dtype:
                    raise ValueError(f"Checkpoint shape/dtype differs: {key}")
                restored.append(a)
            return jax.tree.unflatten(structure, restored)
        restored = self.coordinated(lambda: read() if self.primary else self.local_tree(template), "restore full SVI state")
        return self.replicate(restored, broadcast=True) if self.enabled else jax.tree.map(jnp.asarray, restored)

    def save_inference_tree(self, path, tree):
        # Called inside the R primary-call envelope, which coordinates errors.
        if not self.primary:
            raise RuntimeError("Only rank zero may write inference arrays")
        with self._checkpointer() as writer:
            writer.save(str(Path(path).resolve()), jax.tree.map(np.asarray, self.local_tree(tree)))

    def load_inference_tree(self, path, abstract_tree, restore_args=None):
        # The outer R primary-call envelope owns synchronization. Orbax must
        # not start another collective while the other rank waits there.
        if not self.primary:
            raise RuntimeError("Only rank zero may read inference arrays")
        with self._checkpointer() as reader:
            return reader.restore(str(Path(path).resolve()), item=abstract_tree,
                                  restore_args=restore_args)

    def shutdown(self):
        if self.trace_active:
            jax.profiler.stop_trace()
            self.trace_active = False
        profile_dir = os.environ.get("STRATEGIZE_DP_PROFILE_DIR")
        if profile_dir:
            _atomic_json(Path(profile_dir) / f"rank-{self.rank}" / "runtime.json",
                         {"runtime": self.metadata(), "executables": self.profiles, "update_seconds": self.timings})
        self.release_locks()
        if self.count > 1 and jax.distributed.is_initialized():
            from jax.experimental import multihost_utils as mh
            mh.sync_global_devices("strategize-training-complete")
            jax.distributed.shutdown()


_runtime = None


def initialize(config=None):
    global _runtime
    cfg = Config.parse(config)
    if _runtime is not None:
        if cfg != _runtime.config:
            raise RuntimeError("Changing data parallelism requires a fresh process")
        return _runtime
    _runtime = Runtime(dataclasses.asdict(cfg))
    return _runtime


# Collective schedule inspection adapted from sailfast-software.
import re

def collective_schedule(hlo):
    """Collective channel/payload order, independent of kernel names and async wrappers.

    NCCL matches operations by issue order. Separate target compilations can
    reorder independent collectives on heterogeneous GPUs, even for one JAX
    program. Compare the scheduled HLO before launching a training executable.
    """
    computations, current, definitions = [], [], {}
    for line in hlo.splitlines():
        if line and not line[0].isspace() and line.endswith("{"):
            if current:
                computations.append(current)
            current, definitions = [], {}
        definition = re.match(r"\s*(?:ROOT )?(%[\w.-]+) = (.+?) [\w-]+\(", line)
        if definition:
            definitions[definition.group(1)] = definition.group(2)
        match = re.search(r"\b(all-reduce|all-gather|reduce-scatter|all-to-all|collective-permute)(?:-start)?\((.*?)\), .*?channel_id=(\d+)", line)
        if match:
            payloads = re.findall(r"\b(?:pred|[sufc]\d+|bf16)\[[^\]]*\](?:\{[^}]*\})?", match.group(2))
            if not payloads:
                # Executable HLO normally omits operand types. Resolve names
                # from this computation instead of silently hashing no shapes.
                names = re.findall(r"%[\w.-]+", match.group(2))
                if not names or any(name not in definitions for name in names):
                    raise ValueError(f"cannot resolve collective operand types: {line.strip()}")
                payloads = [definitions[name] for name in names]
            reducer = re.search(r"to_apply=%([\w.-]+)", line)
            # A manual psum can reuse one channel for many reductions. Their
            # pre-optimization reducer IDs distinguish equally shaped buffers.
            origin = reducer.group(1).split(".clone")[0] if reducer else None
            current.append([match.group(1), int(match.group(3)), payloads, origin])
    if current:
        computations.append(current)
    # Definitions can appear in different textual order; order within each
    # communicating computation must agree. Channel IDs identify those sites.
    return sorted(computations, key=lambda c: json.dumps(sorted(c, key=lambda op: op[1])))


def collective_execution_schedule(hlo):
    """Include communicating calls/loops at their scheduled call sites.

    Comparing bodies separately misses a scan moving across a collective in
    its caller. Computation names and definition order are compiler details;
    only reachable communication and its control-flow placement matter here.
    """
    computations, current, entry = {}, None, None
    for line in hlo.splitlines():
        header = re.match(r"^(ENTRY )?%([\w.-]+) .*\{$", line)
        if header:
            current = header.group(2)
            computations[current] = [line]
            if header.group(1):
                entry = current
        elif current is not None:
            computations[current].append(line)
            if line == "}":
                current = None
    if entry is None:
        raise ValueError("cannot find the compiled HLO entry computation")
    memo = {}

    def visit(name):
        if name in memo:
            return memo[name]
        lines = computations[name]
        sites = collective_schedule("\n".join(lines))
        sites = iter(sites[0] if sites else ())
        result = []
        for line in lines:
            if re.search(r"\b(all-reduce|all-gather|reduce-scatter|all-to-all|collective-permute)(?:-start)?\(", line):
                result.append(next(sites))
                continue
            children = re.findall(r"\b(condition|body|calls|to_apply)=%([\w.-]+)", line)
            branches = re.search(r"branch_computations=\{([^}]+)\}", line)
            if branches:
                children += [("branch", child) for child in re.findall(r"%([\w.-]+)", branches.group(1))]
            nested = [[role, visit(child)] for role, child in children]
            if any(schedule for _, schedule in nested):
                kind = "while" if re.search(r"\bwhile\(", line) else "call"
                result.append([kind, nested])
        memo[name] = result
        return result

    return visit(entry)
