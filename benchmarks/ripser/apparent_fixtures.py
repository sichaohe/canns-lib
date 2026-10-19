#!/usr/bin/env python3
"""
Fixture generator for the apparent-pair benchmark.

Split into its own module on purpose: the benchmark driver runs every cell in a
SUBPROCESS (so env vars and the STATS stderr line are isolated per cell and an
OOM in one cell cannot kill the run), and that subprocess has to rebuild the
exact same point cloud without importing the driver's argparse code.

Every fixture is a pure function of (name, n): no global RNG state, and the
seed is derived from n so a given cell is reproducible on any machine. That is
what makes a rerun on a different box a comparison rather than a coincidence.
"""
from __future__ import annotations

import numpy as np

FIXTURES = ["circle", "sphere", "torus", "random", "clusters", "grid", "swiss_roll"]


def _clusters(n: int, rng: np.random.Generator, k: int = 4, d: int = 3) -> np.ndarray:
    """Gaussian mixture. Hand-rolled because tadasets 0.2.2 has no `clusters`."""
    centres = rng.normal(scale=6.0, size=(k, d))
    counts = rng.multinomial(n, np.full(k, 1.0 / k))
    pts = [c + rng.normal(scale=1.0, size=(int(cnt), d))
           for c, cnt in zip(centres, counts) if cnt > 0]
    X = np.vstack(pts)
    # deterministic shuffle so the k-blocks are interleaved, not sequential
    return X[rng.permutation(X.shape[0])][:n]


def make_fixture(name: str, n: int) -> np.ndarray:
    # tadasets is a standalone top-level package; the 0.2.x series takes
    # `seed=` (NOT `rng=`), and has no `clusters`.
    import tadasets

    if n < 10:
        raise ValueError(f"n={n} too small for a meaningful filtration")
    seed = 0xC0FFEE + n

    if name == "circle":
        return np.asarray(tadasets.dsphere(n=n, d=1, noise=0.05, seed=seed),
                          dtype=np.float64)
    if name == "sphere":
        return np.asarray(tadasets.dsphere(n=n, d=2, noise=0.05, seed=seed),
                          dtype=np.float64)
    if name == "torus":
        return np.asarray(tadasets.torus(n=n, c=2, a=1, noise=0.05, seed=seed),
                          dtype=np.float64)
    if name == "random":
        return np.asarray(np.random.default_rng(seed).standard_normal((n, 3)),
                          dtype=np.float64)
    if name == "clusters":
        return _clusters(n, np.random.default_rng(seed))
    if name == "grid":
        rng = np.random.default_rng(seed)
        m = int(np.ceil(n ** (1 / 3)))
        g = np.stack(np.meshgrid(*([np.arange(m)] * 3), indexing="ij"), -1)
        g = g.reshape(-1, 3).astype(np.float64)
        g = g + rng.normal(scale=1e-3, size=g.shape)
        if g.shape[0] >= n:
            return g[:n]
        extra = g[rng.integers(0, g.shape[0], n - g.shape[0])]
        return np.vstack([g, extra])
    if name == "swiss_roll":
        # tadasets 0.2.2 signature is (n, r, noise, ambient, seed) -- no `d`,
        # the manifold is 2D by default. Keep `ambient` unset so it stays 2D.
        return np.asarray(tadasets.swiss_roll(n=n, noise=0.05, seed=seed),
                          dtype=np.float64)
    raise ValueError(f"unknown fixture {name!r}")
