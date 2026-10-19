"""DDIM sampling with separated classifier-free guidance, as upstream Kimodo.

Upstream (``kimodo/model/diffusion.py``, ``cfg.py``, ``kimodo_model.py``):

- a cosine schedule over 1000 base steps, in float32;
- ``N`` sampling steps at base timesteps ``round(i * 999 / (N - 1))``;
- deterministic DDIM (eta 0) starting from one Gaussian draw;
- each step evaluates three branches in one batch (text only, guides only,
  neither) and combines them as ``u + w_text (text - u) + w_guide (guide - u)``.

The denoiser is injected, so this module never imports ONNX Runtime.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Generator
from dataclasses import dataclass
from functools import cache

import numpy as np

from ...errors import Cancelled, InvalidRequest
from . import rep

BASE_STEPS = 1000
DEFAULT_TEXT_WEIGHT = 2.0
DEFAULT_GUIDE_WEIGHT = 2.0
BRANCHES = 3


@cache
def _base_alphas_cumprod() -> np.ndarray:
    def alpha_bar(t: float) -> float:
        return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = np.array(
        [
            min(1 - alpha_bar((i + 1) / BASE_STEPS) / alpha_bar(i / BASE_STEPS), 0.999)
            for i in range(BASE_STEPS)
        ],
        dtype=np.float32,
    )
    values = np.cumprod(np.float32(1.0) - betas, dtype=np.float32)
    values.setflags(write=False)
    return values


@dataclass(frozen=True)
class Schedule:
    """Per sampling step: base timestep and the DDIM coefficients."""

    timesteps: np.ndarray
    alphas_cumprod: np.ndarray
    alphas_cumprod_prev: np.ndarray

    @property
    def steps(self) -> int:
        return len(self.timesteps)


def schedule(steps: int) -> Schedule:
    if not 2 <= int(steps) <= BASE_STEPS:
        raise InvalidRequest(f"Sampling steps must be 2 to {BASE_STEPS}, not {steps}")
    steps = int(steps)
    stride = (BASE_STEPS - 1) / (steps - 1)
    timesteps = np.minimum(
        np.round(np.arange(steps, dtype=np.float64) * stride), BASE_STEPS - 1
    ).astype(np.int64)
    selected = _base_alphas_cumprod()[timesteps]
    # Upstream rebuilds the cumulative products from per-step betas in
    # float32; repeat that so the coefficients match bit for bit.
    previous = np.concatenate((np.ones(1, dtype=np.float32), selected[:-1]))
    alphas = np.float32(1.0) - (np.float32(1.0) - selected / previous)
    cumulative = np.maximum(np.cumprod(alphas, dtype=np.float32), np.float32(1e-9))
    cumulative_prev = np.concatenate((np.ones(1, dtype=np.float32), cumulative[:-1]))
    return Schedule(timesteps, cumulative, cumulative_prev)


def ddim_step(
    noisy: np.ndarray, predicted_clean: np.ndarray, plan: Schedule, index: int
) -> np.ndarray:
    """The next, less noisy motion from step ``index`` (counting down to 0)."""
    alpha = plan.alphas_cumprod[index]
    sqrt_recip = np.float32(1.0) / np.sqrt(alpha)
    sqrt_recip_m1 = np.float32(1.0) / np.sqrt(alpha / (np.float32(1.0) - alpha))
    noise = (sqrt_recip * noisy - predicted_clean) / sqrt_recip_m1
    previous = plan.alphas_cumprod_prev[index]
    return (
        predicted_clean * np.sqrt(previous)
        + np.sqrt(np.float32(1.0) - previous) * noise
    ).astype(np.float32)


def combine_guidance(
    branches: np.ndarray, text_weight: float, guide_weight: float
) -> np.ndarray:
    """Separated CFG over the [text, guides, unconditional] batch."""
    text, guides, unconditional = branches[0], branches[1], branches[2]
    return (
        unconditional
        + np.float32(text_weight) * (text - unconditional)
        + np.float32(guide_weight) * (guides - unconditional)
    ).astype(np.float32)


@dataclass(frozen=True)
class DenoiseInputs:
    """One three-branch denoiser call; names follow the ONNX graph."""

    motion: np.ndarray
    observed_motion: np.ndarray
    motion_valid: np.ndarray
    constraint_mask: np.ndarray
    text_embedding: np.ndarray
    timestep: np.ndarray
    first_heading_angle: np.ndarray


DenoiseFn = Callable[[DenoiseInputs], np.ndarray]


@dataclass(frozen=True)
class SampleProgress:
    step: int
    total: int


def text_tokens(embedding: np.ndarray | None) -> np.ndarray:
    """The model's [50, 4096] text input: the pooled vector, then zeros."""
    tokens = np.zeros((rep.TEXT_TOKENS, rep.TEXT_WIDTH), dtype=np.float32)
    if embedding is not None:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vector.shape != (rep.TEXT_WIDTH,) or not np.all(np.isfinite(vector)):
            raise InvalidRequest(
                f"Text embeddings must be {rep.TEXT_WIDTH} finite values"
            )
        tokens[0] = vector
    return tokens


def initial_noise(seed: int, frame_count: int) -> np.ndarray:
    """The starting Gaussian draw [T, 369] for a seed (NumPy PCG64)."""
    generator = np.random.Generator(np.random.PCG64(int(seed)))
    return generator.standard_normal((frame_count, rep.FEATURE_COUNT), dtype=np.float32)


def iter_sample(
    denoise: DenoiseFn,
    noise: np.ndarray,
    observed_motion: np.ndarray,
    constraint_mask: np.ndarray,
    text: np.ndarray,
    first_heading_angle: float,
    steps: int,
    text_weight: float = DEFAULT_TEXT_WEIGHT,
    guide_weight: float = DEFAULT_GUIDE_WEIGHT,
    cancelled: Callable[[], bool] | None = None,
) -> Generator[SampleProgress, None, np.ndarray]:
    """Sample normalised frames [T, 369], yielding after every denoiser call.

    ``observed_motion`` must already be normalised; ``text`` is the [50, 4096]
    token input from ``text_tokens``. The generator's return value is the
    final clean motion.
    """
    motion = np.asarray(noise, dtype=np.float32)
    count = len(motion)
    if (
        motion.shape != (count, rep.FEATURE_COUNT)
        or not 1 <= count <= rep.MAX_WINDOW_FRAMES
    ):
        raise InvalidRequest(
            f"Noise must have shape [T, {rep.FEATURE_COUNT}] with T up to "
            f"{rep.MAX_WINDOW_FRAMES}"
        )
    observed = np.asarray(observed_motion, dtype=np.float32)
    mask = np.asarray(constraint_mask, dtype=bool)
    if observed.shape != motion.shape or mask.shape != motion.shape:
        raise InvalidRequest("Observed motion and mask must match the noise shape")
    tokens = np.asarray(text, dtype=np.float32)
    if tokens.shape != (rep.TEXT_TOKENS, rep.TEXT_WIDTH):
        raise InvalidRequest("Text input must have shape [50, 4096]")
    plan = schedule(steps)
    no_mask = np.zeros_like(mask)
    constant = {
        "observed_motion": np.broadcast_to(observed, (BRANCHES, *observed.shape)),
        "motion_valid": np.ones((BRANCHES, count), dtype=bool),
        "constraint_mask": np.stack((no_mask, mask, no_mask)),
        "text_embedding": np.stack(
            (tokens, np.zeros_like(tokens), np.zeros_like(tokens))
        ),
        "first_heading_angle": np.full(BRANCHES, first_heading_angle, dtype=np.float32),
    }
    for done, index in enumerate(reversed(range(plan.steps)), 1):
        if cancelled is not None and cancelled():
            raise Cancelled("Motion generation was cancelled")
        inputs = DenoiseInputs(
            motion=np.broadcast_to(motion, (BRANCHES, *motion.shape)),
            timestep=np.full(BRANCHES, plan.timesteps[index], dtype=np.int64),
            **constant,
        )
        branches = np.asarray(denoise(inputs), dtype=np.float32)
        if branches.shape != (BRANCHES, *motion.shape):
            raise InvalidRequest(
                f"The denoiser returned {branches.shape}, expected "
                f"{(BRANCHES, *motion.shape)}"
            )
        predicted = combine_guidance(branches, text_weight, guide_weight)
        motion = ddim_step(motion, predicted, plan, index)
        yield SampleProgress(done, plan.steps)
    return motion


def sample(*args, **kwargs) -> np.ndarray:
    """Drain ``iter_sample`` and return the clean motion."""
    iterator = iter_sample(*args, **kwargs)
    while True:
        try:
            next(iterator)
        except StopIteration as stop:
            return stop.value


__all__ = [
    "BASE_STEPS",
    "BRANCHES",
    "DEFAULT_GUIDE_WEIGHT",
    "DEFAULT_TEXT_WEIGHT",
    "DenoiseFn",
    "DenoiseInputs",
    "SampleProgress",
    "Schedule",
    "combine_guidance",
    "ddim_step",
    "initial_noise",
    "iter_sample",
    "sample",
    "schedule",
    "text_tokens",
]
