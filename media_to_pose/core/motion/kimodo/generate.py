"""Generate a whole motion: windows, guides, sampling and stitching.

Everything here is NumPy and runs on a worker thread. The denoiser and the
text embeddings are passed in, so the same code serves the ONNX backend and
the deterministic stub used in tests.

Coordinates: guides and the result are in one Kimodo-space scene frame. Each
window is moved so that it starts at the ground origin, as the model was
trained; its initial facing is passed as ``first_heading_angle``. The result
is moved back afterwards.
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from ...errors import InvalidRequest
from .. import soma
from . import guides as guide_types
from . import rep, sampler, windows


@dataclass(frozen=True)
class GenerationRequest:
    """Inputs for one generated motion at 30 fps.

    ``frame_count`` frames are produced. ``guides`` use frame indices of the
    whole generation. ``start_xz`` and ``start_heading`` place frame 0 when
    no guide does. ``embeddings`` maps each segment's prompt to its pooled
    [4096] vector; empty prompts need none.
    """

    frame_count: int
    stats: rep.FeatureStats
    guides: tuple[guide_types.Guide, ...] = ()
    segments: tuple[windows.Segment, ...] = ()
    embeddings: Mapping[str, np.ndarray] = field(default_factory=dict)
    start_xz: tuple[float, float] = (0.0, 0.0)
    start_heading: float = 0.0
    seed: int = 0
    steps: int = 50
    text_weight: float = sampler.DEFAULT_TEXT_WEIGHT
    guide_weight: float = sampler.DEFAULT_GUIDE_WEIGHT

    def __post_init__(self) -> None:
        if self.frame_count < 2:
            raise InvalidRequest("A generation needs at least two frames")
        for guide in self.guides:
            if not 0 <= guide.frame < self.frame_count:
                raise InvalidRequest(
                    f"Guide frame {guide.frame} is outside 0..{self.frame_count - 1}"
                )


@dataclass(frozen=True)
class GenerationProgress:
    window: int
    windows: int
    step: int
    steps: int

    @property
    def fraction(self) -> float:
        return (self.window + self.step / max(self.steps, 1)) / max(self.windows, 1)


@dataclass(frozen=True)
class GeneratedMotion:
    """Unnormalised features [T, 369] and their decoded frames."""

    features: np.ndarray
    frames: rep.KimodoFrames
    windows: tuple[windows.Window, ...]
    seed: int


def _window_guides(
    request: GenerationRequest, window: windows.Window, origin: np.ndarray
) -> list[guide_types.Guide]:
    return [
        guide_types.translate_guide(guide, -origin, -window.start)
        for guide in request.guides
        if window.start <= guide.frame < window.stop
    ]


def _embedding(request: GenerationRequest, prompt: str) -> np.ndarray | None:
    if not prompt.strip():
        return None
    try:
        return request.embeddings[prompt]
    except KeyError as exc:
        raise InvalidRequest(f"No text embedding for the prompt {prompt!r}") from exc


def _start_pose(request: GenerationRequest) -> tuple[np.ndarray, float]:
    """Where frame 0 stands and faces: its guide, if any, else the request's."""
    for guide in request.guides:
        if guide.frame != 0:
            continue
        if isinstance(guide, guide_types.RootGuide):
            heading = request.start_heading if guide.heading is None else guide.heading
            return guide.xz.copy(), float(heading)
        root = (
            guide.smooth_root_xz
            if guide.smooth_root_xz is not None
            else guide.positions[soma.ROOT, [0, 2]]
        )
        return np.asarray(root, dtype=np.float64), float(
            soma.heading_angle(guide.positions)
        )
    return np.asarray(request.start_xz, dtype=np.float64), float(request.start_heading)


def iter_generate(
    denoise: sampler.DenoiseFn,
    request: GenerationRequest,
    cancelled: Callable[[], bool] | None = None,
) -> Generator[GenerationProgress, None, GeneratedMotion]:
    """Sample every window in order; returns the stitched motion."""
    plan = windows.plan_windows(request.frame_count, request.segments or None)
    segments = request.segments or (windows.Segment(0, request.frame_count),)
    generator = np.random.Generator(np.random.PCG64(int(request.seed)))
    stats = request.stats
    origin, heading = _start_pose(request)
    stitched: np.ndarray | None = None
    for index, window in enumerate(plan):
        transition: list[guide_types.Guide] = []
        if window.transition:
            if stitched is None:
                raise InvalidRequest("A transition window needs a previous window")
            tail = stitched[window.start : window.start + windows.TRANSITION_FRAMES]
            tail_frames = rep.decode(tail)
            origin = tail_frames.smooth_root[0, [0, 2]].copy()
            heading = float(soma.heading_angle(tail_frames.positions[0]))
            transition = guide_types.guides_from_frames(
                tail_frames.translated(-origin), range(windows.TRANSITION_FRAMES)
            )
        window_guides = transition + _window_guides(request, window, origin)
        observed, mask = guide_types.compile_guides(window_guides, window.frame_count)
        noise = generator.standard_normal(
            (window.frame_count, rep.FEATURE_COUNT), dtype=np.float32
        )
        prompt = segments[window.segment].prompt
        sampling = sampler.iter_sample(
            denoise,
            noise,
            stats.normalize(observed.astype(np.float32)),
            mask,
            sampler.text_tokens(_embedding(request, prompt)),
            heading,
            request.steps,
            request.text_weight,
            request.guide_weight,
            cancelled,
        )
        while True:
            try:
                progress = next(sampling)
            except StopIteration as stop:
                result = stop.value
                break
            yield GenerationProgress(index, len(plan), progress.step, progress.total)
        features = stats.denormalize(np.asarray(result, dtype=np.float64))
        features[:, rep.SMOOTH_ROOT.start] += origin[0]
        features[:, rep.SMOOTH_ROOT.start + 2] += origin[1]
        if stitched is None:
            stitched = features
        else:
            overlap = windows.TRANSITION_FRAMES if window.transition else 0
            stitched = windows.crossfade(
                stitched[: window.start + overlap], features, overlap
            )
    if stitched is None or len(stitched) != request.frame_count:
        raise InvalidRequest("Generated windows do not cover the requested frames")
    return GeneratedMotion(
        stitched, rep.decode(stitched), tuple(plan), int(request.seed)
    )


def generate(
    denoise: sampler.DenoiseFn,
    request: GenerationRequest,
    cancelled: Callable[[], bool] | None = None,
) -> GeneratedMotion:
    iterator = iter_generate(denoise, request, cancelled)
    while True:
        try:
            next(iterator)
        except StopIteration as stop:
            return stop.value


def window_prompts(segments: Sequence[windows.Segment]) -> list[str]:
    """Distinct non-empty prompts that need embeddings, in order."""
    seen: list[str] = []
    for segment in segments:
        if segment.prompt.strip() and segment.prompt not in seen:
            seen.append(segment.prompt)
    return seen


__all__ = [
    "GeneratedMotion",
    "GenerationProgress",
    "GenerationRequest",
    "generate",
    "iter_generate",
    "window_prompts",
]
