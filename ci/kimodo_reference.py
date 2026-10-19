"""Record the Kimodo reference fixtures in tests/fixtures/kimodo/ (phase K0).

Needs a separate Python 3.13 environment with CPU torch, the upstream sources
and the models; none of it ships or runs in CI:

- ``nv-tlabs/kimodo`` at 1aece8c124d73d255ceff5086d983b844c9f4e94 (on
  ``PYTHONPATH``) with hydra-core, omegaconf, einops, scipy, safetensors,
  ``transformers==5.1.0`` and ``peft``;
- ``nvidia/Kimodo-SOMA-RP-v1.1`` at 6c9233af1180b8151e3c4703477104af5dce9dd5,
  as ``<checkpoints>/Kimodo-SOMA-RP-v1.1``;
- ``KitsuMate/Llama-3-LLM2Vec-MNTP-Supervised-ONNX`` at
  883615966398adc12e787d9b3a2dc4da26d4b44c, with onnxruntime and tokenizers.

    python ci/kimodo_reference.py all --checkpoints DIR --encoder DIR

``embeddings`` must run before ``fixture``: the parity run uses the int4
embedding of the first prompt. No NVIDIA weights or statistics are written:
the observed motion is zero wherever the constraint mask is false.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "kimodo"
MODEL_NAME = "Kimodo-SOMA-RP-v1.1"
HEADER = "<|start_header_id|>user<|end_header_id|>\n\n"
EOT = "<|eot_id|>"
MARKER = "!@#$%^&*()"
ENCODER_FILE = "onnx/model_int4_b128_fp32act_s64.onnx"
SEQUENCE = 64

FRAMES, STEPS, CHECK_STEPS = 60, 25, (0, 12, 24)
SOURCE_SEED, NOISE_SEED = 1, 2024
FULL_BODY_FRAME, HANDS_FRAME, ROOT_FRAMES = 40, 59, np.arange(0, 30)
HANDS = ("LeftHand", "RightHand")

PROMPTS = (
    "A person walks forward and waves.",
    "A person jogs in a circle.",
    "A person jumps forward with both feet.",
    "A person sits down on the floor and stands up again.",
    "A person kicks with the right leg.",
    "A person turns around and walks back.",
    "A person crouches and sneaks forward.",
    "A person throws a ball with the right hand.",
    "A person dances happily.",
    "A person stumbles and catches their balance.",
    "A person walks to the left.",
    "A person raises both arms above the head.",
    "A person picks something up from the ground.",
    "A person bows.",
    "A person punches with the left hand, then the right hand.",
    "A person walks slowly like a zombie.",
    "A person climbs up stairs.",
    "A person stretches their arms and back.",
    "A person runs forward and stops suddenly.",
    "A person waves with the left hand while standing still.",
)

TOKEN_STRINGS = (
    "A person walks forward and waves.",
    "",
    " ",
    "jump",
    "A person walks to NYC.",
    "a person walks to nyc.",
    "Une personne marche à gauche et salue.",
    "Eine Person läuft rückwärts.",
    "Una persona baila flamenco con pasión.",
    "Человек идёт вперёд и машет рукой.",
    "人が前に歩いて手を振る。",
    "사람이 앞으로 걸어간다.",
    "一个人向前走然后挥手。",
    "شخص يمشي إلى الأمام",
    "A person dances 💃 and laughs 😂.",
    "Café naïve résumé coöperate",
    "tab\tseparated\tand\nnew line",
    "Multiple   spaces    between words",
    "A person walks 10 steps, turns 90 degrees and jumps 2 times.",
    "!!! leading punctuation",
    "trailing punctuation ---",
    "Quotes \"inside\" and 'single' ones.",
    "<|eot_id|> special token text",
    "!@#$%^&*() marker inside",
    "C:\\path\\to\\file and http://example.com/x?y=1",
    "UPPERCASE PROMPT SHOUTING",
    "camelCaseWords and snake_case_words",
    "A person " + "walks and turns and " * 20 + "stops.",
    "Ünïcödé ßtraße ﬁligature",
    "A person\u00a0walks\u2009with odd\u200bspaces.",
)


def _sanitize(text: str) -> str:
    from kimodo.sanitize import sanitize_text

    return sanitize_text(text)


def _upstream_tokens(tokenizer, sanitised: str) -> tuple[list[int], int]:
    """LLM2Vec.prepare_for_tokenization + tokenize for one prompt."""
    text = HEADER + (MARKER + sanitised).strip() + EOT
    parts = text.split(MARKER)
    pooled_text = parts[1] if len(parts) > 1 else ""
    ids = tokenizer(["".join(parts)], truncation=True, max_length=512)["input_ids"][0]
    pooled = tokenizer(
        [pooled_text], add_special_tokens=False, truncation=True, max_length=512
    )["input_ids"][0]
    return [int(value) for value in ids], len(pooled)


def record_tokens(encoder: Path) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(encoder))
    rows = [(raw, _sanitize(raw)) for raw in TOKEN_STRINGS]
    encoded = [_upstream_tokens(tokenizer, sanitised) for _, sanitised in rows]
    lengths = [len(ids) for ids, _ in encoded]
    np.savez_compressed(
        FIXTURES / "reference_tokens.npz",
        schema_version=np.int64(1),
        strings=np.array([raw for raw, _ in rows]),
        sanitised=np.array([sanitised for _, sanitised in rows]),
        ids=np.concatenate([np.array(ids, np.int64) for ids, _ in encoded]),
        offsets=np.cumsum([0, *lengths]).astype(np.int64),
        pooled=np.array([pooled for _, pooled in encoded], np.int64),
        bos_id=np.int64(tokenizer.bos_token_id),
        eot_id=np.int64(tokenizer.convert_tokens_to_ids(EOT)),
        pad_id=np.int64(tokenizer.eos_token_id),
        template=np.array("<|begin_of_text|>" + HEADER + "{prompt}" + EOT),
    )
    print("tokens:", len(rows), "strings, longest", max(lengths))


def record_embeddings(encoder: Path) -> None:
    import onnxruntime as ort
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(encoder))
    pad = int(tokenizer.eos_token_id)
    session = ort.InferenceSession(
        str(encoder / ENCODER_FILE), providers=["CPUExecutionProvider"]
    )
    sanitised = [_sanitize(prompt) for prompt in PROMPTS]
    vectors = []
    for text in sanitised:
        ids, pooled = _upstream_tokens(tokenizer, text)
        feed = {
            "input_ids": np.full((1, SEQUENCE), pad, np.int64),
            "attention_mask": np.zeros((1, SEQUENCE), np.int64),
            "pooling_mask": np.zeros((1, SEQUENCE), np.int64),
        }
        feed["input_ids"][0, SEQUENCE - len(ids) :] = ids
        feed["attention_mask"][0, SEQUENCE - len(ids) :] = 1
        feed["pooling_mask"][0, SEQUENCE - pooled :] = 1
        vectors.append(session.run(None, feed)[0][0])
    np.savez_compressed(
        FIXTURES / "reference_embeddings.npz",
        schema_version=np.int64(1),
        variant=np.array("int4_b128_fp32act_s64"),
        prompts=np.array(PROMPTS),
        sanitised=np.array(sanitised),
        embeddings=np.array(vectors, np.float32),
    )
    print("embeddings:", len(vectors))


class _NoEncoder:
    def __call__(self, texts):
        raise RuntimeError("Text features are always passed in")

    def to(self, *_args, **_kwargs):
        return self

    def eval(self):
        return self


def _noise(seed: int) -> np.ndarray:
    generator = np.random.Generator(np.random.PCG64(seed))
    return generator.standard_normal((FRAMES, 369), dtype=np.float32)


def _sample(model, noise, embedding, constraints, trace: bool) -> dict:
    """kimodo_model._generate with the noise injected and every step recorded."""
    import torch
    from kimodo.motion_rep.feature_utils import length_to_mask

    lengths = torch.tensor([FRAMES])
    pad_mask = length_to_mask(lengths)
    text = torch.from_numpy(embedding)[None, None]
    text_mask = torch.ones(1, 1, dtype=torch.bool)
    heading = torch.zeros(1)
    weights = (2.0, 2.0)
    observed = mask = None
    if constraints:
        observed, bool_mask = (
            model.motion_rep.create_conditions_from_constraints_batched(
                [constraints], lengths, to_normalize=True, device="cpu"
            )
        )
        mask = 1 * bool_mask
    use_timesteps, step_map = model.diffusion.space_timesteps(STEPS)
    model.diffusion.calc_diffusion_vars(use_timesteps)
    motion = torch.from_numpy(noise)[None].clone()
    record: dict[str, list] = {
        k: [] for k in ("x_in", "pred", "branches", "x_out", "t")
    }
    for index in reversed(range(STEPS)):
        step = torch.tensor([index])
        timestep = step_map[step]
        predicted = model.denoiser(
            weights,
            motion,
            pad_mask,
            text,
            text_mask,
            timestep,
            heading,
            mask,
            observed,
            cfg_type="separated",
        )
        if trace:
            zeros = torch.zeros_like(motion)
            branch_mask = mask if mask is not None else zeros
            branch_observed = observed if observed is not None else zeros
            branches = model.denoiser.model(
                torch.cat([motion] * 3),
                torch.cat([pad_mask] * 3),
                torch.cat([text, 0 * text, 0 * text]),
                torch.cat([text_mask, False * text_mask, False * text_mask]),
                torch.cat([timestep] * 3),
                first_heading_angle=torch.cat([heading] * 3),
                motion_mask=torch.cat([0 * branch_mask, branch_mask, 0 * branch_mask]),
                observed_motion=torch.cat([branch_observed] * 3),
            )
            record["branches"].append(branches.numpy().copy())
            record["x_in"].append(motion[0].numpy().copy())
            record["pred"].append(predicted[0].numpy().copy())
            record["t"].append(int(timestep[0]))
        motion = model.sampler(use_timesteps, motion, predicted, step)
        if trace:
            record["x_out"].append(motion[0].numpy().copy())
    result = {k: np.asarray(v) for k, v in record.items()}
    result["final"] = motion[0].numpy().copy()
    if mask is not None and observed is not None:
        result["mask"] = mask[0].numpy().astype(bool)
        result["observed"] = observed[0].numpy()
    return result


def _decode(model, normalised: np.ndarray) -> dict:
    import torch

    output = model.motion_rep.inverse(
        torch.from_numpy(normalised)[None], is_normalized=True
    )
    return {key: value[0].numpy() for key, value in output.items()}


def record_fixture(checkpoints: Path) -> None:
    import torch

    os.environ["CHECKPOINT_DIR"] = str(checkpoints)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from kimodo.constraints import (
        EndEffectorConstraintSet,
        FullBodyConstraintSet,
        Root2DConstraintSet,
    )
    from kimodo.model.load_model import load_model

    torch.set_grad_enabled(False)
    model = load_model(MODEL_NAME, device="cpu", text_encoder=_NoEncoder()).eval()
    skeleton = model.skeleton
    with np.load(FIXTURES / "reference_embeddings.npz", allow_pickle=False) as data:
        prompt = str(data["prompts"][0])
        embedding = data["embeddings"][0].astype(np.float32)

    # The guides come from an unconstrained sample of the same prompt.
    source = _decode(
        model, _sample(model, _noise(SOURCE_SEED), embedding, None, False)["final"]
    )
    positions = torch.from_numpy(source["posed_joints"])
    rotations = torch.from_numpy(source["global_rot_mats"])
    root_xz = torch.from_numpy(source["smooth_root_pos"][:, [0, 2]])
    full_body = np.array([FULL_BODY_FRAME])
    hands = np.array([HANDS_FRAME])
    constraints = [
        Root2DConstraintSet(
            skeleton, torch.from_numpy(ROOT_FRAMES), root_xz[ROOT_FRAMES]
        ),
        FullBodyConstraintSet(
            skeleton,
            torch.from_numpy(full_body),
            positions[full_body],
            rotations[full_body],
        ),
        EndEffectorConstraintSet(
            skeleton,
            torch.from_numpy(hands),
            positions[hands],
            rotations[hands],
            None,
            joint_names=list(HANDS),
        ),
    ]
    noise = _noise(NOISE_SEED)
    run = _sample(model, noise, embedding, constraints, True)
    final = run["final"]
    decoded = _decode(model, final)
    rows = list(CHECK_STEPS)
    unnormalised = model.motion_rep.unnormalize(torch.from_numpy(final)[None])[
        0
    ].numpy()
    np.savez_compressed(
        FIXTURES / "reference_60f.npz",
        schema_version=np.int64(1),
        prompt=np.array(prompt),
        noise_seed_pcg64=np.int64(NOISE_SEED),
        noise=noise,
        text_embedding=embedding,
        first_heading_angle=np.float32(0.0),
        cfg_weights=np.array([2.0, 2.0], np.float32),
        steps=np.int64(STEPS),
        timesteps=run["t"].astype(np.int64),
        observed_motion=np.where(run["mask"], run["observed"], 0.0).astype(np.float32),
        constraint_mask=run["mask"],
        root_frames=ROOT_FRAMES.astype(np.int64),
        root_xz=root_xz[ROOT_FRAMES].numpy().astype(np.float32),
        full_body_frame=np.int64(FULL_BODY_FRAME),
        full_body_positions=positions[FULL_BODY_FRAME].numpy().astype(np.float32),
        full_body_rotations=rotations[FULL_BODY_FRAME].numpy().astype(np.float32),
        hands_frame=np.int64(HANDS_FRAME),
        hands_effectors=np.array(HANDS),
        hands_positions=positions[HANDS_FRAME].numpy().astype(np.float32),
        hands_rotations=rotations[HANDS_FRAME].numpy().astype(np.float32),
        trace_predicted_clean=run["pred"].astype(np.float16),
        check_steps=np.array(CHECK_STEPS, np.int64),
        check_motion_in=run["x_in"][rows].astype(np.float32),
        check_branches=run["branches"][rows].astype(np.float16),
        check_motion_out=run["x_out"][rows].astype(np.float32),
        final_features=unnormalised.astype(np.float32),
        decoded_posed_joints=decoded["posed_joints"].astype(np.float32),
        decoded_global_rotations=decoded["global_rot_mats"].astype(np.float32),
        decoded_root_positions=decoded["root_positions"].astype(np.float32),
        decoded_smooth_root=decoded["smooth_root_pos"].astype(np.float32),
        decoded_heading=decoded["global_root_heading"].astype(np.float32),
        decoded_foot_contacts=decoded["foot_contacts"].astype(bool),
    )
    print("fixture: mask bits", int(run["mask"].sum()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", choices=("tokens", "embeddings", "fixture", "all"))
    parser.add_argument(
        "--checkpoints", type=Path, help="folder holding Kimodo-SOMA-RP-v1.1"
    )
    parser.add_argument("--encoder", type=Path, help="LLM2Vec ONNX folder")
    args = parser.parse_args()
    FIXTURES.mkdir(parents=True, exist_ok=True)
    if args.what in ("tokens", "all"):
        record_tokens(args.encoder)
    if args.what in ("embeddings", "all"):
        record_embeddings(args.encoder)
    if args.what in ("fixture", "all"):
        record_fixture(args.checkpoints)


if __name__ == "__main__":
    main()
