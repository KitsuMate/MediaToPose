"""Prompt text for Kimodo: upstream sanitising and the LLM2Vec encoder inputs.

``sanitize`` follows ``kimodo/sanitize.py`` exactly, including its
``str.capitalize()``, which lowercases everything after the first letter: the
model saw prompts in that form during training.

The encoder input follows LLM2Vec as Kimodo uses it: an empty instruction,
the ``!@#$%^&*()`` separator, the Llama 3 user turn template, and mean
pooling over the tokens of the prompt and the closing ``<|eot_id|>``. The
tokenizer is passed in, so this module needs no tokenizer package.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ...errors import InvalidRequest

MAX_TOKENS = 64
PAD_TOKEN_ID = 128001  # <|end_of_text|>
SEPARATOR = "!@#$%^&*()"
END_OF_TURN = "<|eot_id|>"
TEMPLATE = "<|start_header_id|>user<|end_header_id|>\n\n{text}" + END_OF_TURN
# Changes whenever the encoder input format changes; part of cache keys.
TEMPLATE_VERSION = 1
_FINAL_PUNCTUATION = ".!?\"])'"

# tokenize(text, add_special_tokens) -> token ids
Tokenize = Callable[[str, bool], list[int]]


def sanitize(text: str) -> str:
    """Upstream ``sanitize_text(text, paragraph=True)``."""
    text = " ".join(text.strip().split())
    if not text:
        return text
    start = next((i for i, char in enumerate(text) if char.isalnum()), len(text) - 1)
    text = text[start:].capitalize()
    end = next(
        (
            i
            for i in range(len(text) - 1, -1, -1)
            if text[i].isalnum() or text[i] in _FINAL_PUNCTUATION
        ),
        0,
    )
    text = text[: end + 1]
    if text[-1] not in ".!?":
        text += "."
    for sentence_break in ".!?":
        parts = [part.strip() for part in text.split(sentence_break)]
        text = f"{sentence_break} ".join(
            part[0].capitalize() + part[1:] if part else part for part in parts
        ).strip()
    return text


@dataclass(frozen=True)
class EncoderInputs:
    """One prompt as the [1, 64] int64 arrays the encoder graph takes."""

    input_ids: np.ndarray
    attention_mask: np.ndarray
    pooling_mask: np.ndarray
    text: str
    truncated: bool


def _layout(text: str, tokenize: Tokenize) -> tuple[list[int], int]:
    ids = tokenize(TEMPLATE.format(text=SEPARATOR + text), True)
    pooled = len(tokenize(text + END_OF_TURN, False))
    return ids, pooled


def encoder_inputs(prompt: str, tokenize: Tokenize) -> EncoderInputs:
    """Sanitise, template, tokenise and left-pad a prompt to 64 tokens.

    Words are dropped from the end of the prompt until it fits; the result
    then says ``truncated``.
    """
    text = sanitize(prompt)
    if not text:
        raise InvalidRequest("An empty prompt is not encoded; generate without text")
    words = text.split(" ")
    truncated = False
    ids, pooled = _layout(text, tokenize)
    while len(ids) > MAX_TOKENS:
        if len(words) <= 1:
            raise InvalidRequest("The prompt does not fit the text encoder")
        words = words[:-1]
        text = sanitize(" ".join(words))
        truncated = True
        ids, pooled = _layout(text, tokenize)
    if not 0 < pooled < len(ids):
        raise InvalidRequest("The tokenizer returned an unexpected prompt layout")
    padding = MAX_TOKENS - len(ids)
    input_ids = np.full((1, MAX_TOKENS), PAD_TOKEN_ID, dtype=np.int64)
    input_ids[0, padding:] = ids
    attention = np.zeros((1, MAX_TOKENS), dtype=np.int64)
    attention[0, padding:] = 1
    pooling = np.zeros((1, MAX_TOKENS), dtype=np.int64)
    pooling[0, MAX_TOKENS - pooled :] = 1
    return EncoderInputs(input_ids, attention, pooling, text, truncated)


__all__ = [
    "END_OF_TURN",
    "MAX_TOKENS",
    "PAD_TOKEN_ID",
    "SEPARATOR",
    "TEMPLATE",
    "TEMPLATE_VERSION",
    "EncoderInputs",
    "Tokenize",
    "encoder_inputs",
    "sanitize",
]
