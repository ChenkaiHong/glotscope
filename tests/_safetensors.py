"""Hand-built safetensors fixtures, shared by the tests that need weights.

Written by hand rather than through ``safetensors.numpy.save_file`` because the
format's own header is part of what is under test, and because two of §7.9's
three reference checkpoints store BF16 — a dtype numpy does not have, so
``safetensors.numpy`` cannot round-trip it and a fixture built through that API
could not reach the code path the real checkpoints take.

Underscore-prefixed so pytest does not collect it as a test module.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

Tensor = tuple[str, list[int], bytes]
"""``(safetensors dtype tag, shape, raw bytes)``."""


def write_safetensors(path: Path, tensors: dict[str, Tensor]) -> Path:
    """Write a minimal safetensors file: u64 header length, JSON header, buffer."""
    header: dict[str, Any] = {}
    buffer = bytearray()
    for name, (dtype, shape, payload) in tensors.items():
        start = len(buffer)
        buffer.extend(payload)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [start, len(buffer)]}
    encoded = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(buffer))
    return path


def f32(values: Any) -> Tensor:
    array = np.ascontiguousarray(values, dtype=np.float32)
    return "F32", list(array.shape), array.tobytes()


def bf16(values: Any) -> Tensor:
    """Truncate float32 to bfloat16 — the top 16 bits of each word."""
    array = np.ascontiguousarray(values, dtype=np.float32)
    truncated = (array.view(np.uint32) >> 16).astype(np.uint16)
    return "BF16", list(array.shape), truncated.tobytes()
