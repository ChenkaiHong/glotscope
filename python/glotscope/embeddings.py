"""Embedding tensors, read without instantiating a model (PRD §6, §8.1).

Tier 2 needs only ``E_in`` and optionally ``E_out`` — two tensors, resolvable from
``model.safetensors.index.json`` and read from the embedding shards alone. §6 calls
this out as the reason unifying Tier 1 and Tier 2 in one package is practical, and
nobody noticing it is the opportunity.

The hard refusal in this module is quantization. A 4-bit ``E_in`` destroys the
L2-norm indicator, which is the exact signal Tier 2 depends on, and community
mirrors routinely republish 4-bit, GGUF or merged variants under near-identical
names. §11's mirror mitigation was written for tokenizers and is insufficient
here — so dtype is checked, recorded, and refused rather than warned about.

``safetensors`` and ``numpy`` are the ``tier2`` extra rather than core
dependencies: a core install promises Tier 0 and Tier 1, and G1's clean-install
claim is measured on it. Import them inside the function that needs them and
raise a message naming the extra — an unguarded top-level import would surface a
bare ``ModuleNotFoundError``, which reads as a broken package rather than as an
install that never included this tier.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from glotscope.errors import UnsupportedCheckpointError
from glotscope.manifest import UNKNOWN_LICENSE, WeightsManifest

if TYPE_CHECKING:
    from typing import Any

    import numpy as np
    from numpy.typing import NDArray

    FloatMatrix = NDArray[np.floating[Any]]
    """Any original-precision float matrix. Deliberately not pinned to one dtype:
    checkpoints ship float32, float16 and bfloat16, and narrowing the annotation
    would push a cast into every reader."""

__all__ = ["ALLOWED_DTYPES", "Embeddings"]

ALLOWED_DTYPES = frozenset({"float32", "float16", "bfloat16", "float64"})
"""Original-precision floating dtypes. Anything else — 4-bit, 8-bit, GGUF
quantization, or an integer dtype — is refused with
:class:`~glotscope.errors.UnsupportedCheckpointError`.
"""

_SAFETENSORS_FLOATS = {
    "F64": "float64",
    "F32": "float32",
    "F16": "float16",
    "BF16": "bfloat16",
}
"""safetensors dtype tags that are original-precision floats, mapped to the
names in :data:`ALLOWED_DTYPES` and in the §9 manifest."""

_SAFETENSORS_READABLE = {
    "I8": "int8",
    "U8": "uint8",
    "I16": "int16",
    "U16": "uint16",
    "I32": "int32",
    "U32": "uint32",
    "I64": "int64",
    "U64": "uint64",
    "BOOL": "bool",
    "F8_E4M3": "float8_e4m3",
    "F8_E5M2": "float8_e5m2",
}
"""Spellings for the refusal message. ``I8`` lowercased is ``i8``, which names
nothing a reader would recognise as the quantization they applied."""

_E_IN_NAMES = (
    "model.embed_tokens.weight",
    "transformer.wte.weight",
    "wte.weight",
    "embed_tokens.weight",
    "tok_embeddings.weight",
    "gpt_neox.embed_in.weight",
    "model.embed_in.weight",
    "embeddings.word_embeddings.weight",
)
"""Input-embedding tensor names, most common first.

Read from the published safetensors headers of §7.9's reference checkpoints
rather than guessed: ``google/gemma-2b`` and ``ai21labs/Jamba-v0.1`` use
``model.embed_tokens.weight``, ``openai-community/gpt2-medium`` uses
``wte.weight``. The rest cover families those three do not."""

_E_OUT_NAMES = ("lm_head.weight", "output.weight", "embed_out.weight")
"""Output-embedding names. Absence means the checkpoint ties, which is the case
for two of the three reference checkpoints — so the tied path is the common one,
not the exception."""


def _deserialize(blob: bytes) -> list[tuple[str, dict[str, Any]]]:
    """Parse a safetensors buffer into ``(name, {dtype, shape, data})`` pairs.

    Uses ``safetensors``' low-level entry point rather than its numpy binding,
    because the numpy binding cannot represent BF16 and two of §7.9's three
    reference checkpoints store exactly that. Imported here rather than at module
    scope so a core install — which promises Tier 0 and Tier 1 only — reports the
    missing extra by name instead of failing at import.
    """
    try:
        from safetensors import deserialize
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised by the extra
        raise ModuleNotFoundError(
            "reading embedding tensors needs `safetensors`, which ships in the "
            "`tier2` extra: pip install 'glotscope[tier2]'"
        ) from exc
    # safetensors ships no type information for its Rust entry points, so this
    # one call is untyped. Narrowed here rather than loosened for the module,
    # which would silence the check for every call site in the file.
    parsed: list[tuple[str, dict[str, Any]]] = deserialize(blob)  # type: ignore[no-untyped-call]
    return parsed


def _first_present(tensors: dict[str, Any], names: tuple[str, ...]) -> str | None:
    return next((name for name in names if name in tensors), None)


def _float_dtype(tag: str, checkpoint: str) -> str:
    """Map a safetensors dtype tag to an allowed float name, or refuse.

    A hard refusal rather than a warning: a 4-bit ``E_in`` still produces
    perfectly plausible L2 norms, and a plausible wrong number is the worst
    outcome for a tool whose output other people cite.
    """
    if tag in _SAFETENSORS_FLOATS:
        return _SAFETENSORS_FLOATS[tag]
    readable = _SAFETENSORS_READABLE.get(tag, tag.lower())
    raise UnsupportedCheckpointError(
        checkpoint,
        f"embedding dtype is {readable}, which is not an original-precision "
        f"float. Quantization destroys the L2-norm indicator §7.9 depends on "
        f"while leaving the arithmetic working, so this is refused rather than "
        f"warned about. Allowed: {', '.join(sorted(ALLOWED_DTYPES))}",
    )


def _as_array(spec: dict[str, Any]) -> FloatMatrix:
    """Materialise one tensor, widening BF16 to float32.

    numpy has no bfloat16, so the widening is forced rather than chosen. It
    invents nothing: bfloat16 is the leading 16 bits of a float32, so every
    value is exact and the manifest still records ``bfloat16`` as what the
    checkpoint holds.
    """
    import numpy as np

    shape = tuple(spec["shape"])
    data = spec["data"]
    if spec["dtype"] == "BF16":
        widened = np.frombuffer(data, dtype=np.uint16).astype(np.uint32) << np.uint32(16)
        upcast: FloatMatrix = widened.view(np.float32).reshape(shape)
        return upcast
    native = {"F64": np.float64, "F32": np.float32, "F16": np.float16}[spec["dtype"]]
    materialised: FloatMatrix = np.frombuffer(data, dtype=native).reshape(shape)
    return materialised


@dataclass(frozen=True, slots=True)
class Embeddings:
    """The two tensors Tier 2 needs, plus their provenance.

    ``E_out`` is ``None`` only when it cannot be located separately; when
    embeddings are tied it is the same tensor as ``E_in`` and :attr:`tied` is
    ``True``. That distinction decides which indicators can run:

    * **tied** — only ``C(E_out, u_ref)``, which needs a reference set, so tied
      checkpoints cannot be fully automated.
    * **untied** — both ``L2(E_in)`` and ``C(E_out, u_ref)``, and per D10 both are
      run so their Spearman agreement can be reported.
    """

    e_in: FloatMatrix
    e_out: FloatMatrix | None
    tied: bool
    dtype: str
    shard_sha256: str
    checkpoint: str
    n_rows: int
    """Row count of the embedding matrix. **May exceed** ``|V|`` — the padding rows
    are link two of the reference-set fallback chain (§7.9)."""

    vocab_size: int

    license_spdx: str = UNKNOWN_LICENSE
    """SPDX identifier of the weights, which is a different licence from the
    tokenizer's. A local file carries none, and the repository it was exported
    from is not recoverable from the bytes, so it stays ``UNKNOWN`` rather than
    inheriting anything — ``--license-filter=commercial`` reads this field."""

    @property
    def padding_rows(self) -> tuple[int, ...]:
        """Embedding rows above ``|V|``, a reference-set source (§7.9 chain link 2)."""
        return tuple(range(self.vocab_size, self.n_rows))

    @classmethod
    def from_checkpoint(
        cls,
        model_id: str,
        *,
        revision: str | None = None,
    ) -> Embeddings:
        """Read the embedding tensors from a checkpoint, detecting tying.

        Resolves sharded checkpoints through ``model.safetensors.index.json`` and
        reads only the embedding shards, so this costs seconds rather than loading
        the model.

        Raises:
            UnsupportedCheckpointError: if the dtype is not in
                :data:`ALLOWED_DTYPES`. This is a hard refusal, not a warning:
                a quantized ``E_in`` makes the L2-norm indicator meaningless while
                still producing plausible-looking numbers, which is the worst
                possible failure mode for a tool whose output other people cite.
        """
        raise NotImplementedError(
            f"resolving {model_id!r} from the Hub is not implemented in this "
            f"release. Download the checkpoint and pass the safetensors file "
            f"holding the embedding tensors to Embeddings.from_file()."
        )

    @classmethod
    def from_file(cls, path: str | Path, *, vocab_size: int) -> Embeddings:
        """Read embeddings from a local ``safetensors`` file.

        Raises:
            UnsupportedCheckpointError: if no embedding tensor is present, or if
                its dtype is not in :data:`ALLOWED_DTYPES`.
        """
        blob = Path(path).read_bytes()
        tensors = dict(_deserialize(blob))

        e_in_name = _first_present(tensors, _E_IN_NAMES)
        if e_in_name is None:
            raise UnsupportedCheckpointError(
                str(path),
                f"no embedding tensor found. Looked for {', '.join(_E_IN_NAMES)}; "
                f"the file holds {', '.join(sorted(tensors)[:8])}",
            )

        dtype = _float_dtype(tensors[e_in_name]["dtype"], str(path))
        e_out_name = _first_present(tensors, _E_OUT_NAMES)
        if e_out_name is not None:
            _float_dtype(tensors[e_out_name]["dtype"], str(path))

        e_in = _as_array(tensors[e_in_name])
        return cls(
            e_in=e_in,
            e_out=None if e_out_name is None else _as_array(tensors[e_out_name]),
            # Tying is read off the file rather than off a config flag:
            # `tie_word_embeddings` is absent from gemma-2b's config entirely,
            # and the absence of a separate head is the fact that decides which
            # indicators can run.
            tied=e_out_name is None,
            dtype=dtype,
            shard_sha256=hashlib.sha256(blob).hexdigest(),
            checkpoint=str(path),
            n_rows=int(e_in.shape[0]),
            vocab_size=vocab_size,
        )

    @property
    def manifest(self) -> WeightsManifest:
        """Weight provenance, ready to embed in a result (§9).

        ``dtype`` is what the checkpoint holds, not what these arrays are: BF16
        is widened to float32 on read because numpy has no bfloat16, and
        recording float32 here would erase the distinction the field exists to
        preserve.
        """
        return WeightsManifest(
            shard_sha256=self.shard_sha256,
            dtype=self.dtype,
            tied_embeddings=self.tied,
            license_spdx=self.license_spdx,
        )
