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

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

    import numpy as np
    from numpy.typing import NDArray

    from glotscope.manifest import WeightsManifest

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
        raise NotImplementedError

    @classmethod
    def from_file(cls, path: str | Path, *, vocab_size: int) -> Embeddings:
        """Read embeddings from a local ``safetensors`` file."""
        raise NotImplementedError

    @property
    def manifest(self) -> WeightsManifest:
        """Weight provenance, ready to embed in a result (§9)."""
        raise NotImplementedError
