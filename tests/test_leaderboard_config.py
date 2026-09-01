"""The leaderboard configuration (PRD §16.1, §11).

A leaderboard is a published artifact, so its configuration is the place to be
strict: every refusal here is something that would otherwise reach a table other
people cite. The three that matter are an unpinned Hub row, an unlabelled mirror,
and a key nobody reads.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from glotscope.enums import Normalization, Segmenter
from glotscope.leaderboard import ConfigError, load_config

_MINIMAL = """
version: 1
corpus:
  id: flores_plus
  languages: [eng_Latn, hin_Deva]
parameters:
  parity_reference: eng_Latn
roster:
  - id: "tiktoken:o200k_base"
"""


def _write(tmp_path: Path, text: str, name: str = "leaderboard.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_a_minimal_config_takes_its_defaults_from_the_registry(tmp_path: Path) -> None:
    config = load_config(_write(tmp_path, _MINIMAL))

    assert config.corpus.id == "flores_plus"
    # The registry knows FLORES+ is pinned at 2024.08 and read from devtest;
    # repeating that in every config is a second place for it to drift.
    assert config.corpus.version == "2024.08"
    assert config.corpus.split == "devtest"
    assert config.parameters.normalization is Normalization.NFC
    assert config.parameters.segmenter is None
    assert [entry.id for entry in config.roster] == ["tiktoken:o200k_base"]


def test_json_is_accepted_so_the_core_install_can_read_a_config(tmp_path: Path) -> None:
    """YAML is what §8.2 spells, and it needs a parser that is not in the core
    install. JSON needs nothing, so a core install is not locked out of its own
    leaderboard."""
    document = {
        "version": 1,
        "corpus": {"id": "flores_plus", "languages": ["eng_Latn"]},
        "parameters": {},
        "roster": [{"id": "tiktoken:o200k_base"}],
    }
    path = _write(tmp_path, json.dumps(document), name="leaderboard.json")

    assert load_config(path).roster[0].id == "tiktoken:o200k_base"


def test_an_unpinned_hub_row_is_refused(tmp_path: Path) -> None:
    """§11 pins every row by commit revision. An unpinned run still produces a
    *pinned* manifest — the branch is resolved and recorded — but it pins a
    commit nobody chose, and §16.1's nightly job would then report a number
    moving as a regression when the upstream branch simply advanced."""
    text = _MINIMAL.replace('  - id: "tiktoken:o200k_base"', "  - id: Qwen/Qwen3-8B")

    with pytest.raises(ConfigError) as caught:
        load_config(_write(tmp_path, text))

    message = str(caught.value)
    assert "Qwen/Qwen3-8B" in message
    assert "revision" in message


def test_a_local_path_and_an_encoding_take_no_revision(tmp_path: Path) -> None:
    """Neither has a commit to pin, so demanding one would be asking for a
    string that cannot be true."""
    text = _MINIMAL.replace(
        '  - id: "tiktoken:o200k_base"',
        '  - id: "tiktoken:o200k_base"\n  - id: ./local/tokenizer.json',
    )

    config = load_config(_write(tmp_path, text))

    assert [entry.revision for entry in config.roster] == [None, None]


def test_an_unlabelled_mirror_is_refused(tmp_path: Path) -> None:
    """§11: mirror-sourced rows must be visibly labelled, because a leaderboard
    silently using unofficial re-uploads is a line of attack. A row flagged as a
    mirror with nothing to show a reader is that failure with a flag set."""
    text = _MINIMAL.replace(
        '  - id: "tiktoken:o200k_base"',
        "  - id: unsloth/Meta-Llama-3.1-8B-Instruct\n"
        "    revision: " + "c" * 40 + "\n"
        "    is_mirror: true",
    )

    with pytest.raises(ConfigError) as caught:
        load_config(_write(tmp_path, text))

    assert "note" in str(caught.value)


def test_a_labelled_mirror_is_accepted(tmp_path: Path) -> None:
    text = _MINIMAL.replace(
        '  - id: "tiktoken:o200k_base"',
        "  - id: unsloth/Meta-Llama-3.1-8B-Instruct\n"
        "    revision: " + "c" * 40 + "\n"
        "    is_mirror: true\n"
        "    note: ungated re-upload of a manually gated repository",
    )

    entry = load_config(_write(tmp_path, text)).roster[0]

    assert entry.is_mirror is True
    assert entry.note


@pytest.mark.parametrize(
    ("fragment", "expected"),
    [
        ("\nnonsense: 1\n", "nonsense"),
        ("\nroster:\n  - id: x\n    colour: red\n", "colour"),
    ],
)
def test_an_unreadable_key_is_refused_by_name(tmp_path: Path, fragment: str, expected: str) -> None:
    """Silently ignoring a key is how a leaderboard gets published under
    parameters its author believed were applied."""
    with pytest.raises(ConfigError) as caught:
        load_config(_write(tmp_path, _MINIMAL + fragment))

    assert expected in str(caught.value)


def test_an_empty_roster_is_refused(tmp_path: Path) -> None:
    text = _MINIMAL.replace('  - id: "tiktoken:o200k_base"', "")

    with pytest.raises(ConfigError) as caught:
        load_config(_write(tmp_path, text))

    assert "roster" in str(caught.value)


def test_parity_reference_must_be_in_the_language_set(tmp_path: Path) -> None:
    """Parity is measured against a language the run actually read. A reference
    outside the set produces a KeyError deep inside the fold, where it reads as a
    bug rather than as a configuration mistake."""
    text = _MINIMAL.replace("parity_reference: eng_Latn", "parity_reference: fra_Latn")

    with pytest.raises(ConfigError) as caught:
        load_config(_write(tmp_path, text))

    assert "fra_Latn" in str(caught.value)


def test_parameters_are_parsed_into_their_enums(tmp_path: Path) -> None:
    text = _MINIMAL.replace(
        "  parity_reference: eng_Latn",
        "  parity_reference: eng_Latn\n"
        "  segmenter: whitespace\n"
        "  normalization: NFKC\n"
        "  renyi_alpha: 2.5\n"
        "  gini: true",
    )

    parameters = load_config(_write(tmp_path, text)).parameters

    assert parameters.segmenter is Segmenter.WHITESPACE
    assert parameters.normalization is Normalization.NFKC
    assert parameters.renyi_alpha == 2.5
    assert parameters.gini is True


def test_an_unknown_segmenter_names_the_members(tmp_path: Path) -> None:
    text = _MINIMAL.replace(
        "  parity_reference: eng_Latn", "  parity_reference: eng_Latn\n  segmenter: guess"
    )

    with pytest.raises(ConfigError) as caught:
        load_config(_write(tmp_path, text))

    assert "guess" in str(caught.value)
    assert "whitespace" in str(caught.value)


def test_a_yaml_config_reads_the_same_as_its_json(tmp_path: Path) -> None:
    """§8.2 spells the file `leaderboard.yaml`; JSON exists so a core install is
    not locked out. Both must mean the same thing, or the extra changes results."""
    from_yaml = load_config(_write(tmp_path, _MINIMAL))
    from_json = load_config(
        _write(
            tmp_path,
            json.dumps(
                {
                    "version": 1,
                    "corpus": {"id": "flores_plus", "languages": ["eng_Latn", "hin_Deva"]},
                    "parameters": {"parity_reference": "eng_Latn"},
                    "roster": [{"id": "tiktoken:o200k_base"}],
                }
            ),
            name="same.json",
        )
    )

    assert from_yaml == from_json


@pytest.mark.parametrize(
    ("name", "text", "expected"),
    [
        ("broken.yaml", "corpus: [unclosed\n", "YAML"),
        ("broken.json", "{not json", "JSON"),
    ],
)
def test_a_malformed_file_names_the_format(
    tmp_path: Path, name: str, text: str, expected: str
) -> None:
    with pytest.raises(ConfigError) as caught:
        load_config(_write(tmp_path, text, name=name))

    assert expected in str(caught.value)


def test_a_missing_file_is_a_config_error(tmp_path: Path) -> None:
    """Not an OSError escaping to a traceback: the CLI maps ConfigError to exit
    1 with the message, and a bare traceback is not an answer about the input."""
    with pytest.raises(ConfigError) as caught:
        load_config(tmp_path / "absent.yaml")

    assert "absent.yaml" in str(caught.value)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("version: 1\ncorpus: 3\nroster:\n  - id: x\n", "mapping"),
        (
            "version: 1\ncorpus:\n  id: flores_plus\n  languages: []\nroster:\n  - id: x\n",
            "languages",
        ),
        (
            "version: 1\ncorpus:\n  id: fineweb2\n  languages: [eng_Latn]\nroster:\n  - id: x\n",
            "version",
        ),
        ("version: 1\ncorpus:\n  id: flores_plus\n  languages: [a]\nroster:\n  - 7\n", "mapping"),
        (
            "version: 1\ncorpus:\n  id: flores_plus\n  languages: [a]\nroster:\n  - id: 7\n",
            "non-empty string",
        ),
        (
            "version: 1\ncorpus:\n  id: flores_plus\n  languages: [a]\n"
            "roster:\n  - id: ./local.json\n    revision: " + "e" * 40 + "\n",
            "pin nothing",
        ),
    ],
)
def test_the_remaining_refusals_name_what_is_wrong(
    tmp_path: Path, text: str, expected: str
) -> None:
    """Each of these would otherwise reach a published table as a silently
    dropped or silently invented field."""
    with pytest.raises(ConfigError) as caught:
        load_config(_write(tmp_path, text))

    assert expected in str(caught.value)


def test_a_row_names_itself_when_it_carries_no_label(tmp_path: Path) -> None:
    """The label is what a reader sees; without one the identifier is the
    honest fallback, since a blank cell names nothing."""
    labelled = _MINIMAL.replace(
        '  - id: "tiktoken:o200k_base"',
        '  - id: "tiktoken:o200k_base"\n    label: OpenAI o200k',
    )

    assert load_config(_write(tmp_path, _MINIMAL)).roster[0].display == "tiktoken:o200k_base"
    assert load_config(_write(tmp_path, labelled)).roster[0].display == "OpenAI o200k"


def test_a_corpus_outside_the_registry_lists_the_known_ones(tmp_path: Path) -> None:
    """Gating is on corpus capabilities and never on identity (D5) — but the
    registry is what supplies those capabilities, so a corpus it does not know
    has none to gate on."""
    text = _MINIMAL.replace("id: flores_plus", "id: not_a_corpus")

    with pytest.raises(ConfigError) as caught:
        load_config(_write(tmp_path, text))

    assert "not_a_corpus" in str(caught.value)
    assert "flores_plus" in str(caught.value)
