"""The published leaderboard (PRD §16.1, G6).

Split across three modules because they fail differently: a configuration is
wrong before anything runs, a row can be skipped while the board still publishes,
and rendering must never invent a number it was not given.
"""

from __future__ import annotations

from glotscope.leaderboard.check import ALL_TIERS, check_board
from glotscope.leaderboard.config import (
    ConfigError,
    CorpusPlan,
    LeaderboardConfig,
    ParameterPlan,
    RosterEntry,
    load_config,
)
from glotscope.leaderboard.render import render_markdown
from glotscope.leaderboard.run import (
    TOKENIZER_ONLY,
    LeaderboardDocument,
    LeaderboardRow,
    run_leaderboard,
)

__all__ = [
    "ALL_TIERS",
    "TOKENIZER_ONLY",
    "ConfigError",
    "CorpusPlan",
    "LeaderboardConfig",
    "LeaderboardDocument",
    "LeaderboardRow",
    "ParameterPlan",
    "RosterEntry",
    "check_board",
    "load_config",
    "render_markdown",
    "run_leaderboard",
]
