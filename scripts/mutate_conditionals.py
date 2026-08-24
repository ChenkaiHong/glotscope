"""Force each side of every conditional expression and report what no test notices.

``coverage.py`` does not treat ``X if C else Y`` as a branch, so a conditional
expression whose two sides are never both exercised still reports as covered.
This project uses them heavily in load-bearing places — they are how optional
values are threaded through frozen dataclasses — so the 85% gate is blind to a
whole class of untested behaviour.

The check is mutation rather than coverage: rewrite each conditional expression
down to its true side, run the suite, then down to its false side and run again.
If forcing one side breaks nothing, no test distinguishes the other side, whatever
coverage reports. The source file is restored after every mutant, including on
failure.

Slow by construction — two full suite runs per conditional expression, a few
minutes for this package — so it is a deliberate sweep rather than a CI job.

    uv run --no-sync python scripts/mutate_conditionals.py

Surviving mutants are not automatically defects. Many are defensive
``x if x is not None else default`` branches that no legitimate input reaches.
The output is a list of places to think about, not a list of bugs.
"""

from __future__ import annotations

import ast
import pathlib
import subprocess
import sys

ROOT = pathlib.Path("python/glotscope")
results: list[tuple[str, int, str, bool]] = []


def splice(src: str, node: ast.expr, repl_node: ast.expr) -> str:
    lines = src.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))
    assert node.end_lineno is not None and node.end_col_offset is not None
    assert repl_node.end_lineno is not None and repl_node.end_col_offset is not None
    a = starts[node.lineno - 1] + node.col_offset
    b = starts[node.end_lineno - 1] + node.end_col_offset
    ra = starts[repl_node.lineno - 1] + repl_node.col_offset
    rb = starts[repl_node.end_lineno - 1] + repl_node.end_col_offset
    return src[:a] + src[ra:rb] + src[b:]


def run_tests() -> bool:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-x", "-q", "--no-header", "-p", "no:cacheprovider"],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


for path in sorted(ROOT.rglob("*.py")):
    original = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(original)
    except SyntaxError:
        continue
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.IfExp)]
    for node in nodes:
        for label, repl in (("true-side", node.body), ("false-side", node.orelse)):
            try:
                mutant = splice(original, node, repl)
                path.write_text(mutant, encoding="utf-8")
                survived = run_tests()
            finally:
                path.write_text(original, encoding="utf-8")
            results.append((str(path), node.lineno, label, survived))
            print(
                f"{'SURVIVED' if survived else 'killed  '} {path}:{node.lineno} forced {label}",
                flush=True,
            )

print("\n=== SURVIVORS (no test distinguishes the other side) ===")
for p, line, label, survived in results:
    if survived:
        print(f"{p}:{line}  forcing {label} broke nothing")
print(f"\ntotal mutants {len(results)}, survivors {sum(1 for r in results if r[3])}")
