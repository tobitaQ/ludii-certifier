# A certifier for board-game re-implementations

This is the measurement side of *Faithful Does Not Imply Usable*, released so that the method can be
applied to any re-implementation of a Ludii game, and so that the paper's sensitivity measurement can be
reproduced. It is code only: no certification records and no translations are in this deposit.

    certifier/   the fidelity ladder, the game-property certificate, the run-comparison rule, the
                 mutation harness, and everything they import
    games/       the five hand-written games the certifier was built against
    agents/      the random and greedy players the game-property certificate uses

## What it needs

A JVM, the packages in certifier/requirements.txt, and the Ludii jar, which is not redistributable:
certifier/LUDII_JAR.md gives the exact build and its sha256. Nothing here calls a language model, so no
account with any provider is required.

    python -m venv .venv && .venv/bin/pip install -r certifier/requirements.txt
    export PYTHONPATH="$PWD/certifier:$PWD"

## Certifying your own package

`ludii_oracle.certify(module, lud_name, episodes=K)` takes any module implementing the `BoardGame`
interface in certifier/base.py and replays K games in both engines. Levels are comparable only at equal
K and equal budgets, which is what certifier/ladder_diff.py enforces; the paper's numbers are K=30.

## What is not here

The repair side -- the operators, their adoption gates, and the rule agent's loop, prompts and retrieval
-- is not included, and neither is the corpus of translations. `bgame.repair()` therefore raises on
import of its operator module; everything else runs. The paper says why, and the certification records
for all 305 games are available to reviewers as supplementary material.
