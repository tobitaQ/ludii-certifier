# Ludii 1.3.14 — the reference engine, and the one file git does not hold

`.gitignore` excludes `ludii/*.jar`, so a fresh clone of this repository **cannot certify anything**:
`pipeline/ludii_oracle.py` starts a JVM against this jar on import, and without it even
`pipeline/paper_check.py` fails before its first check. Verified by cloning into a scratch directory on
2026-10-21: the clone had the 305 translations, the certification records and the paper, and could not run.

The build is pinned, not merely the version. Every level in `results/ladder_certification_K30_effect.json`
is a statement about **this file**. If ludii.games later ships a different 1.3.14 artefact, or stops
shipping it, the corpus becomes unreproducible, so verify the hash rather than assuming the download
matches.

    file    Ludii-1.3.14.jar
    bytes   21666905
    sha256  248a8bde801f347bc380a4957fdb48012b4bdf234a5591c9bf7479913d73068e

    sha256sum -c <<< "248a8bde801f347bc380a4957fdb48012b4bdf234a5591c9bf7479913d73068e  ludii/Ludii-1.3.14.jar"

Recovery: the jar is carried in the `core` backup archive (`pipeline/backup.py`), which exists precisely
to cover what git does not. Obtaining it again from ludii.games is the fallback, and the hash above is how
you would know you got the same engine.
