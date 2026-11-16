#!/usr/bin/env python3
"""Repo-stratified slice of an id list (proportional, >= 1 per repo, seeded).

Usage: make_slice.py IDS_IN N SEED IDS_OUT
Quotas: Hamilton (largest remainder) on N * count / total, with every repo
getting at least 1. Within a repo, ids are drawn with random.Random(SEED);
the output keeps the input file order.
"""

import math
import random
import sys
from collections import defaultdict


def repo_of(iid: str) -> str:
    owner, rest = iid.split("__", 1)
    return f"{owner}/{rest.rsplit('-', 1)[0]}"


def quotas(counts: dict[str, int], n: int) -> dict[str, int]:
    total = sum(counts.values())
    exact = {r: n * c / total for r, c in counts.items()}
    q = {r: max(1, math.floor(x)) for r, x in exact.items()}
    while sum(q.values()) < n:
        r = max(exact, key=lambda r: (exact[r] - q[r], r))
        q[r] += 1
    while sum(q.values()) > n:
        r = min((r for r in q if q[r] > 1), key=lambda r: (exact[r] - q[r], r))
        q[r] -= 1
    return q


def main() -> None:
    src, n, seed, dst = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    ids = [x.strip() for x in open(src) if x.strip()]
    by_repo = defaultdict(list)
    for iid in ids:
        by_repo[repo_of(iid)].append(iid)
    q = quotas({r: len(v) for r, v in by_repo.items()}, n)
    rng = random.Random(seed)
    picked = set()
    for r in sorted(by_repo):
        picked |= set(rng.sample(by_repo[r], q[r]))
    out = [iid for iid in ids if iid in picked]
    with open(dst, "w") as f:
        f.write("\n".join(out) + "\n")
    for r in sorted(q, key=lambda r: -q[r]):
        print(f"{r}: {q[r]}/{len(by_repo[r])}")
    print(f"total {len(out)}")


if __name__ == "__main__":
    main()
