# Releases: the `latest` branch

Hosts consume this flake as `github:charlie-horse/vllm-sm70/latest`, never `main`. The host flake re-locks its inputs daily, so whatever `latest` points at reaches c4130 on its next switch.

## Moving `latest`

`latest` only fast-forwards, to a commit that is already on `main`. Penny (release) moves it, and only once all four of these hold for **that exact commit**:

1. `main` CI is green on it (patch series, both Python unit jobs, Nix eval).
2. C1 passes on it: the side-by-side bench and quality run against the fork on c4130 (see `docs/roadmap.md`, Phase 5).
3. Bea's GPU gate passed on it, in a downtime window (00:00–06:00 ET).
4. Astro says OK.

A gate run on another commit doesn't count, even if only docs changed in between. Never force-push `latest` or move it backwards. To roll back, revert on `main`, run the gates again, and fast-forward.

To move it:

```
git fetch origin main latest
git merge-base --is-ancestor origin/latest <commit>   # must succeed: fast-forward only
git push origin <commit>:refs/heads/latest
```

Record each move in the table below, in the same PR or straight after it.

## History

| date (UTC) | commit | CI | C1 | GPU gate | OK |
|---|---|---|---|---|---|
| 2026-11-05 | `3869cd6` | green | – | – | Astro (bootstrap only) |

The first entry is a **bootstrap**. It created the branch so the host flake has something to lock, and it has not passed C1 or the GPU gate. Don't deploy it: the cutover (C3) waits until `latest` has been fast-forwarded to a commit that passed all four gates.
