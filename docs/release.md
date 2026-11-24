# Releases: the `latest` branch

Hosts consume this flake as `github:charlie-horse/vllm-sm70/latest`, never
`main`. The host flake re-locks its inputs, so whatever `latest` points at
reaches the serving host on its next switch.

## Moving `latest`

`latest` only fast-forwards, to a commit already on `main`, once all of these
hold for **that exact commit**:

1. `main` CI is green on it.
2. The [GPU gate](checks.md#gpu-gate) passes on it.
3. The repo owner approves the move.

A gate run on another commit doesn't count, even if only docs changed in
between. Never force-push `latest` or move it backwards. To roll back, revert
on `main`, run the gates again, and fast-forward.

```
git fetch origin main latest
git merge-base --is-ancestor origin/latest <commit>   # must succeed: fast-forward only
git push origin <commit>:refs/heads/latest
```
