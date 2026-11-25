# Releases: the `latest` branch

`main` is where changes land. `latest` is the branch to use: it points only
at commits that passed every test, including the GPU tests.

## Moving `latest`

`latest` only moves forward ("fast-forwards") to a commit already on
`main`, once all of these hold for **that exact commit**:

1. CI on `main` is green for it.
2. The [GPU gate](checks.md#gpu-gate) passes on it. A commit that changes
   Inkling 2 also passes the [Inkling 2 gate](checks.md#inkling-2-gate).
3. The repo owner approves the move.

A test run on any other commit doesn't count, even if only docs changed in
between. Never force-push `latest` or move it backwards. To undo a release,
revert the change on `main`, run the tests again, and move `latest` forward.

```
git fetch origin main latest
git merge-base --is-ancestor origin/latest <commit>   # must succeed: forward only
git push origin <commit>:refs/heads/latest
```
