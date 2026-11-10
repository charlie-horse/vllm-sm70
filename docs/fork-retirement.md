# Fork retirement: charlie-horse/vLLM as reference only

**Status: draft.** Archiving the repository is Charlie's call. Nothing here runs until C3 (cutover) is done and Charlie says so. Owner of C4 (roadmap): Charlie.

## Preconditions

- C3 done: `latest` of `charlie-horse/vllm-sm70` is the host's live build, and the host flake imports `nixosModules.default` (see `docs/cutover-runbook.md`).
- The system has run stably for a period agreed with Astro (see `docs/cutover-runbook.md` section 6).
- The rollback to the fork (`docs/cutover-runbook.md` section 5) is no longer needed, and Charlie has said so.

## Steps after C3

1. **README pointer.** On `charlie-horse/vLLM`, put a short notice at the top of `README.md`: the fork is kept as a reference only; production builds from `charlie-horse/vllm-sm70` (link to its `latest` branch and `docs/release.md`); no further changes land here. Commit to the default branch. Charlie merges or pushes.
2. **Pin the last fork commit.** Record the fork's final commit SHA in `docs/release.md` History, so the reference point is recoverable.
3. **Stop automation that points at the fork.** Check the host flake, CI and any scripts for references to the fork's flake input or the fork's `vllm-lmcache` unit. Remove them. (The host flake change itself is in `docs/cutover-runbook.md` section 2.)
4. **Keep, as reference:** the fork's source tree and history (read-only use), the TurboMind provenance check inputs (`check_turbomind_provenance.py --fork`, PR #78), and the fork's operator docs that `docs/roadmap.md` cites for switch names.
5. **Archive, after Charlie's yes:** set the repository to archived on GitHub (Settings → Archive). Archived repos stay readable and forkable; no pushes, issues or PRs. This is hard to undo cleanly (unarchiving is possible but needs Charlie).

## What stays

- The fork's history, tags and branches (nothing is deleted).
- The README pointer, and this doc's reference to the final SHA.

## What does not stay

- No new commits, issues, PRs or releases on the fork after archiving.
- No deployment pointing at the fork.

## Open items for Charlie

- Confirm the archive step (step 5). It is outward-facing and not reversible without Charlie.
- Confirm the period before archiving (section 6 of the cutover runbook).
