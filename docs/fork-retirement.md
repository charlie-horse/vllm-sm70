# Fork retirement: charlie-horse/vLLM as reference only

**Status: draft.** Archiving the repository is Charlie's call. Nothing here runs until C3 (cutover) is done and Charlie says so. Owner of C4 (roadmap): Charlie.

## Preconditions

- C3 done: `latest` of `charlie-horse/vllm-sm70` is the host's live build, and the host flake imports `nixosModules.default` (see `docs/cutover-runbook.md`).
- The system has run stably for a period agreed with Astro (see `docs/cutover-runbook.md` section 6).
- The rollback to the fork (`docs/cutover-runbook.md` section 5) is no longer needed, and Charlie has said so.

## Steps after C3

1. **README pointer.** On `charlie-horse/vLLM`, put a short notice at the top of `README.md`: the fork is kept as a reference only; production builds from `charlie-horse/vllm-sm70` (link to its `latest` branch and `docs/release.md`); no further changes land here. Commit to the default branch. Charlie merges or pushes.
2. **Pin the last fork commit.** Record the fork's final commit SHA in this doc (section "What stays"), not in `docs/release.md` History, which only logs `latest` moves. Fill it in after C3.
3. **Stop automation that points at the fork.** Check the host flake, CI and any scripts for references to the fork's flake input or the fork's `vllm-lmcache` unit. Remove them. (The host flake change itself is in `docs/cutover-runbook.md` section 2.)
4. **Keep, as reference:** the fork's source tree and history (read-only use), and the fork's operator docs that `docs/roadmap.md` cites for switch names. Tools that read a fork checkout at cf9492a, all of which must keep working against a read-only fork:
   - `sm70/tools/check_turbomind_provenance.py --fork` (PR #78);
   - `sm70/tools/check_flash_v100_provenance.py`;
   - `sm70/tools/check_qpn8_provenance.py`;
   - `scripts/verify-port.py --fork`, which every port review uses.
   Before archiving, run each once against the fork tree and record the result here.

   Results, 2026-11-16 (fork checkout at cf9492a, this repo at 2d18ea8, `python3 -I`), all exit 0:

   | tool | result |
   |---|---|
   | `check_turbomind_provenance.py --fork` | 124 rows (107 verbatim), fork checked: OK |
   | `check_flash_v100_provenance.py --fork` | 11 files, 92 definitions (82 verbatim), fork checked: OK |
   | `check_qpn8_provenance.py --fork` | 5 rows (3 verbatim), fork checked: OK |
   | `check_flint_moe_provenance.py --fork` | 3 files, 43 definitions (36 verbatim), fork checked: OK |
   | `verify-port.py` (non-strict, pairs from each PROVENANCE.md) | dflash2 .cu IDENTICAL=9 CHANGED=2 NEW=7, .py CHANGED=4 NEW=33 DROPPED=5; flint_moe .cu IDENTICAL=34 CHANGED=1 MOVED=2 NEW=6 DROPPED=11, .py IDENTICAL=4 NEW=28 DROPPED=89 DROPPED(expected)=10; gdn IDENTICAL=20 CHANGED=3 NEW=10 DROPPED=23; qpn8 IDENTICAL=21 CHANGED=8 NEW=2 DROPPED=41 |
5. **Archive, after Charlie's yes:** set the repository to archived on GitHub (Settings → Archive). Archived repos stay readable and forkable; no pushes, issues or PRs. This is hard to undo cleanly (unarchiving is possible but needs Charlie).

## What stays

- Final fork commit: `cf9492a01eeca0cac731c0fa62015e2c85aaf272` (`main` and `latest` of the fork). The fork has no tags.
- The fork's history, tags and branches (nothing is deleted), if it is archived. See "Deletion instead of archiving".
- The README pointer, and this doc's reference to the final SHA.
- Fork-only content copied into this repo (each directory has a PROVENANCE.md naming the fork commit or branch):
  - `docs/benchmarks/flint_swebench_verified.md`, `scripts/bench/flint_swebench/`, `scripts/bench/flint_coherence/` (branch `claude/flint-swebench`, bd0123f);
  - `docs/history/sm70_flint_max_num_seqs_postmortem.md` (branch `claude/v100-playbook`, 1863a2f), `docs/history/v100_model_playbook.md`, `docs/history/sm70_flint_tp2_pcie.md` (cf9492a);
  - `sm70/tools/flint/`: the Flint checkpoint converter (`convert_gguf.py`, `gguf_nvfp4.py`) and `tools/flint/*` (cf9492a).

## Deletion instead of archiving

Charlie is considering deleting the fork rather than archiving it. Compared with archiving, deletion loses:

- `git clone`/fetch of the fork (and any flake input or `fetchFromGitHub` on it) stops working. The host flake must not lock the fork any more (2d18ea8 removes the `vllm-legacy` input) before the delete.
- The provenance tools above (`--fork`) and `verify-port.py` need a fork checkout at cf9492a; they cannot be re-run unless someone keeps an offline copy (e.g. `git bundle create fork.bundle --all`, ~393 MB) and records where it is.
- The fork commits that docs here cite by hash (PROVENANCE files, postmortem, roadmap) become unresolvable without that copy.
- Branch `claude/flint-gpu-opt` (681a520, a logging-only Flash-V100 fix superseded here) is not copied.

## What does not stay

- No new commits, issues, PRs or releases on the fork after archiving.
- No deployment pointing at the fork.

## Open items for Charlie

- Confirm the archive step (step 5). It is outward-facing and not reversible without Charlie.
- Confirm the period before archiving (section 6 of the cutover runbook).
