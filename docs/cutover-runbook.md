# Cutover runbook (C3): draft

**Status: draft. Not for use yet.** C3 is blocked on C1 (side-by-side bench and quality run vs the fork, not run) and on the four gates in `docs/release.md` for the commit that `latest` will point at. Nothing in this file has been run on c4130.

Steps marked **[SUDO: Charlie]** need root on c4130 and are for Charlie to run himself. This doc does not carry or ask for credentials.

Host names below are as used elsewhere in the repo. Keep IPs, tailnet names and passwords out of this file.

## 0. Preconditions (all must hold before step 1)

- [ ] C1 passed on the target commit (`docs/roadmap.md`, Phase 5). Not run yet.
- [ ] C2 passed on c4130: LMCache check (`scripts/check-lmcache-cpu.sh`, plus the GPU smoke and `kvtier_test.py`, both still open in `docs/status.md`).
- [ ] Bea's GPU gate passed on the same commit, in a downtime window (00:00–06:00 ET, per `docs/release.md`).
- [ ] `main` CI is green on that commit.
- [ ] Astro says OK.
- [ ] `latest` is fast-forwarded to that commit (`docs/release.md`, "Moving `latest`"). The current `latest` is the bootstrap `3869cd6`, which has not passed the gates and must not be deployed.
- [ ] The rollback target (the fork's current flake input and LMCache unit) is recorded, see section 5.

## 1. GC c4130 (before the switch)

Old generations on c4130 have to go first, so the new closure has room and the rollback generation is kept.

1. Record the current generation and the fork's store paths for rollback:
   ```sh
   # on c4130, as charlie
   readlink -f /run/current-system
   nix-store --query --requisites /run/current-system | wc -l
   ```
   Save the output with the cutover PR.
2. **[SUDO: Charlie]** Keep the current generation as a GC root, then collect the rest:
   ```sh
   sudo nix-env --profile /nix/var/nix/profiles/system --list-generations
   # keep the generation from step 1 and the one before it; delete the older ones
   sudo nix-env --profile /nix/var/nix/profiles/system --delete-generations <old-numbers>
   sudo nix-collect-garbage
   ```
   Check free space afterwards (`df -h /nix`). The closure needed for the new build is about 10 GiB (`docs/phase1-gpu-runbook.md`).

## 2. Point the host flake at vllm-sm70

The host flake (charlie-horse/flake, `hosts/c4130/vllm`) consumes this repo. Change its input from the fork to this flake's `latest` branch, not `main`.

1. In the host flake, change the vllm input to `github:charlie-horse/vllm-sm70/latest`. Lock only that input:
   ```sh
   nix flake lock --update-input <vllm-input-name>
   ```
2. Build the closure on the build host first (r440), then realise it on c4130 with `nix-store --realise <outPath>` (same method as `docs/phase1-gpu-runbook.md` section 1). Do not switch yet.
3. Confirm the outPath matches the commit on `latest`.

## 3. Move vllm-lmcache.service to the new LMCache

The prod LMCache server runs an older torch, which cannot register the new build's KV (Bea's finding). The unit has to move to the new flake's LMCache.

1. Read the current unit and note its environment and ports, so the rollback can restore it exactly:
   ```sh
   systemctl cat vllm-lmcache.service
   ```
2. In the host flake, point the LMCache service at the new flake's LMCache package (same input as step 2). Confirm the new unit's torch matches the vLLM build's torch.
3. **[SUDO: Charlie]** Switch:
   ```sh
   sudo nixos-rebuild switch --flake <host-flake>#c4130
   ```
4. **[SUDO: Charlie]** Restart the LMCache unit and check it:
   ```sh
   sudo systemctl restart vllm-lmcache.service
   systemctl status vllm-lmcache.service --no-pager
   ```

## 4. Verify

Run in order; stop at the first failure and go to section 5.

1. `systemctl is-active vllm-lmcache.service` returns `active`.
2. The vLLM service starts and serves: one completion request against the served model (Qwen or Flint, whichever prod uses), returns a non-empty answer.
3. LMCache store and retrieve: the same prompt twice, with the second send showing `prompt_tokens_by_source` for an external source (not `local_cache_hit`). This is the 1d check (`docs/status.md`, Flint step 1).
4. etcd health OK (C3's exit test).
5. Greedy identity on the 5 prompts from the gate run matches the recorded output.

## 5. Rollback to the fork

Use this if any step in section 4 fails, or if prod misbehaves within the first hour after the switch.

1. **[SUDO: Charlie]** Roll the system back to the recorded generation:
   ```sh
   sudo nixos-rebuild switch --rollback
   ```
   Or switch to the generation from section 1 by number.
2. Restore the fork's flake input in the host flake (the commit recorded in section 1) and the original `vllm-lmcache.service` unit (the `systemctl cat` output saved in section 3). Rebuild with **[SUDO: Charlie]** `sudo nixos-rebuild switch --flake <host-flake>#c4130`.
3. **[SUDO: Charlie]** `sudo systemctl restart vllm-lmcache.service`, then the vLLM service.
4. Confirm the service serves a request with the fork's build.
5. Record the rollback in `docs/release.md` (History) and in the cutover PR. Do not move `latest` backward (`docs/release.md` says never move it backward; revert on `main` and re-gate instead).

## 6. After a successful cutover

- Record the cutover in the `docs/release.md` History table (commit, CI, C1, GPU gate, OK).
- C4: archive the fork (Charlie). Do not do this until the system has run stably for a period agreed with Astro.

## Open items for Charlie

- Confirm the host flake's vllm and LMCache input names and the `#c4130` attribute.
- Confirm the GC retention policy (how many generations to keep).
- Confirm the downtime window for the switch.
