# Cutover runbook (C3): draft

**Status: draft. Not for use yet.** C3 is blocked on C1 (side-by-side bench and quality run vs the fork, not run) and on the four gates in `docs/release.md` for the commit that `latest` will point at. Nothing in this file has been run on c4130.

Steps marked **[SUDO: Charlie]** need root on c4130 and are for Charlie to run. This doc does not carry or ask for credentials.

Host names below are as used elsewhere in the repo. Keep IPs, tailnet names and passwords out of this file.

## 0. Preconditions (all must hold before step 1)

- [ ] C1 passed on the target commit (`docs/roadmap.md`, Phase 5). Not run yet.
- [ ] C2 passed on c4130: LMCache check (`scripts/check-lmcache-cpu.sh`, plus the GPU smoke and `kvtier_test.py`, both still open in `docs/status.md`).
- [ ] Bea's GPU gate passed on the same commit, in a downtime window (00:00–06:00 ET, per `docs/release.md`).
- [ ] `main` CI is green on that commit.
- [ ] Astro says OK.
- [ ] `latest` is fast-forwarded to that commit (`docs/release.md`, "Moving `latest`"). The current `latest` is the bootstrap `3869cd6`, which has not passed the gates and must not be deployed.
- [ ] The rollback target (the fork's current flake input and `vllm-lmcache` unit) is recorded, see section 5.

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

1. In the host flake, change the vllm input to `github:charlie-horse/vllm-sm70/latest`. Update only that input:
   ```sh
   nix flake update <vllm-input-name>
   ```
2. Build the closure on the build host first (r440), then realise it on c4130 with `nix-store --realise <outPath>` (same method as `docs/phase1-gpu-runbook.md` section 1). Do not switch yet.
3. Confirm the outPath matches the commit on `latest`.

## 3. Replace the fork's vllm-lmcache with this flake's module

This repo already defines the LMCache unit: `systemd.services.vllm-lmcache` in `nix/lmcache.nix:194`, run as user `vllm` (`nix/module.nix:286`). `vllm.service` requires it and starts after it (`nix/module.nix:344-348`). So the host must take the unit from this flake's NixOS module, not repoint the fork's definition.

1. Read the current fork unit and save it, so the rollback can restore it exactly:
   ```sh
   systemctl cat vllm-lmcache.service
   ```
2. **Host flake change:** import `inputs.<vllm-input>.nixosModules.default`. That module is the same as `services.vllm-sm70` and already imports `lmcache`, so do **not** also import `.lmcache`. **Delete** the fork's own `vllm-lmcache` definition. Do not keep both: two definitions conflict. Only the input name and the `#c4130` attribute remain for Charlie to fill in.
3. Torch and env: on the build `aczb9zwk`, `bin/lmcache` already runs on the same env (`vzniv8cf…-vllm-sm70-env`) and torch 2.14.0 that the LMCache server used in testing. Confirm with the closure check in step 3.3a, rather than assuming.
   - 3.3a Closure check, on c4130 after the realise in section 2: `nix-store -qR <outPath> | grep -E 'vzniv8cf|005jhwin'` must show both the env and torch 2.14.0.
4. **[SUDO: Charlie]** Switch:
   ```sh
   sudo nixos-rebuild switch --flake <host-flake>#c4130
   ```
   **Note:** restarting `vllm-lmcache` also restarts `vllm.service` (because of `Requires=`). The engine re-registers its KV at start-up. Bea saw a 300 s `register_kv_caches` timeout, so wait for both units to be active before section 4.
5. **[SUDO: Charlie]** Restart the LMCache unit (this restarts `vllm.service` too) and wait for both to report `active`:
   ```sh
   sudo systemctl restart vllm-lmcache.service
   systemctl is-active vllm-lmcache.service vllm.service
   ```

## 4. Verify

Run in order; stop at the first failure and go to section 5.

0. **Users:** both units must run as the same user (`nix/lmcache.nix:96`, CUDA IPC). Bea's prod-server attempt failed with "mapping of buffer object failed" when they ran as different users.
   ```sh
   systemctl show -p User vllm.service vllm-lmcache.service
   ```
   Both must print `User=vllm`.
1. `systemctl is-active vllm-lmcache.service vllm.service` returns `active` for both (after the wait in section 3.4).
2. The vLLM service serves: one completion request against the served model (Qwen or Flint, whichever prod uses) returns a non-empty answer.
3. **LMCache reuse, with prefix caching on** (prod args set `--enable-prefix-caching`, `nix/module.nix:195`). A second identical send is served by the GPU prefix cache, so a plain resend cannot show LMCache reuse. Use these two checks instead:
   - (a) Send a long prompt P (at least two 256-token chunks). Then read the LMCache status and confirm `total_object_count` and `memory_used_bytes` went up. The endpoint is `GET 127.0.0.1:<httpPort>/status` (loopback only, no auth). `httpPort` is `services.lmcache-sm70.httpPort`, default 8080 (`nix/lmcache.nix:127-131`), unless the host overrides it: `curl -s 127.0.0.1:8080/status`.
   - (b) **[SUDO: Charlie]** Restart `vllm.service` only. LMCache keeps its L1 store, and the GPU prefix cache is cleared. Resend P. `prompt_tokens_by_source[external_kv_transfer]` should be close to the prompt length minus the last partial 256-token chunk, and the answer should match the first send.
4. etcd health OK (C3's exit test).
5. **Greedy identity, against a fresh base recorded on `latest`.** The existing `S1c-base1/greedy.json` (under `~/vllm-sm70-work/bea/nov8/logs/`) is **not** a reference for this build: it came from `bfbbf360` with tuning off, the V1 runner and no LMCache connector. Use it only as the source of the 5 prompts. During Bea's GPU gate run on the `latest` commit, with the prod unit's args and env, record a fresh base `greedy.json` and compare the cutover run against it. Path of the new file: TODO, filled in when it exists.

## 5. Rollback to the fork

Use this if any step in section 4 fails, or if prod misbehaves within the first hour after the switch.

1. **[SUDO: Charlie]** Roll the system back to the recorded generation:
   ```sh
   sudo nixos-rebuild switch --rollback
   ```
   Or switch to the generation from section 1 by number.
2. Restore the fork's flake input in the host flake (the commit recorded in section 1) and the original `vllm-lmcache` definition (the `systemctl cat` output saved in section 3.1). Rebuild with **[SUDO: Charlie]** `sudo nixos-rebuild switch --flake <host-flake>#c4130`.
3. **[SUDO: Charlie]** `sudo systemctl restart vllm-lmcache.service`, then wait for `vllm.service` to be active.
4. Confirm the service serves a request with the fork's build.
5. Record the rollback in `docs/release.md` (History) and in the cutover PR. Do not move `latest` backward (`docs/release.md` says never move it backward; revert on `main` and re-gate instead).

## 6. After a successful cutover

- Record the cutover in the `docs/release.md` History table (commit, CI, C1, GPU gate, OK).
- C4: archive the fork (Charlie). Do not do this until the system has run stably for a period agreed with Astro.

## Open items for Charlie

- Name the host flake's vllm input name and the `#c4130` attribute (section 2 and 3.2).
- Confirm the GC retention policy (how many generations to keep).
- Confirm the downtime window for the switch.
- Record the fresh base `greedy.json` on `latest` during the gate run and fill its path in section 4.5.
- Run §4.3b (restart `vllm.service`, then resend) once in Bea's gate window before C3 relies on it. It is untested: every reuse proven so far happened within one vLLM instance.
