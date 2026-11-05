<!-- DRAFT for github.com/LMCache/LMCache issues. Not filed: Charlie decides. -->

# [MP] vLLM worker reaped while alive when no full-chunk request arrives within `--worker-registration-grace-seconds`; it never re-registers

## Summary

In MP mode (`lmcache server` + vLLM's `LMCacheMPConnector`), the worker
adapter only starts its heartbeat on its first STORE, RETRIEVE or KV-event
poll. If a vLLM instance registers its KV caches and then serves only prompts
shorter than one chunk (or nothing) for longer than
`--worker-registration-grace-seconds` (default 3600 s), the server reaps the
instance even though the worker is alive. The worker never notices: `PING`
always returns `True`, so the heartbeat never sees an unhealthy→healthy edge
and never re-registers. From then on every STORE is rejected as
"unregistered" and lookups find nothing, until vLLM restarts.

## Versions

- LMCache 0.5.6 (3e11e10812b448142c6cdb177a3058652bd826fa). Also present in
  v0.5.7rc4 (236f3eec) and on `dev` at 6e88e8e7 (2026-11-04).
- vLLM v0.33.0 with `kv_connector=LMCacheMPConnector`,
  `kv_connector_module_path=lmcache.integration.vllm.lmcache_mp_connector`,
  TP=2 (each TP worker registers its own instance), chunk size 256.

## Reproduction

1. Start `lmcache server --chunk-size 256 ...` and vLLM with the MP connector.
   The server logs `Registered KV cache for GPU ID <id>` once per TP worker.
2. Send only short requests (< 256 prompt tokens), or nothing, for 61 min.
3. At 3600 s the server logs
   `Reaped GPU instance <id>: silent for 3600.1s (pinged=False)`.
4. Send any long prompt (e.g. 11k tokens) twice: the server logs
   `Rejecting STORE for unregistered GPU instance ID <id>`, `/status` shows no
   registered GPU instances, and the second request gets no external cache hit.

Faster: `--worker-registration-grace-seconds 60 --worker-reap-timeout-seconds 60`
reproduces it about a minute after registration.

Observed in production (2026-11-04, times EST): registration 12:25:58 and
12:26:00; 18 short chat requests at 12:27; reaped at 13:25:58 and 13:26:28
(`pinged=False`); 248 `Rejecting STORE` lines between 15:20 and 15:42. On an
earlier run where long prompts started 4 minutes after startup, there were no
reaps and the external prefix-cache hit rate was ~9%.

## Root cause (file:line at v0.5.6)

- `lmcache/integration/vllm/vllm_multi_process_adapter.py`
  - `LMCacheMPWorkerAdapter._ensure_heartbeat_started` (1682-1705) is lazy.
    It is called only from `submit_store_request` (1811),
    `submit_retrieve_request` (1892) and `get_kv_events` (2326), not from
    `register_kv_caches` (1591-1631).
  - `HeartbeatThread._execute` (575-603) re-registers only on an
    unhealthy→healthy transition.
- `lmcache/integration/vllm/lmcache_mp_connector.py:985-1019`
  (`wait_for_save`) submits stores only for full-chunk STORE ops, so short
  prompts never reach the worker adapter.
- `lmcache/v1/multiprocess/modules/lmcache_driven_transfer.py`
  - `reap_stale_instances` (335-378, log at 368) judges never-pinged instances
    against `registration_grace_s`.
  - `touch_instance` (315-328) is a no-op for an instance that is no longer
    tracked.
  - STORE for an untracked instance is rejected (615-627).
- `lmcache/v1/multiprocess/modules/management.py:105-120`: `ping()` returns
  `True` for every instance, including reaped ones.
- `lmcache/v1/multiprocess/config.py:134`: `worker_registration_grace_seconds`
  defaults to 3600.

## Proposed fix

Start the heartbeat when the KV caches are registered. A live worker then
pings every `heartbeat_interval` from startup, and a dead one is still reaped
after `worker_reap_timeout_seconds`:

```diff
--- a/lmcache/integration/vllm/vllm_multi_process_adapter.py
+++ b/lmcache/integration/vllm/vllm_multi_process_adapter.py
@@ -1629,6 +1629,12 @@ class LMCacheMPWorkerAdapter:
             layout_hints if layout_hints is not None else vllm_layout_hints()
         )
         self._send_register_kv_caches_request(kv_caches)
+        # Ping from registration on, not from the first store/retrieve: a
+        # worker that serves only sub-chunk prompts (or none) for longer than
+        # --worker-registration-grace-seconds would otherwise be reaped while
+        # alive, and a later PING (always True) never re-registers it, so every
+        # STORE is rejected until vLLM restarts.
+        self._ensure_heartbeat_started()
 
     def _block_ids_per_group(self, op: LoadStoreOp) -> list[list[int]]:
         return expand_engine_block_ids(self.engine_group_infos, op.block_ids)
```

A follow-up for robustness: let `PING` report an unknown `instance_id`, for
example by returning `False` or a distinct status, so a worker that was reaped
anyway (e.g. after a long stall) re-registers through the existing recovery
callback instead of staying silently unregistered.

## Workaround

Raise `--worker-registration-grace-seconds`, at the cost of keeping dead
never-pinged registrations (and their IPC-mapped KV) for that long, or make
sure each vLLM instance serves at least one full-chunk prompt within the grace
period.
