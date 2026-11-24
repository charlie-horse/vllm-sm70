# Provenance: inkling2_tiny fixtures

Outputs of the Hugging Face reference classes on a tiny random checkpoint.
`gen_fixture.py` writes every other file here; none is edited by hand.

| | |
|---|---|
| Reference | https://github.com/huggingface/transformers, `src/transformers/models/inkling2_loop` and `inkling2` |
| Commit | `2f1bc5e321b9a736200a590d6e7743ccdeb60772` |
| License | Apache-2.0 (the reference's license) |
| Run | CPU, fp32 (`dtype=torch.float32`), one thread, eager attention, torch conv fallbacks |
| Seeds, torch version, CPU capability, sha256 of each file | `manifest.json` |

## Changes from the reference

- The checkpoint is random, in the released names, shapes and dtypes, with
  tiny config values (`config.json`); `hf_quant_config.json` holds the
  released groups over the tiny tensor names.
- `attention_mask` is stored as boolean "may attend"; the reference passes an
  additive mask.
- `cases.safetensors` `text.visit_states[v]` is the output of each visit in
  execution order. The reference's `output_hidden_states` differs: it is the
  normed embedding, visits 0-46, then the final normed state.
- `manifest.json` `hf_state_dict` holds digests of the loaded weights, not the
  weights.
