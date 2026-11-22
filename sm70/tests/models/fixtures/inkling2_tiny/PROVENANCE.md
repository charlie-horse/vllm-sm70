# Provenance: inkling2_tiny fixtures

Outputs of the Hugging Face reference classes on a tiny random checkpoint.
`gen_fixture.py` writes every other file here; none is edited by hand.

| | |
|---|---|
| Reference | https://github.com/huggingface/transformers, `src/transformers/models/inkling2_loop` and `inkling2` |
| Commit | `2f1bc5e321b9a736200a590d6e7743ccdeb60772` |
| Run | CPU, fp32 (`dtype=torch.float32`), one thread, eager attention, torch conv fallbacks |
| Seeds, torch version, sha256 of each file | `manifest.json` |

Regenerate (the reference tree first on `PYTHONPATH`; `--check` writes to a
temporary directory and compares sha256 with `manifest.json`):

```
PYTHONPATH=<transformers>/src python sm70/tests/models/fixtures/inkling2_tiny/gen_fixture.py [--check]
```

## Files

| file | holds |
|---|---|
| `config.json` | the released `config.json` field names with tiny values: 24 layers, loop 6-18 x 3, hidden 64, 8 experts, window 8, rel extent 8, n_floor 16 |
| `hf_quant_config.json` | the released file's groups over the tiny tensor names |
| `model.safetensors` | the checkpoint in released names, shapes and dtypes: NVFP4 experts in layers 0-19, MXFP8 in 20-23, interleaved `w13`, padded `unembed`, `attn_queries.{i}`, the two cross-pass query tensors |
| `ops.safetensors` | inputs and outputs of single modules during the text case |
| `cases.safetensors` | whole-model inputs and outputs |
| `manifest.json` | provenance, the index of `ops.safetensors`, and a digest of every weight as the reference loaded it |

## Reading them

- `manifest.json` `ops`: `<op>.<call>` gives the module path, its class, each
  forward argument and each output. A string is a key of `ops.safetensors`
  (equal tensors are stored once, so a key can carry another op's name); a
  one-element list is a plain value. `<call>` counts that module's calls in
  the forward from 0.
- `attention_mask` is boolean, true where the query may attend. The reference
  passes an additive mask.
- `manifest.json` `hf_state_dict`: reference parameter name ->
  `[shape, sum, sum of magnitudes, sum of x[i] * (i % 251 + 1)]` over the
  flattened fp32 tensor, in fp64.
- `cases.safetensors`: `text.*` is one prefill of 20 tokens.
  `text.visit_states[v]` is the output of visit `v` in execution order
  (`manifest.json` `visit_layers`); `text.final_hidden_state` is the input of
  the LM head. `text.logits_cached_decode` and `text.logits_chunked_prefill`
  are the same tokens through the reference's cache. `greedy.*` is a greedy
  continuation. `image.*` is two images through the tower and one text + image
  prefill; `image.features` are the rows written over the image tokens.
- The reference's `output_hidden_states` is not `text.visit_states`: it is
  the normed embedding, visits 0-46, then the final normed state.
