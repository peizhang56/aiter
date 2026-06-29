# MLA decode produces wrong output on gfx950 fp8/fp8 nhead=32 when get_mla_metadata_v1 is called without dtype_q/dtype_kv

## Summary

On gfx950, `mla_decode_fwd` (persistent mode) produces NaN/garbage output for the
fp8/fp8, `nhead=32`, `qlen=1` decode case when the persistent metadata is built by
`get_mla_metadata_v1` **without** passing `dtype_q`/`dtype_kv`. These two args are
typed `Optional[torch.dtype] = None` and are undocumented in the function's
docstring, so a caller reasonably treats them as optional — but omitting them
silently corrupts the split/reduce metadata for the `nhead 32 -> 16` fold path.

This surfaced as an end-to-end accuracy collapse in vLLM (Kimi-K2.5, TP2 = 32
q-heads/rank, fp8 KV): gsm8k ~0.93 -> ~0. vLLM was calling `get_mla_metadata_v1`
without the dtypes.

- aiter HEAD at repro: `620287969`
- Device: gfx950 (MI355X)
- Path: `aiter/mla.py` persistent fold branch (`nhead in range(32,128+1,16)` ->
  fold to 16)

## Minimal reproduction (inside aiter's own op_test)

I added a `--mimic-vllm-metadata` flag to `op_tests/test_mla_persistent.py` that
omits `dtype_q`/`dtype_kv` from the `get_mla_metadata_v1` call; everything else is
unchanged. The flag is the only difference between pass and fail:

```bash
# PASS  (dtypes passed -> correct metadata), exit 0
python3 op_tests/test_mla_persistent.py -n 32,1 -d fp8 -kvd fp8 -c 8192 -b 128

# FAIL  (dtypes omitted -> corrupted metadata), exit 1
#   AssertionError: cos_diff < 3e-2   (golden vs aiter_asm)
python3 op_tests/test_mla_persistent.py -n 32,1 -d fp8 -kvd fp8 -c 8192 -b 128 --mimic-vllm-metadata
```

The stock op_test cannot show this today because it always passes the dtypes (and
the qh32 path keeps kv-split disabled). The flag exercises exactly the call shape
a downstream caller hits when treating the dtype args as optional.

## What I measured (controlled, single variable)

Holding q / KV / scales / all other metadata flags fixed and toggling **only**
`dtype_q`/`dtype_kv` on `get_mla_metadata_v1`, comparing the kernel output to a
pure-torch absorbed-MLA reference (gfx950, nhead=32, ctx=8192, bs=128):

| metadata call | cosine vs torch ref |
|---|---|
| dtype_q=fp8, dtype_kv=fp8 | ~0.9999 (OK) |
| dtype_q/dtype_kv omitted  | ~0.59 / NaN (BROKEN) |

Ablation of the other args (is_causal, intra_batch_mode) individually leaves the
result correct — only dropping the dtypes breaks it, and adding them back fixes
it. So the dtypes are necessary and sufficient for this case.

## Ask

The downstream (vLLM) fix is simply to pass the dtypes, and we are doing that.
But since the args are `Optional[...] = None` and undocumented, omitting them
silently yields wrong numerics rather than an error. Could aiter make this
fail-loud instead — e.g. one of:

1. require `dtype_q`/`dtype_kv` when they affect metadata (drop the `None`
   default for the persistent path), or
2. assert / raise when they are needed but missing, or
3. infer a safe default and document the behavior.

Any of these prevents the next caller from hitting a silent-corruption trap.
Happy to share the standalone single-call reproducer and the captured params as
well.
