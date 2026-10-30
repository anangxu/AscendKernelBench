# NPU self-test evidence (manual fixture)

Not an LLM result and not a benchmark score. It exists to show that one AKB
execution path works end to end on the device.

## How to reproduce

1. Deploy the repository HEAD to `/cache/akb/src` on the 910B4 host.
2. `source /usr/local/Ascend/ascend-toolkit/set_env.sh`
3. `export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:$LD_LIBRARY_PATH`
4. `python /cache/akb/npu_selftest.py`

The script copies the fixture into a throwaway sample directory, calls
`build_custom_op(..., build_backend="cmake")` with the `ascend910b4` hardware
profile, loads the resulting `libcustom_op.so` with `torch.ops.load_library`,
runs one 512x4096 fp32 case, compares against `silu(x0) * x1`, and times 20
iterations with `torch.npu.Event`.

## Observed

See `RESULT.json`. Compile, run, and correctness passed; max abs diff was
9.5e-07 and the measured 27 us comes from 20 iterations.
