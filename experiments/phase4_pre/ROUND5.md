# Round 5 (structural skeleton in case prompts)

| Task | Static | Detail | Device |
|---|---|---|---|
| swiglu | FAIL | comparison on non-scalar values — tensor comparisons are com | pending device |
| softmax | PASS |  | pending device |
| gather | PASS |  | pending device |
| scatter | FAIL | comparison on non-scalar values — tensor comparisons are com | pending device |
| linear_cross_entropy | FAIL | comparison on non-scalar values — tensor comparisons are com | pending device |

The device became unreachable during this round: two SSH attempts to
dev-modelarts.cn-southwest-2.huaweicloud.com:31583 timed out, after the same
host accepted deployments earlier in the session. Device evaluation for
these candidates is therefore pending, not failed.

Offline state: every candidate is saved unmodified, static checks pass for
softmax and gather, and the skeleton did not change the KernelBench prompt.
