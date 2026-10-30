# AKB 第三阶段 3.3 结果审计与验收报告

## A. 背景

- 分支 `feat/domain-task-suite`，代码基线 `7a18146`
- CANN Bench 来源 `2d7cd332247f33a66ca67c77d9f09d446a780dfe`
- 冻结配置：`deepseek-flash` / `zero_shot` / `low` / `32768` / `temperature=0` / `n=1` / `ascend910b4` / `ascendc` + `cmake`
- 五任务：rmsnorm、swiglu、softmax、gather、apply_rotary_pos_emb（各 20 官方 Case）
- 设备：Ascend 910B4（本轮评测时为 NPU 2）Health OK，CANN 9.1.0，driver 25.5.1，torch 2.10.0 + torch_npu

## B. 执行情况（5 次真实请求，无重试）

| Task | 耗时 | 生成 | 说明 |
|---|---|---|---|
| rmsnorm | 5:47 | ❌ truncated | `response truncated at max_tokens` |
| swiglu | 1:37 | ✅ 1 候选 | |
| softmax | 2:18 | ✅ 1 候选 | |
| gather | 1:16 | ✅ 1 候选 | |
| apply_rotary_pos_emb | 6:11 | ❌ truncated | `response truncated at max_tokens` |

## C. 结果总表

| Task | generation | static | compile_attempted | compile | runtime | case_exec | case_passed | task_correct | timing | speedup | failure_stage |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rmsnorm | truncated | N/A | N/A | N/A | N/A | 0 | 0 | N/A | N/A | N/A | generation |
| swiglu | success | **FAIL（2 项）** | no | not attempted | no | 0 | 0 | False | N/A | N/A | static_check |
| softmax | success | **FAIL（2 项）** | no | not attempted | no | 0 | 0 | False | N/A | N/A | static_check |
| gather | success | **FAIL（2 项）** | no | not attempted | no | 0 | 0 | False | N/A | N/A | static_check |
| rope | truncated | N/A | N/A | N/A | N/A | 0 | 0 | N/A | N/A | N/A | generation |

> 设备侧历史 JSON 中 `build_state=build_failed` 是**内部状态映射**（静态检查失败被归入同一 `build_state` 字段），**不代表编译器被实际调用**；本报告按 `static_check` 记录真实失败阶段，且**未修改任何历史 JSON**。

## D. 三份候选审计证据

| Task | custom_op.asc SHA256 | model_new.py SHA256 |
|---|---|---|
| swiglu | `1850f79c47cf4c012ce531b274dd939ce23c95c737c6cb6722fdd03863e2f361` | `142a562a8de40a9b64109cdbd68e21b40f7309e1ba4e9c1f6302d62943a019dc` |
| softmax | `34d9c8c8797c0adf708135206fe1aeb5dd6f22b8b47580610d63fd0758306fbf` | `b4f70bf6c034f37deafb93f977ee4d4de98f28f0c7c310dfbbf8bb265b8ef55f` |
| gather | `cb50c16978aad9ee7248720fe1c65ebade8c0f96269e8f4afa00d1e906e16cf8` | `2fad921f01a71009ff9c609595420e801ab4d5f151b41b4bcbfa3e481773ed73` |

- Kernel 入口均为 `__global__ __vector__ void <name>(`，**无 `GM_ADDR`** ✓（3.1 的入口契约被模型采纳）
- `response_raw.txt` 中**没有** `model_new_py` JSON 字段（模型走 fenced 代码块路径）
- **关键发现 1**：`model_new.py` 用 `ast.parse` 可以**正常解析** ✓，但静态检查报告 `model_new.py does not parse as Python: invalid syntax (line 8/10/9)` ✗ → **二者矛盾** ✗
- **关键发现 2**：`try/except` 在**原始响应**与**保存的 wrapper** 中都出现（各 2 处），说明该规则命中来自模型实际输出（`Contains try-except block` 属规则触发，**不断言作弊**）
- 结论：`try-except` 违规可归因于**模型输出**（A 类）；`does not parse as Python` 一项**不能归因于模型**，疑为**静态检查规则与解析路径的交互问题**（B 类，评测框架侧）→ 需最小复现后由用户决定是否修复，本轮**未修改**

## E. 统计（分母明确）

| 指标 | 值 |
|---|---|
| 计划生成 / 实际请求 | 5 / 5 |
| 生成成功率 | **3/5** |
| 接受静态检查的候选 | 3 |
| 条件静态检查通过率 | **0/3** |
| 真实进入编译的候选 | **0** → `Compilation not attempted: 0 eligible candidates` |
| 编译成功 | 0（分母 0，**不写 0%**） |
| 真实设备执行 | 0 |
| 端到端任务成功（5 次尝试中） | **0/5** |
| 有效正确性样本 | 0 |
| 有效性能样本 | 0（Speedup = N/A） |

## F. 归因

| 失败 | 归因 | 依据 |
|---|---|---|
| 2 次截断 | 模型/生成侧 | 日志明确 `truncated at max_tokens` |
| `try-except` 命中（3/3） | 模型输出 | 原始响应与落盘 wrapper 均含该文本 |
| `model_new.py` 语法错误（3/3） | **疑似框架侧** | 同一文件 `ast.parse` 成功，规则报告失败 → 矛盾 |
| 未编译 | 前置拦截（预期行为） | 静态检查先行，非编译器缺陷 |

## G. 第三阶段验收判断

- **3.1 完成** ✓（环境、五任务 100 case、属性通道、比较器、Prompt、评分预检）
- **3.2 按协议执行完成** ✓（5 次请求、3 份候选、失败如实记录、无伪造数据）
- **3.3 审计完成** ✓（本报告）
- **第三阶段端到端链路未获正向证据** ✗：从未有候选通过静态检查 → 编译、NPU 执行、多 Case 正确性、动态属性设备一致性、有效计时与 Speedup、RoPE 双输出设备验证**均未执行**
- **是否存在框架缺陷**：**是**（`model_new.py` 解析矛盾），只记录，未修复
- **是否足以支持第四阶段**：**不足** ✗ —— 缺少"候选通过静态检查 → 编译 → NPU 执行 → 正确性/计时"的正向链路证明

## H. 证据索引

- Run：`runs/s32_{swiglu,softmax,gather}/cann/nn/<task>/sample_0/`（候选、原始响应、Prompt）
- 生成日志：`experiments/phase3_2/gen_<task>.log`、`progress.log`、`driver.log`
- 设备：`/cache/akb/exp/s32/eval_<task>.log`、`case_results.json`、`eval_result.json`
- 协议：`experiments/phase3_2/PROTOCOL.md`
- 官方 SHA：`2d7cd332247f33a66ca67c77d9f09d446a780dfe`
- 代码基线：`7a18146`

> token 明细未被 CLI 输出，故不报告；无编造的性能或正确性数据。
