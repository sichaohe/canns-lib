# canns-lib ripser：CPU 与 GPU 双路径，缩小与 ripser++ / GUDHI 的差距（v2）

> 相对 v1 的改动摘要见文末「变更说明」。

## Summary

**目标**：ripser 模块维护 **CPU 与 GPU 两条独立可交付的路径**，把时间与主机内存的差距拆到可归因粒度并逐段消除。

**范围**：`src/ripser/` 归约路径（`core/reduction.rs` 及调用方）；基准 harness 与门禁；grid-cells 真实数据集。

**非目标**：整模块重写；不删 `lockfree.rs`；不改 `APPARENT_MIN_COLUMNS`；不把 `gpu` 加进 maturin 默认 feature；不推送、不合并（本地临时分支允许，见「回滚」）。

### 两条路径与对手

| 路径 | 当前（封顶输入，ASA 1200，maxdim=2） | 对手 | 差距 |
|---|---|---|---|
| **GPU**（`--features gpu`, A100） | 43.32 s / 26.98 GB | ripser++ 27.53 s / 14.10 GB 主机 + 33.8 GiB 显存 | 慢 1.57x，主机内存 1.91x |
| **CPU**（pip 默认 wheel） | 68.26 s / 14.60 GB | **待定（见阶段 0.3）** | GUDHI 86.00 s 的 barcode 为 `[1195,481,0]`，与我们不等价，**不能作为 1.26x 的依据** |

ripser++ 是 GPU-only，CPU 路径无法与之对等比较。CPU 目标在阶段 0.3 重定，暂以「相对自身基线下降 X%」为准。

### 三条设计决策（均来自已验证证据）

1. **一切对比在相同输入形态下做。** 同构建同 barcode，inf 保留 18.08 s / 8.12 GB，封顶 43.32 s / 26.98 GB。ripser++ 无法表达 `inf`，故与它可比的只有封顶侧。
2. **封顶值语义维持现状。** ripser++ 用完全相同的 `diam <= threshold` 过滤（`ripser++.cu:624,670,871,1008,1062`）。把 P21 扩展到封顶矩阵会改 barcode，是换问题。
3. **先量后改。** 先拿两边同批的「时间 + 内存」分段表，再决定投哪里。

---

## 已验证的现状

### 性能矩阵（ASA 1200x1200，maxdim=2）

| 目标 | 构建 | 输入 | wall | 主机 RSS | GPU 显存 | barcode |
|---|---|---|---:|---:|---:|---|
| ripser++ | CUDA | capped | 27.53 s | 14.10 GB | 16.9->33.8 GiB | `[1195,481,58]` |
| canns-lib | GPU / A100 | inf 保留 | 18.08 s | 8.12 GB | - | `[1195,481,58]` |
| canns-lib | GPU / A100 | capped | 43.32 s | 26.98 GB | - | `[1195,481,58]` |
| canns-lib | CPU 默认 | inf 保留 | 27.12 s | 4.49 GB | - | `[1195,481,58]` |
| canns-lib | CPU 默认 | capped | 68.26 s | 14.60 GB | - | `[1195,481,58]` |
| GUDHI 3.13.0 | CPU 单线程 | capped | 86.00 s | - | - | `[1195,481,0]`（dim2 缺失，原因待查） |

准确率：dim0/1 与 GUDHI 条数一致、数值 1e-6 内一致（非逐位）；dim2 在 n=200 三方（16 条）与 n=1200 对 ripser++（58 条）一致。

### 差距来源

**① CPU 路径缺 P14（分片 pivot 表）。** `core/reduction.rs:1126` 以 `use_gpu_pre` 为硬前置，建表只吃 `gpu_pre`；CPU 有 `apparent_pre`（`reduction.rs:1348-1352`）却无并行建表。CPU 构建 STATS 为 `sharded_pivot=false`。P14 在 GPU 上值 −11.46 s。

**② 归约遍历全部列，ripser++ 只遍历非 apparent 列。** ripser++（`ripser++.cu:2119-2131`）只迭代 `h_num_nonapparent`；我们（`reduction.rs:1320-1350`）对全部列 `append_column()`，99.9% 随即 `continue`。
NOTE: `append_column()` 必须留在快路径前（`subrange` 按列号索引）。跳过 apparent 列需要 `column_number` 与 `submatrix_slot` 的映射，而 `PivotLookup` 用 `a.pos() < before` 判序，**动的是已验证的地基**。
dim<=2 共 288,053,001 个单纯形，`DiameterEntryT` 16 B，即 `columns_to_reduce` 单项 4.6 GB。

**③ 工作集位置不同。** ripser++ 主机 14.10 GB + 设备 33.8 GiB；我们主机 26.98 GB。内存差距很大一部分只是放置位置不同。

**④ 无按显存自适应 dim_max**（`ripser++.cu:1718-1760`）。属健壮性差异，不是性能差异。

### 未解线索

`reduction.rs:1145-1149`：GPU 构建上 walk **63 ns/iter x 286,562,199 ~ 18.1 s**，每次仅搬约 44 B，比单核带宽低一个数量级，疑有停顿；候选缩到 `bounds` push / `births_and_deaths` 扩容两条写模式。
**注意**：该 walk 本质上就是 ② 里的「遍历全部列」，18.1 s 占 GPU 封顶 wall 的 **42%**，已远超 25% 门槛。因此 ② 是 **GPU 路径主战场**，而 63 ns 的停顿很可能随 ② 一并消失。

### 已排除（不再提）

系数环、`--threshold` 语义、`+inf` 处理、ripser++ 有 bug；P13 / P17 / P18 / P19 / P20、每列 `shrink_to_fit`--全部实测负收益。

---

## 分阶段方案

### 阶段 0：前置修复与核实（阻塞项）

**0.1 修 harness。** `gca_bench.py` 冒烟 6/6 FAILED。
1. `FAILED` 分支打印子进程 stderr 与 returncode。
2. 直接跑 `gca_one.py` 定位是崩溃还是正则不匹配。
3. 输出统一为无空格、无逗号的 `key=value` 格式，空值用显式占位符（如 `ratio=NA`），避免 `h1top=[9.77, 9.72, 2.49]` 与 ` md5=` 粘连类解析错误。

验收：单格 `canns-gpu / inf_preserved / 24365_2_ASA_mec_gridModule01_n29_cm` 解析出 `bars` 与 `ratio`，且 `ratio / prev_H1_ratio_l2_over_l3` ∈ **[0.5, 2]**。

**0.2 参照值复现（由 v1 风险 5 前移）。** 用 1 个文件核实 canns-lib 0.7.0 的 manifest 参照值能否复现。能 to 4a 逐项对表；不能 to 降级为量级/排序参照。结论写入 CSV 元数据。

**0.3 重定 CPU 对手。** 查明 GUDHI `[1195,481,0]` 的 dim2 为 0 的原因（maxdim 参数 / 内存失败 / 超时）。若 GUDHI 在该规模无法算 dim2，则 CPU 对手改为：(a) 原版 ripser.py（若可跑完），或 (b) GUDHI 在可跑完的缩小规模上的对比；并保留「相对自身基线」作为主指标。

### 阶段 1：分段耗时 + 内存（阻塞项，不改算法）

**1a - ripser++ 自报时间。** 在**独立目录**重编带 `-DPROFILING` 的二进制（不覆盖 `build2/`），读 `CUDA PREPROCESSING TIME`、`SUM OF GPU MATRIX SCAN … dim N`、`SUBMATRIX REDUCTION TIME … dim N` 等。先读其 Makefile/CMake，确认 `PROFILING` 与 `PRINT_PERSISTENCE_PAIRS` 等宏无互相干扰（v1 未决项 6）。

**1b - 我们两条路径的分段。** 封顶输入上分别跑 GPU / CPU 构建，取 `pre_pass_s / gpu_pre_s / sharded_pivot_build_s / loop_s / total_s / bd_pairs / apparent_iters`，以及 `LOOPFAULT` 的 `iters / minflt / majflt`。

**1c - 内存分段。** 记各阶段峰值 RSS 与主要分配项（至少：`columns_to_reduce`、pivot 表、`reduction_matrix_soa`、`bounds`、`births_and_deaths`），用 `Instant` + `/proc/self/status` 埋点（perf 不可用）。

**1d - 低成本探针（原 2c）。** 用已有 `walk-probe` / `WRITEPROBE`（纯加法，不改 barcode）跑一次，记录 `bounds` push 与 `births_and_deaths` 扩容各占多少。仅作为 2b 被否决时的备选依据，不单独立项优化。

**测量规范（所有阶段通用）**
- 同一时段同批跑，记 loadavg。
- 每个臂 **>=5 次取中位数**，同时记 min/max；「劣化」定义为中位数 **> +3%**；「提升」需中位数改善 > 噪声带（max−min 的一半与 3% 取大者）。
- 内存取 `/usr/bin/time -v` 输出，用正则解析，不按下标取。

**验收**：一张表，行 = 阶段（预扫/建表/scan/归约/装配），列 = ripser++ GPU / ours GPU / ours CPU 的秒数、占比、**未归因余量**，另附内存分段表。没有这两张表不进入阶段 2。

### 阶段 2：只遍历非 apparent 列（2b，GPU 优先判定）

**前提**：阶段 1 数字显示 apparent 列 walk 占比 >=25%（GPU 侧已有证据 42%；CPU 侧待 1b 判定）。

**做法**：引入 `column_number to submatrix_slot` 映射（对应 ripser++ 的 `h_pivot_column_index_array_OR_nonapparent_cols`）。

**风险**：`PivotLookup::get` 的 `a.pos() < before` 判序、`PivotVal.pos` 语义、`reduction_matrix_soa` 按列号索引，三者互相咬合。

**门禁（缺一不可，任一不过则回滚该分支）**
1. ASA 规模**逐位 md5** 一致（224 格网格抓不到：n<=300 时 dim=1 低于 `GPU_MIN_COLUMNS`）。
2. 全臂 md5 一致。
3. 新增 Rust 级差分测试：同输入下新旧路径逐元素比较 `columns_to_reduce`。
4. **`check_inf_bridge` 复核**：inf 保留侧与封顶侧 barcode 仍逐位相同（原风险 4 并入门禁）。
5. 性能：GPU 封顶臂中位数提升超过噪声带；`apparent_iters` 与 walk 时间同步下降。

若 2b 之后 63 ns/iter 停顿消失，则 1d 的探针结论归档，不再追。

### 阶段 3：CPU 路径专项（2a）

**依据**：阶段 1b 中 CPU 的 `loop_s` 占比。占比低则本阶段让位给阶段 2 在 CPU 上的复用。

**3a - CPU 并行建分片 pivot 表。** 去掉 `use_gpu_pre` 硬前置，从 `apparent_pre` 并行构建 `ShardedPivotTable`（`build_from_columns` 本身为并行哈希插入）。
验收：CPU STATS `sharded_pivot=true` 且记录 `sharded_pivot_build_s`；barcode 与改动前**逐位 md5 相同**；中位数不劣化超过 3%（提升则按噪声带判定）；新增 `CANNS_RIPSER_*` kill switch 默认开启。

### 阶段 4：GPU 路径专项

**依赖**：**阶段 2（2b）完成并通过后才可开始**，3a 依赖其 `column->slot` 映射。

**4a - 工作集上设备。** 把 `columns_to_reduce` 与 pivot 表移到设备，CPU 只留归约所需列。
验收：GPU 构建主机 RSS 从 26.98 GB 向 14.10 GB 靠（目标值由 1c 内存分段决定，不预设）；wall 下降；barcode 逐位不变。
NOTE: 会再次改变按列号索引假设，**独立分支、独立验收，不与 2b 混提交**。

**4b - 显存自适应 dim_max。** 注意：这**会改变输出**（少算一维），不是纯健壮性。
- 默认行为：显存不足时**报错**，不静默降档。
- 降档仅在显式开关（如 `CANNS_RIPSER_AUTO_DIM_DOWN=1`）下启用；启用后必须 `warnings.warn` 并在返回值/STATS 中标记 `dim_max_requested` 与 `dim_max_effective`。
验收：构造显存受限场景，验证默认报错、开关下降档且 barcode 在已算维度上正确、标记可见。

### 阶段 5：基准铺开

**5a - grid-cells 主基准（553 格）。** 97 个矩阵缓存于 `~/gridcells/matrices/`（需生成完毕，此前 18/97）。网格 = 97 文件 x 5 臂（canns-gpu x2 形态、canns-cpu x2、ripser++ capped）+ 68 个有 manifest 参照的 GUDHI（若 0.3 改了 CPU 对手则相应调整）。分母写入 checker 的 `EXPECTED`，**矩阵文件数也纳入固定分母**，少一个即 FAIL。
对表策略取决于 0.2 的结论。

**5b - ASA 单点回归守卫（13 格）。** 现有 `final_results.csv` 作为「改动没碰坏已验证地基」的门禁，阶段 2、3、4 每次改动后必跑。

### 阶段 6：交付件

CSV（唯一数据源）+ checker（固定分母 + 负向 self-test）+ MD。

checker 核心约束：

> **加速比必须 barcode 与 `input_form` 都相同。**

其余：列数检查先行；`not_run` 不许填数字；同框架同 maxdim 的 bar 数与 md5 必须一致；ripser++ 废格永不标 `measured`；裁判不在则独立退出码 2，绝不「跳过->报通过」；**对手 barcode 与我们不等价的行（如 GUDHI dim2=0）不得计算加速比**。

---

## 端到端流程

```
.npz 记录 --canns embed/PCA/denoise--▶ 1200x1200 距离矩阵（一次性缓存，含 inf）
                                              |
                    ┌-------------------------┴-------------------------┐
                    ▼                                                   ▼
        inf_preserved（原样）                                    capped（inf->2·tmax+1）
                    |                                                   |
     ┌--------------┼--------------┐                        ┌-----------┴-----------┐
     ▼              ▼              ▼                        ▼                       ▼
 canns-gpu      canns-cpu     （P21 仅 thresh=inf        canns-gpu             canns-cpu
  18.08 s        27.12 s       时触发，基准均传            43.32 s               68.26 s
  8.12 GB        4.49 GB       thresh=cap，故未触发）     26.98 GB             14.60 GB
     `---- 对照 ripser++ 27.53 s / 14.10 GB + 33.8 GiB ----┘（仅 capped 侧可比）
                                  CPU 对手：见阶段 0.3
```

---

## 状态、并发、失败与恢复

- `pip default` 与 `--features gpu` 是两个独立 site 目录（`site_cpudefault` / `site_shard`），互不覆盖；重建任一路径不得影响另一条。
- 新增任何「自动启用」都必须带 `CANNS_RIPSER_*` kill switch（现有：`_APPARENT` / `_SHARDED_PIVOT` / `_NO_AUTO_SPARSE` / `_USE_LOCKFREE` / `_STATS`）。
- GPU 不可用时静默回落 CPU，并通过 `gpu_err` 暴露。
- **barcode 逐位相同是所有改动的硬门禁。**
- **回滚**：每个子项在独立本地临时分支（如 `exp/2b-nonapparent`）上开发，不推送；门禁失败则丢弃分支。未经用户要求不做正式提交到 master。
- 长跑 benchmark 必须带 `timeout`，收尾用 `ps` 核查孤儿进程。
- `/usr/bin/time` 输出用正则解析。

## 环境约束

- 本机 Mac 只改代码；计时 / benchmark 上 `222.29.101.57:1693`（`sichaohe`，conda env `rl`）。
- 本地 cargo：`CARGO_TARGET_DIR=$HOME/.cache/canns-lib-target`（`/Volumes/home` 是 NAS，不支持文件锁）。
- 非交互 ssh：`export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH`。
- `perf_event_paranoid = 4`，只能用 `Instant` 埋点。
- 目录级 `rsync` 会静默不传输；用 `tar -cf - | ssh ... tar -xf -` 并 grep 关键标记验证。
- 机器共享，跨时段数字不可直接比。

## 未决事项与风险

1. grid-cells 矩阵缓存仍在生成，5a 需等待完成。
2. 阶段 2（GPU 主战场）与阶段 3（CPU）的最终先后，以 1b 的 CPU `loop_s` 占比为准；GPU 侧 2b 已有 42% 的证据，可先行。
3. **2b 与 4a 都改 `reduction_matrix_soa` 索引假设**，必须串行（2b to 4a），不可并行，不可混提交。（v1 误写为 2a↔3a。）
4. ripser++ 构建系统能否直接加 `-DPROFILING` 尚未验证（见 1a）。
5. CPU 对手尚未确定（见 0.3），直接影响摘要表与 5a 的 GUDHI 臂。

---

## 变更说明（v1 to v2）

| # | 改动 | 原因 |
|---|---|---|
| 1 | 风险 3 的冲突对象改为 2b↔4a | v1 误写 2a↔3a，2a 不碰索引假设 |
| 2 | 2b 提前并明确为 GPU 主战场，4a 显式依赖 2b | 18.1 s walk = 42% wall，3c 与 3a 关系前后矛盾 |
| 3 | 原 2c 降为阶段 1 的低成本探针 | 63 ns 停顿大概率随 2b 消失 |
| 4 | 3b（现 4b）降档默认报错、显式开关 | 降 dim_max 会改变输出，不是纯健壮性 |
| 5 | 新增阶段 0.3 重定 CPU 对手 | GUDHI barcode `[1195,481,0]` 不等价，违反自身门禁，1.26x 不成立 |
| 6 | 回滚机制改为本地临时分支 | 「不提交」与「回滚到提交」矛盾 |
| 7 | 阶段 1 增加内存分段 | 内存差距 1.91x 是目标，原方案无归因依据 |
| 8 | 新增测量规范（>=5 次中位数，3% 阈值） | 共享机器，原「噪声带」无数字 |
| 9 | 阶段 0 验收给出 ratio 范围 [0.5, 2] | 「同量级」不可判定 |
| 10 | 参照值复现前移到 0.2 | 决定 5a 对表策略，成本仅一个文件 |
| 11 | `check_inf_bridge` 复核并入 2b 门禁 | 原本埋在风险里无人执行 |
| 12 | 摘要 CPU 差距列改写；checker 禁止对不等价行算加速比；矩阵数入固定分母 | 防止误读与重复 v1 类错误 |
