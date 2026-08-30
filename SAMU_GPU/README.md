# SAMU GPU：strengthened RG-LRU 下的 H800 训练后端

这是 SAMU GPU backend 的静态报告、Triton 实现、correctness tests 与原始
H800 结果。主页只呈现通过预声明 correctness gate 的最终 shape dispatch；
被拒绝的候选与反例仍保留，避免把探索结果误写成默认路径。

## 当前证据范围

当前 strengthened-baseline 端到端证据覆盖：

1. 一个 complete mixer：controller/gate preparation、recurrence、dense output
   cotangent、input gradient 与全部参数梯度；
2. 一个完整 `RecurrentBlock`：公共 projection、depthwise convolution、norm、
   join/output 与 gated MLP 全部计入；
3. 同一个单 block 的 fused AdamW step，optimizer state 在计时前初始化。

结果中的 `400M-width` 只表示一个 block 使用论文 400M 配置对应的宽度，
不是当前 strengthened dispatch 下的 12 层完整 400M 语言模型。仓库中的较早
400M/1.3B 多层训练与生成脚本、JSON 仍作为历史实验轨道保存，但不进入当前
主页排名，也不能用来声称当前完整语言模型或推理已经获胜。

## 最终 H800 结论

在 BF16 input/output、FP32 recurrent carry、编译排除、两轮 AB/BA、每轮五个
计时样本的主协议下：

- 四个主要 complete mixer 形状的 F+B 分别快 24.6%、27.6%、37.3%、40.2%，
  peak allocated 低 22.0%–23.5%；
- 单个完整 recurrent block F+B 快 5.8%–7.6%；
- one-block AdamW step 快 3.8%–6.5%；
- `B=1,L=8192,D=1024` 是保留反例：SAMU 1.801 ms，strengthened RG-LRU
  1.603 ms，SAMU 慢 12.4%；
- `D=1024` 的 65K/131K 严格 dispatch 使用 serial forward prefix + grouped
  conjugate reverse，最终优势为 27.2%/25.7%。此前 full-group forward 约 43%
  的结果未通过输出误差 gate，不进入最终结论。

完整数字、区间、显存、scratch、launches 与 claim 边界见
[`docs/SAMU_FINAL_H800_REPORT.md`](docs/SAMU_FINAL_H800_REPORT.md)。

## SAMU backend

GPU 优化不改变正式 controller：normalized projection directions、
zero-initialized signed amplitudes、规定的 H/mode scaling、
`q(r)=r/(1+r²)`，且 controller amplitude 为零时精确退化到 RTU。

最终实现利用 low-rank coherent control 的结构：

- 每 token 只生成共享 `E=exp(c), C=cos(d), S=sin(d)`；
- mode-wise complex transition 在 register 中在线重建，不向 HBM 写出
  `lambda_r/lambda_i [B,L,M]`；
- forward 使用 exact complex-affine chunk summaries、group-local/outer/local
  correction hierarchy 与 K=32 replay；
- backward 使用 conjugated transition 的 reverse affine hierarchy，并在 replay
  内形成 tile-local controller-gradient partials；
- deterministic FP32 second stage 把 `[B,L,N_tiles,2]` 归约到 `[B,L,2]`，
  不物化 `[B,L,M]` controller-gradient tensor；
- reset、packed sequence 或 unsupported shape 自动回退 exact canonical path。

backend 名称里的 `grouped` 只表示 chunk summaries 的执行分组。它不对 controller、
mode 或参数分组，也不改变 SAMU 方程。

compressed G/D 的 exact closure 已实现，但 compressed + grouped K32 在最终 BF16
interaction gate 的 output relative L2 为 `6.38e-6`，超过预声明 `5e-6`，且层级
路径需要重建 transition；因此它保留为 memory representation 研究，不作为默认
latency path。

## Strengthened RG-LRU

主对手是本仓库按 Google/Hugging Face RecurrentGemma canonical 方程实现并强化的
H800 backend，不是退回旧实现。它包含 fused gate preparation/replay、reset-aware
exact chunk forward/backward、partial final chunk、nonzero `h0`/`grad_h0`、clipped
sqrt backward、BF16 boundaries、FP32 carry、backward recompute 与 O(C) grouped
real-affine forward/reverse hierarchy。Google/HF 只提供数学与 correctness reference；
本项目 kernel 不称作 Google 官方 CUDA 实现。

公开库对比分开报告 contract：

- 相同受限 zero-state/no-reset complete path 中，我们为 2.260 ms，Fattori 原始
  `hawk-pytorch` 为 4.535 ms；Fattori 不称“官方实现”；
- pure scan-only 中 accelerated-scan/Hippogriff 为 0.710 ms、Lingua wrapper
  0.736 ms、我们的 generic materialized chunk32 为 0.858 ms，因此不声称我们的
  pure scan primitive 最快。

## 复现与校验

最终 mixer、block、optimizer、very-long 与公开库 runner 位于 `benchmark/`，
原始结果位于 `results/gpu_optimization/`。站点在发布前运行：

```bash
cd SAMU_GPU
node --check src/final-report.js
node scripts/validate-site.mjs
```

校验器检查主页引用的全部本地资产、主结果的十个原始样本、绝对时间、严格
very-long dispatch、单 block 范围、被拒绝候选和公开库 contract。旧的
`run_paper_scale_*` 脚本及对应结果没有删除，但仅用于审计历史轨道。

主要实现文件：

- `benchmark/triton_training_scan.py`：SAMU forward/reverse training backend；
- `benchmark/triton_fused_rglru_training.py`：strengthened canonical RG-LRU；
- `benchmark/triton_samu_affine_warp_scan.py`：complex-affine hierarchy reference/candidate；
- `benchmark/triton_samu_controller.py`：controller reduction candidates；
- `benchmark/run_selected_dispatch_comparison.py`：最终 mixer AB/BA runner；
- `benchmark/run_long_block_dispatch_comparison.py`：单个 recurrent block runner；
- `docs/OUR_RGLRU_GPU_AUDIT.md`：RG-LRU baseline 审计；
- `docs/SAMU_FINAL_H800_REPORT.md`：当前正式结论。
