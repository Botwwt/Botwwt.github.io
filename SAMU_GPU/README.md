# SAMU GPU：与官方方程 RG‑LRU 的 H800 系统对比

静态汇报页面与 H800 测试代码。主页只加载标准 SAMU 与按固定 RecurrentGemma 提交审计的 RG‑LRU‑16 主测，不加载分组控制、直接控制、特殊函数近似或压缩分块转移候选。RG‑LRU 的 GPU 内核是本项目按官方方程实现的 Triton 版本，不称作 Google 官方 CUDA 实现。

## 主对比

- RG‑LRU 来源：RecurrentGemma 提交 `2efa84dac0e68e63547a27a18fa943c98f1c312e`。
- RG‑LRU 结构：Griffin 方程、分段重置语义、16 个块对角门分块。
- SAMU 结构：单组共享控制的标准方程，把 Hawk 已有输入分支直接拆为复数写入，并使用通用指数/正弦/余弦。
- 精度：训练保留 FP32 主参数与 AdamW 动量，矩阵和激活使用 BF16 自动混合精度，递推累加与缓存使用 FP32；推理权重为 BF16。
- 状态匹配：M 个复数 SAMU 模态按 2M 个实数计，双方缓存字节相同。

## Griffin 第 4、5 节实验轴

当前包含四条互不替代的主测：

1. 附录图 8(a)：B=8、1024 个实状态量、L=2K/4K/8K/16K，比较框架逐步循环、自定义片上顺序扫描、精确分块扫描与 BF16/FP32 结合扫描。
2. 图 3：按 Griffin 表 2 的 400M 与 1.3B 宽度和层数运行；L=2K/4K/8K，每步固定 8192 个词元，记录完整训练步与峰值显存。双方先在独立样本上选择最快的精确 GPU 后端，再以新样本按正序和逆序复测。默认使用 32K 词表。
3. 附录图 8(b)：按表 2 的 400M 与 1.3B 配置，只更换递推扫描后端，记录完整优化器步骤；PyTorch 逐步路径只作为官方语义参考，不作为架构主对手。
4. 第 5 节：1.3B 配置完整模型，宽度 2048、递推宽度 2560、24 层、32K 词表；B=16、空提示/4K 提示、连续生成 128–4096 步；吞吐在预先声明的 B=1–512 候选中完成 512–4096 步整条轨迹。

SAMU 与 RG‑LRU 都有 FP32 状态驻留的 Triton 前向、反向和融合解码路径。系统训练输入使用随机词元提供固定算子形状；它不产生准确率、损失或收敛结论。单卡规模轨道真实运行 400M/1.3B 完整优化器步骤。详细协议见 [GRIFFIN_PROTOCOL_STATUS.md](GRIFFIN_PROTOCOL_STATUS.md)。

## 2026-08-31 strengthened-baseline 训练后端

当前训练 dispatch 不再使用单一全局 kernel。SAMU 在短/高 batch 使用 shared-SFU、fused write/output 和 serial K=32；在 medium-wide 与 16K–32K 长序列使用 grouped complex-affine prefix64 K=32；65K 以上为通过严格数值 gate 的 serial forward prefix + grouped conjugate reverse。reset/packed sequence 自动回退到 exact canonical 路径。`grouped` 只表示 chunk summary 的执行分组，不改变控制、模态或模型方程。

同一 H800、两轮反序计时、每轮 5 个样本的 complete mixer F+B 主测中，SAMU 在 `(B,L,D)=(4,2048,2048)`、`(1,8192,2560)`、`(1,32768,1024)`、`(1,32768,2048)` 分别快 24.6%、27.6%、37.3%、40.2%，peak allocated 低 22.0%–23.5%。反例也保留：`(1,8192,1024)` 上 SAMU 慢 12.4%。完整 recurrent block 快 5.8%–7.6%，对应 optimizer step 快 3.8%–6.5%。详情见 [最终 H800 报告](docs/SAMU_FINAL_H800_REPORT.md)。

RG-LRU baseline 同时强化为融合 gate preparation/replay、reset-aware exact chunk forward/backward、partial chunk、nonzero initial state、反向重算及 O(C) grouped real-affine hierarchy。相同受限 no-reset 完整路径下，本实现 2.260 ms，Fattori 开源 Hawk/RG-LRU 原仓库为 4.535 ms。pure scan-only 中 accelerated-scan/Hippogriff 为 0.710 ms、Lingua wrapper 为 0.736 ms、我们的 generic chunk32 为 0.858 ms，因此不声称我们的 pure scan primitive 最快。

## 运行

```bash
cd SAMU_GPU/benchmark
python run_canonical_griffin_axes.py \
  --output ../benchmark_results_griffin_complete/canonical_griffin_axes.json \
  --rglru-source /path/to/recurrentgemma-pinned
```

完整模型训练步与推理：

```bash
python run_paper_scale_h800.py --data data/enwik8-bytechars.txt --vocab-size 32000 \
  --samu-backend chunk16 --rglru-backend chunk32 \
  --output results/paper_scale_h800.json
python run_paper_scale_backend_ablation.py --vocab-size 32000 \
  --output results/paper_scale_backend_ablation.json
python run_paper_scale_inference_h800.py \
  --output results/paper_scale_inference_h800.json
```

## 主要文件

- `benchmark/triton_samu.py`：标准 SAMU 顺序、分块和单步解码 Triton 内核。
- `benchmark/triton_rglru.py`：按官方方程实现的 RG‑LRU 顺序、分块和融合单步解码 Triton 内核。
- `benchmark/triton_training_scan.py`：双方训练前向、反向与 SAMU 融合转移重建。
- `benchmark/triton_fused_rglru_training.py`：canonical RG-LRU 融合、reset-aware exact chunk 与 grouped real-affine hierarchy。
- `benchmark/triton_samu_affine_warp_scan.py`：SAMU complex-affine hierarchy 的独立 reference/candidate 路径。
- `benchmark/triton_samu_controller.py`：SAMU controller FP32 reduction 与 backward 候选。
- `benchmark/run_selected_dispatch_comparison.py`：最终 mixer dispatch 的 counterbalanced H800 复测。
- `benchmark/run_small_model_study.py`：Hawk 模型骨架与训练、解码计时基础实现。
- `benchmark/run_paper_scale_backend_ablation.py`：表 2 规模下图 8(b) 的完整模型扫描后端矩阵。
- `benchmark/audit_official_hawk_block.py`：固定官方提交与本实现的逐组件参数量、整块前向和末状态数值审计。
- `benchmark/run_paper_scale_h800.py`：400M/1.3B 表 2 形状的单卡完整训练步骤。
- `benchmark/run_paper_scale_inference_h800.py`：1.3B 配置的完整模型连续生成与批量搜索。
- `benchmark/run_h800_roofline_and_decode.py`：H800 实测复制带宽、BF16 矩阵吞吐和第 5 节字节模型。
- `benchmark/run_canonical_griffin_axes.py`：第 4、5 节实验轴主测。
- `benchmark/audit_training_equations.py`：双方分块前向与反向梯度的独立方程审计。
- `SAMU_CANONICAL_AUDIT.md`：SAMU 方程来源与代码审计。
- `GRIFFIN_PROTOCOL_STATUS.md`：论文系统实验覆盖表。
