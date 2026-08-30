# SAMU GPU：与官方 RG‑LRU 的公平对比

静态汇报页面与 H800 测试代码。主页只加载标准 SAMU 与官方 RG‑LRU‑16 主测，不加载分组控制、直接控制、特殊函数近似或压缩分块转移候选。

## 主对比

- RG‑LRU 来源：RecurrentGemma 提交 `2efa84dac0e68e63547a27a18fa943c98f1c312e`。
- RG‑LRU 结构：Griffin 方程、分段重置语义、16 个块对角门分块。
- SAMU 结构：单组共享控制的标准方程，把 Hawk 已有输入分支直接拆为复数写入，并使用通用指数/正弦/余弦。
- 精度：训练保留 FP32 主参数与 AdamW 动量，矩阵和激活使用 BF16 自动混合精度，递推累加与缓存使用 FP32；推理权重为 BF16。
- 状态匹配：M 个复数 SAMU 模态按 2M 个实数计，双方缓存字节相同。

## Griffin 第 4、5 节实验轴

当前包含四条互不替代的主测：

1. 附录图 8(a)：B=8、1024 个实状态量、L=2K/4K/8K/16K，比较框架逐步循环、自定义片上顺序扫描、精确分块扫描与 BF16/FP32 结合扫描。
2. 图 3：除小、中、大缩放矩阵外，再按 Griffin 表 2 的 400M、1.3B、7B 宽度与层数运行；L=2K/4K/8K，每步固定 8192 个词元，记录完整训练步与峰值显存。400M/1.3B 先在独立样本上为双方选择最快的精确 GPU 后端，再以新样本按正序和逆序复测。默认使用 32K 词表。
3. 附录图 8(b)：按表 2 的 400M、1.3B、7B 配置，只更换递推扫描后端，记录完整优化器步骤；PyTorch 逐步路径只作为官方语义参考，不作为架构主对手；7B 若超过 80GB 则记录精确容量边界。
4. 第 5 节：1.3B 配置完整模型，宽度 2048、递推宽度 2560、24 层、32K 词表；B=16、空提示/4K 提示、连续生成 128–4096 步；吞吐在预先声明的 B=1–512 候选中完成 512–4096 步整条轨迹。

训练质量使用标准 enwik8 字节流：前 9000 万字节训练、中间 500 万验证、最后 500 万测试；每种结构运行三个随机种子、每个种子 3000 次参数更新。所有六条验证曲线在 500–3000 步持续下降。SAMU 与 RG‑LRU 都有 FP32 状态驻留的 Triton 前向、反向和融合解码路径。单卡规模轨道真实运行可分配的 400M/1.3B 完整优化器步骤；7B 先以参数、梯度和 Adam 状态的严格下界判断容量。详细边界见 [GRIFFIN_PROTOCOL_STATUS.md](GRIFFIN_PROTOCOL_STATUS.md)。

## 运行

```bash
cd SAMU_GPU/benchmark
python run_canonical_griffin_axes.py \
  --output ../benchmark_results_griffin_complete/canonical_griffin_axes.json \
  --rglru-source /path/to/recurrentgemma-pinned
```

完整模型与训练矩阵：

```bash
python run_small_model_study.py --mode all --data enwik8-bytechars.txt --output results
python run_griffin_training_matrix.py --data enwik8-bytechars.txt \
  --checkpoint-root results --output results/griffin_training_matrix_final.json
python run_paper_scale_h800.py --data enwik8-bytechars.txt --vocab-size 32000 \
  --samu-backend chunk16 --rglru-backend chunk32 \
  --output results/paper_scale_h800.json
python run_paper_scale_backend_ablation.py --vocab-size 32000 \
  --output results/paper_scale_backend_ablation.json
python run_paper_scale_inference_h800.py \
  --output results/paper_scale_inference_h800.json
```

## 主要文件

- `benchmark/triton_samu.py`：标准 SAMU 顺序、分块和单步解码 Triton 内核。
- `benchmark/triton_rglru.py`：官方方程 RG‑LRU 顺序、分块和融合单步解码 Triton 内核。
- `benchmark/triton_training_scan.py`：双方训练前向、反向与 SAMU 融合转移重建。
- `benchmark/run_small_model_study.py`：三随机种子质量训练与小模型系统对照。
- `benchmark/run_griffin_training_matrix.py`：图 3 的三规模固定词元训练矩阵。
- `benchmark/run_paper_scale_backend_ablation.py`：表 2 规模下图 8(b) 的完整模型扫描后端矩阵。
- `benchmark/audit_official_hawk_block.py`：固定官方提交与本实现的逐组件参数量、整块前向和末状态数值审计。
- `benchmark/run_paper_scale_h800.py`：400M/1.3B/7B 表 2 形状的单卡完整训练步与容量审计。
- `benchmark/run_paper_scale_inference_h800.py`：1.3B 配置的完整模型连续生成与批量搜索。
- `benchmark/run_h800_roofline_and_decode.py`：H800 实测复制带宽、BF16 矩阵吞吐和第 5 节字节模型。
- `benchmark/run_canonical_griffin_axes.py`：第 4、5 节实验轴主测。
- `benchmark/audit_training_equations.py`：双方分块前向与反向梯度的独立方程审计。
- `SAMU_CANONICAL_AUDIT.md`：SAMU 方程来源与代码审计。
- `GRIFFIN_PROTOCOL_STATUS.md`：论文系统实验覆盖表。
