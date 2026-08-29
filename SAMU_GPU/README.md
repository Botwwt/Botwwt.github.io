# SAMU GPU：与官方 RG‑LRU 的公平对比

静态汇报页面与 H800 测试代码。主页只加载 canonical SAMU 与官方 RG‑LRU‑16 主测，不加载 grouped SAMU、direct control、特殊函数近似或压缩分块转移候选。

## 主对比

- RG‑LRU 来源：RecurrentGemma 提交 `2efa84dac0e68e63547a27a18fa943c98f1c312e`。
- RG‑LRU 结构：Griffin 方程、分段重置语义、16 个块对角门分块。
- SAMU 结构：单组共享控制的 canonical 方程，稠密复数写入，通用指数/正弦/余弦。
- 精度：BF16 输入、输出与投影；FP32 递推缓存。
- 状态匹配：M 个复数 SAMU 模态按 2M 个实数计，双方缓存字节相同。

## Griffin 第 4、5 节实验轴

当前包含四条互不替代的主测：

1. 附录图 8(a)：B=8、1024 个实状态量、L=2K/4K/8K/16K，比较框架逐步循环、自定义片上顺序扫描、精确分块扫描与 BF16/FP32 结合扫描。
2. 图 3：三种单卡缩放模型，L=2K/4K/8K，每步固定 8192 个词元，记录完整训练步与峰值显存。
3. 附录图 8(b)：三种模型规模，只更换扫描后端，记录完整训练步变化。
4. 第 5 节：训练后的完整六层小模型，B=16、空提示/4K 提示、连续生成 128–4096 步；吞吐在固定 B=1–256 候选中完成 512–4096 步整条轨迹。

完整语言模型使用 Tiny Shakespeare 训练，每种结构运行三个随机种子、每个种子 3000 次参数更新。SAMU 与 RG‑LRU 都有 FP32 状态驻留的 Triton 前向、反向和融合解码路径。单张 H800 不能替代原论文的多设备 400M、1.3B、7B 训练，因此三规模结果明确标为单卡缩放实验。详细边界见 [GRIFFIN_PROTOCOL_STATUS.md](GRIFFIN_PROTOCOL_STATUS.md)。

## 运行

```bash
cd SAMU_GPU/benchmark
python run_canonical_griffin_axes.py \
  --output ../benchmark_results_griffin_complete/canonical_griffin_axes.json \
  --rglru-source /path/to/recurrentgemma-pinned
```

完整模型与训练矩阵：

```bash
python run_small_model_study.py --mode all --data tinyshakespeare.txt --output results
python run_griffin_training_matrix.py --data tinyshakespeare.txt \
  --checkpoint-root results --output results/griffin_training_matrix_final.json
python run_griffin_scan_backend_matrix.py --data tinyshakespeare.txt \
  --output results/griffin_scan_backend_matrix.json
```

## 主要文件

- `benchmark/triton_samu.py`：canonical SAMU 顺序、分块和单步解码 Triton 内核。
- `benchmark/triton_rglru.py`：官方方程 RG‑LRU 顺序、分块和融合单步解码 Triton 内核。
- `benchmark/triton_training_scan.py`：双方训练前向、反向与 SAMU 融合转移重建。
- `benchmark/run_small_model_study.py`：三随机种子训练、完整训练步、生成延迟与最大吞吐。
- `benchmark/run_griffin_training_matrix.py`：图 3 的三规模固定词元训练矩阵。
- `benchmark/run_griffin_scan_backend_matrix.py`：图 8(b) 的完整模型扫描后端矩阵。
- `benchmark/run_canonical_griffin_axes.py`：第 4、5 节实验轴主测。
- `benchmark/audit_training_equations.py`：双方分块前向与反向梯度的独立方程审计。
- `SAMU_CANONICAL_AUDIT.md`：SAMU 方程来源与代码审计。
- `GRIFFIN_PROTOCOL_STATUS.md`：论文系统实验覆盖表。
