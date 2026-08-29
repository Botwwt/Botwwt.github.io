# SAMU GPU：与官方 RG‑LRU 的公平对比

静态汇报页面与 H800 测试代码。主页只加载 canonical SAMU 与官方 RG‑LRU‑16 主测，不加载 grouped SAMU、direct control、特殊函数近似或压缩分块转移候选。

## 主对比

- RG‑LRU 来源：RecurrentGemma 提交 `2efa84dac0e68e63547a27a18fa943c98f1c312e`。
- RG‑LRU 结构：Griffin 方程、分段重置语义、16 个块对角门分块。
- SAMU 结构：单组共享控制的 canonical 方程，稠密复数写入，通用指数/正弦/余弦。
- 精度：BF16 输入、输出与投影；FP32 递推缓存。
- 状态匹配：M 个复数 SAMU 模态按 2M 个实数计，双方缓存字节相同。

## Griffin 第 4、5 节实验轴

`benchmark/run_canonical_griffin_axes.py` 包含两条主测：

1. 附录图 8(a) 对应形状：B=8、1024 个实状态量、L=2K/4K/8K/16K，比较投影后的顺序与分块前向递推。
2. 第 5 节对应轴：D_RNN=2560、B=16、空提示/4K 提示、连续 128–4096 步；吞吐在固定 B=1–256 候选中筛选后完成 512–4096 步整条轨迹。

第二条是单递推层测试，不是完整 1.3B 模型。仓库没有训练好的 canonical SAMU Griffin 模型，也没有 backward Triton，因此不报告完整训练速度或模型质量优势。详细范围见 [GRIFFIN_PROTOCOL_STATUS.md](GRIFFIN_PROTOCOL_STATUS.md)。

## 运行

```bash
cd SAMU_GPU/benchmark
python run_canonical_griffin_axes.py \
  --output ../benchmark_results_griffin_complete/canonical_griffin_axes.json \
  --rglru-source /path/to/recurrentgemma-pinned
```

## 主要文件

- `benchmark/triton_samu.py`：canonical SAMU 顺序、分块和单步解码 Triton 内核。
- `benchmark/triton_rglru.py`：官方方程 RG‑LRU 顺序、分块和融合单步解码 Triton 内核。
- `benchmark/run_canonical_griffin_axes.py`：第 4、5 节实验轴主测。
- `SAMU_CANONICAL_AUDIT.md`：SAMU 方程来源与代码审计。
- `GRIFFIN_PROTOCOL_STATUS.md`：论文系统实验覆盖表。
