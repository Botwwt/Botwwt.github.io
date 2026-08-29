# Griffin 第 4、5 节 H800 对照协议

本项目逐项对应 Griffin 正文第 4 节、第 5 节以及直接支撑它们的附录 D、F。比较对象改为 SAMU 与 RG‑LRU，但不改变论文的核心计量问题：片上顺序扫描是否有效、序列变长时完整训练步怎样变化、逐词元生成的延迟与最大吞吐怎样变化、结果能否由显存流量与缓存大小解释。

## 实验覆盖矩阵

| 论文项目 | 原论文设置 | H800 对照设置 | 结果文件 |
|---|---|---|---|
| 第 4.1 节：模型并行 | Megatron 张量切分、16 组块对角门、ZeRO；多台 TPU‑v3 | 保留 16 组块对角门和可切分的数据布局；租用节点只有一张 H800，不能测跨设备归约与 ZeRO 通信 | `benchmark_results_small_model/environment.json` |
| 第 4.2 节与附录 D.1：低计算强度 | 计算每个状态更新的运算量、读写字节与硬件平衡点 | 按相同公式计算；同时记录 H800 型号、显存容量、计算能力和实测峰值显存 | `benchmark_results_small_model/training_on_device_complete.json` |
| 附录图 8(a)：扫描延迟 | 批量 8、状态宽度 1024、长度 2K 到 16K；Pallas、JAX、BF16/FP32 结合扫描 | 形状与长度完全相同；测试框架逐步循环、自定义片上顺序扫描、精确分块扫描和 BF16/FP32 结合扫描；分别对 SAMU 与 RG‑LRU 执行 | `benchmark_results_small_model/training_on_device_complete.json` |
| 附录图 8(b)：扫描后端对完整模型训练步的影响 | Hawk 的 400M、1B、7B 三种规模 | 三种明确标注的小、中、大 Hawk 风格模型；只切换扫描后端，计入完整训练步 | `benchmark_results_small_model/griffin_scan_backend_matrix.json` |
| 第 4.3 节图 3：长序列训练 | 400M、1B、7B；长度 2K、4K、8K；每批词元总数不变 | 三种明确标注的小、中、大模型；长度 2K、4K、8K；每步固定 8192 个词元；计入前向、损失、反向、梯度裁剪与 AdamW 更新 | `benchmark_results_small_model/griffin_training_matrix_final.json` |
| 第 5.1 节：逐词元生成模型 | 权重字节与批量相关缓存字节除以有效显存带宽 | 记录完整模型参数、BF16 权重字节、FP32 递推状态、卷积缓存、批量和完整轨迹时间 | `benchmark_results_small_model/full_model_benchmark.json` |
| 第 5.2 节图 4：生成延迟 | 批量 16；空提示与 4K 提示；连续生成 128 到 4096 个词元 | 轴与计时边界相同；使用训练后的小型完整模型 | `benchmark_results_small_model/full_model_benchmark.json` |
| 第 5.2 节图 1(b)：最大吞吐 | 空提示；连续生成 512、1024、2048、4096 个词元；每个长度选择最佳批量 | 在共同候选批量 1、4、16、32、64、128、256 中，分别为双方选择完整轨迹吞吐最高值 | `benchmark_results_small_model/full_model_benchmark.json` |
| 附录 F.1–F.4：计算与缓存分析 | 线性层、注意力、参数、KV 缓存、递推状态和卷积缓存 | 只比较两种递推模型时不虚构 KV 缓存对手；逐项报告权重、递推状态与卷积缓存，并用完整生成结果核对近似式 | `benchmark_results_small_model/full_model_benchmark.json` |

单张 H800 无法等价复现原论文的多设备 400M、1.3B、7B 训练，也无法测量 TPU‑v3 的 VMEM 或设备间通信。因此，三规模实验是严格保持实验设计的单卡缩放对照，不使用原论文规模标签。真正的原规模复现需要多卡节点、张量并行、ZeRO 和对应规模的训练检查点。

## RG‑LRU 的固定方程

官方来源固定在 RecurrentGemma 提交 `2efa84dac0e68e63547a27a18fa943c98f1c312e`。设输入为 $x_t$，输入门为 $i_t$，递推门为 $r_t$，则

$$
\begin{aligned}
i_t &= \sigma(W_xx_t+b_x),\\
r_t &= \sigma(W_ax_t+b_a),\\
\log a_t &= -8r_t\odot\operatorname{softplus}(a_{\mathrm{param}}),\\
h_t &= a_t\odot h_{t-1}
      +\sqrt{1-a_t^2}\odot(i_t\odot x_t).
\end{aligned}
$$

实现保留以下官方细节：两组门都使用 16 组块对角线性层；参数与激活使用 BF16；递推累加和解码缓存使用 FP32；文档首词元满足 `segment_pos == 0` 时切断旧状态，并取消首词元的归一化乘子。Triton 输出、最终状态和输入梯度都要通过独立的 PyTorch 方程参考测试后才能进入计时。

## SAMU 的固定方程

主实验使用一组全局共享控制，不使用分组控制或近似特殊函数。$M$ 个复数模态等价于 $2M$ 个实状态量：

$$
\begin{aligned}
\rho_{j,t} &= \exp\!\left[-\nu_j\exp(c_t)\right],\\
\phi_{j,t} &= \theta_j+d_t,\\
z_{j,t+1} &= \rho_{j,t}\exp(i\phi_{j,t})z_{j,t}+w_{j,t}.
\end{aligned}
$$

完整模型直接把递推块已有的宽输入分支拆成实部与虚部作为 $w_t$，没有额外增加一个大矩阵投影。训练扫描在内核内由两个逐词元控制量重建各模态转移；反向扫描逆序重算转移并累计参数梯度。解码内核在一次启动中完成控制投影、转移重建、FP32 状态更新和 BF16 输出。

## 公平计时边界

- 每个性能点先完成编译与预热，再记录多次样本的中位数和原始样本。
- SAMU 与 RG‑LRU 使用相同数据、词元窗口、公共层、批量候选、精度、优化器和 CUDA Graph 边界。
- 完整训练步包含前向传播、交叉熵、反向传播、梯度裁剪和 AdamW 更新。
- 完整生成轨迹包含每个新词元的词嵌入、六个递推块、最终归一化、词表投影和词元选择；提示预填充与后续生成分开计时。
- 结果表保留失败、显存不足和不支持状态，不删除不利结果。
