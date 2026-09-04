# PetaKit5D 两项核心创新：5 分钟汇报稿

建议总时长：4 分 40 秒至 5 分钟。公式只说明含义，不逐个念符号。

## 0:00–0:30　开场

各位老师好，我汇报的论文是 2024 年发表在 *Nature Methods* 上的 *Image processing tools for petabyte-scale light sheet microscopy data*。这篇论文提出了 PetaKit5D，用来处理 PB 级光片显微图像。

这次我主要介绍两个创新点：第一是并行读入与分布式处理；第二是 Richardson–Lucy 反卷积中 OMW 反向投影器的设计。

## 0:30–1:00　研究背景

光片显微镜可以快速采集三维图像，但单个相机的数据产生速度可以接近每小时 4 TB。因此，处理速度只有达到或接近采集速度，在线处理才有意义：

$$
v_{\mathrm{processing}}\gtrsim v_{\mathrm{acquisition}}.
$$

这里的瓶颈不只是 GPU 计算，还包括读取、写入、内存容量和任务失败后的恢复。所以论文同时优化了系统层和算法层。

## 1:00–2:15　创新一：多层级并行处理

PetaKit5D 的并行处理可以分成三个层级。

第一层是文件内部的多线程 I/O。Cpp-Tiff 使用 OpenMP，把三维 TIFF 的 z 切片或二维图像的 strips 分给多个线程并行读取和解压。写 TIFF 时，由于 TIFF 是单容器，多个线程不能随意同时修改文件目录和偏移，因此程序并行完成压缩，再由单独的 writer 按顺序写入。这个设计没有把“多线程”简单等同于“所有操作同时写文件”，而是根据文件格式的约束选择可以并行的部分。

第二层是 Zarr 分块。Zarr 把三维图像保存为多个独立 chunk，每个 worker 只读取自己需要的空间区域，也能把不同结果块并行写回。这样，单个体数据即使超过一台机器的内存，也可以使用 split–process–merge 处理。

第三层是集群调度。Conductor 把输入、输出、函数和资源需求组织为任务，再提交给 Slurm worker。它会检查输出是否已经存在、监控任务状态，并在任务失败后重新提交。因此，这项创新不只是让单次计算更快，也提高了 PB 级长流程的可扩展性和可靠性。

论文报告 Cpp-Tiff 的压缩数据读取速度比传统实现提高 22 倍以上，Cpp-Zarr 读取提高约 10 到 23 倍。

## 2:15–3:25　创新二的基础：RL 如何反向修正

第二个创新点是 OMW 反向投影器。先看 RL 反卷积为什么需要反向投影。

把由 PSF 决定的成像算子记为 $H$，真实样本是 $x$，预测图像为：

$$
\hat y=Hx.
$$

由于显微成像中的光子计数可以使用泊松分布描述，忽略与 $x$ 无关的常数后，负对数似然为：

$$
\mathcal L_P(x)=\sum_i\left[(Hx)_i-y_i\ln(Hx)_i\right].
$$

它对样本的梯度是：

$$
\nabla_x\mathcal L_P
=H^{\mathsf T}\left(\mathbf 1-\frac{y}{Hx}\right).
$$

这里最关键的是 $H^{\mathsf T}$。观测图像和预测图像的比值误差先在观测空间中产生，再由 $H^{\mathsf T}$ 传回样本空间。因此，它承担的就是 backward projection。

归一化 PSF 满足 $H^{\mathsf T}\mathbf 1=\mathbf 1$，相应的 RL 乘法更新可以写成：

$$
x_{k+1}=x_k\odot H^{\mathsf T}\left(\frac{y}{Hx_k}\right).
$$

直观上，如果某个位置的实际观测比预测亮，比值就大于 1，反向投影后会增强对应的样本估计；反之则减弱。

## 3:25–4:35　OMW 的具体创新

传统 RL 通常把与 PSF 匹配的算子用于反向投影，方法准确、稳健，但常常需要 10 到 100 次迭代。对 PB 级数据来说，迭代次数会直接放大总计算量。

从二阶优化角度看，泊松目标函数的 Hessian 为：

$$
\nabla_x^2\mathcal L_P
=H^{\mathsf T}\operatorname{Diag}\left(\frac{y}{(Hx)^2}\right)H.
$$

完整 Newton 法需要处理 Hessian 的逆，但对三维大图像代价太高。需要注意，OMW 并不是直接计算 Hessian 的逆。它仍然保留 RL 框架，只是重新设计反向投影器，使它具有稳定的逆滤波性质。

在频域中，OMW 的核心形式是：

$$
\widehat B_{\mathrm{OMW}}
=W\odot\frac{\widehat H^{*}}{|\widehat H|^2+\alpha}.
$$

其中，Wiener 项让反向投影更接近成像过程的稳定逆；参数 $\alpha$ 防止 OTF 很小时噪声被无限放大；窗函数 $W$ 则根据真实 OTF support 构造，只保留显微镜能够可靠传递的频率。

相比已有的 Wiener–Butterworth 方法，OMW 不再使用固定椭球形支持域，而是对 OTF 做阈值分割、主区域保留、凸包填充和 Hann 渐消。因此，对于近矩形频谱的 lattice light-sheet PSF，它不会轻易截掉边缘的高分辨率信息。

论文中 OMW 通常只需 1 到 2 次迭代就能达到全分辨率重建，整体速度比传统 RL 提高一个数量级以上。

## 4:35–5:00　总结

最后总结一下。第一个创新通过多线程 I/O、Zarr 分块和 Conductor 调度，提高了数据在系统中的流动速度；第二个创新通过 OMW 反向投影器，提高了每次 RL 迭代的校正效率，并显著减少迭代次数。

所以，PetaKit5D 的核心不是单独优化某一个函数，而是同时解决“数据搬不动”和“算法收敛慢”两个问题，使 PB 级光片显微数据具备接近实时处理的可能。我的汇报完毕，谢谢各位老师。

## 汇报时的操作提示

1. 开场停留在“总览”，只讲研究背景和两项创新。
2. 点击“创新一：并行”，依次指向 OpenMP、Split–Process–Merge 和 Conductor 三层。
3. 点击“分析与创新点”，重点讲梯度中的 $H^{\mathsf T}$，不要逐项念 Hessian。
4. 点击“OMW 构造”，用一句话解释 Wiener 项、$\alpha$ 和 OTF mask 各自解决什么问题。
5. 如果时间不足，删去 Hessian 公式的口头说明，可节省约 25 秒。

## 可能被问到的问题

### OMW 是不是 Newton 法？

不是。Newton 法需要使用或近似 Hessian 的逆；OMW 是在 RL 迭代框架内，用 OTF mask 与 Wiener filter 构造更有效的 backward projector。两者都体现了利用逆问题结构加速校正的思想，但不是同一种算法。

### 为什么 TIFF 写入没有完全并行？

因为 TIFF 是单容器格式，多个线程同时更新目录结构和文件偏移容易冲突。PetaKit5D 并行化计算量较大的压缩阶段，再由单独 writer 顺序落盘。

### chunk 是不是越小越好？

不是。小 chunk 可以增加并行度和局部访问能力，但会增加文件数量、metadata、调度和固定 I/O 开销，需要在吞吐和并行粒度之间选择合适尺寸。
