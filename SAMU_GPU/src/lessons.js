export const parts = {
  0: { id: "learn", roman: "I", title: "从方程到 GPU 数据", text: "明确 GPU 实际读取什么、保存什么，以及哪些量随时间或模态变化。" },
  3: { id: "gpu-explorer", roman: "II", title: "针对瓶颈选择执行方法", text: "短序列减少显存往返，长序列利用结合律拆开时间依赖。" },
  7: { id: "compare", roman: "III", title: "与 RG‑LRU 对比", text: "对手固定为 Griffin 方程、RecurrentGemma 官方语义和 16 个门分块。" },
  13: { id: "benchmark-course", roman: "IV", title: "完整训练与推理", text: "按照 Griffin 第 4、5 节及附录的实验问题组织 H800 测试。" },
};

const math = (tex, note = "") => `<div class="formula"><div class="math-display" data-katex>${tex}</div>${note ? `<small>${note}</small>` : ""}</div>`;

const steps = items => `<ol class="execution-steps">${items.map((item, index) => `<li><span>${String(index + 1).padStart(2, "0")}</span><div><h4>${item.title}</h4><p>${item.text}</p>${item.check ? `<small><b>检查：</b>${item.check}</small>` : ""}</div></li>`).join("")}</ol>`;

export const lessons = [
  {
    title: "GPU 真正看到的四类数据",
    context: ["输入", "静态参数", "当前控制", "递推状态"],
    lead: "把 SAMU 拆成四类数据后，哪里读显存、哪里能复用、哪里必须逐步更新就清楚了。",
    body: `
      <div class="four-data-grid">
        <div><span>01</span><h4>当前输入</h4><p><b>形状：</b><code>[B,L,D]</code>；<b>精度：</b>BF16。它生成当前写入和两个控制量。B 是批量，L 是序列长度，D 是输入宽度。</p></div>
        <div><span>02</span><h4>静态谱参数</h4><p><b>形状：</b><code>[M]</code>；<b>精度：</b>FP32。每个复数模态有固定的衰减尺度 <i>ν</i> 和基础相位 <i>θ</i>，同一层的所有词元复用它们。</p></div>
        <div><span>03</span><h4>当前控制量</h4><p><b>形状：</b><code>[B,L,2]</code>。每个词元只产生共享的 <i>c</i> 和 <i>d</i>；它们分别调节全部模态的衰减与相位。</p></div>
        <div><span>04</span><h4>递推状态与写入</h4><p><b>形状：</b><code>[B,M,2]</code>。复数状态由实部和虚部两个 FP32 标量保存；写入也有实部和虚部。M 个复数模态等于 2M 个实状态量。</p></div>
      </div>
      ${math(String.raw`\begin{aligned}
      \text{共同输入分支:}\quad &[B,L,D]\,[D,2M]\longrightarrow[B,L,2M],\\
      \text{SAMU 控制归约:}\quad &[B,L,2M]\longrightarrow[B,L,2],\\
      \text{递推状态:}\quad &[B,M,2]_{\mathrm{FP32}}.
      \end{aligned}`, "完整模型不再增加单独写入矩阵：共同输入分支的 2M 个输出直接拆成复数写入；随后用两个归一化方向生成共享控制。")}
      ${steps([
        {title: "从 HBM 读取词元表示", text: "每层先读取形状为 [B,L,D] 的 BF16 输入。这个张量既进入递推分支，也进入保留连接，因此属于双方共同成本。", check: "SAMU 与 RG-LRU 的 B、L、D、精度和输入内容必须完全相同。"},
        {title: "一次矩阵乘法生成递推块输入", text: "Tensor Core 执行 linear_x，将 D 维输入变成 D_RNN 维。SAMU 把前后两半解释为复数写入；RG-LRU 把同一结果作为门控后的候选写入。", check: "不能给 SAMU 额外增加 D_RNN×D_RNN 写入矩阵，也不能从 RG-LRU 计时中删掉门矩阵。"},
        {title: "因果卷积加入局部历史", text: "每个通道独立执行宽度 4 的深度卷积。预填充时一次处理整段序列，解码时只读取每层最近 3 个 BF16 值。", check: "卷积权重、宽度、缓存精度和启动边界在双方一致。"},
        {title: "递推单元更新 FP32 状态", text: "SAMU 用两个共享控制重建每个复数模态的旋转与衰减；RG-LRU 用两组 16 块对角门生成逐通道保留率和输入门。", check: "M 个复数模态按 2M 个实状态量计，双方状态字节相等。"},
        {title: "写出 BF16 序列输出与 FP32 最终状态", text: "逐词元输出供后续线性层使用，最后状态进入解码缓存。训练反向从这些输出和保存的状态逆序计算梯度。", check: "最终状态不能先截断到 BF16 再写入缓存。"},
      ])}
      <div class="terminology"><strong>本报告会反复出现的硬件词</strong><dl>
        <div><dt>HBM（显存）</dt><dd>GPU 的大容量存储。容量大，但每次读写的距离和能耗高于片上存储。</dd></div>
        <div><dt>SRAM（片上静态存储）</dt><dd>包括共享内存和缓存等片上存储，容量较小，但访问更快。Griffin 在 TPU 上对应使用 VMEM。</dd></div>
        <div><dt>寄存器</dt><dd>单个 GPU 线程直接使用的最快存储。数量有限；使用过多会减少同一计算单元同时驻留的线程。</dd></div>
        <div><dt>内核启动</dt><dd>CPU 向 GPU 提交一次计算任务。任务很小时，提交本身的固定时间可能占主要部分。</dd></div>
        <div><dt>Tensor Core</dt><dd>NVIDIA GPU 中专门执行 BF16 等低精度矩阵乘法的计算单元。</dd></div>
        <div><dt>特殊函数</dt><dd>指数、正弦、余弦、平方根等运算；它们通常比普通加法和乘法昂贵。</dd></div>
      </dl></div>`,
  },
  {
    title: "SAMU 的单步方程",
    context: ["衰减", "旋转", "新写入"],
    lead: "一个时间步只做三件事：保留一部分旧状态、旋转旧状态、加入当前输入产生的新写入。",
    body: `
      <p>第 <i>j</i> 个复数模态的静态参数为 <i>ν<sub>j</sub></i> 和 <i>θ<sub>j</sub></i>。当前词元产生共享控制 <i>c<sub>t</sub></i> 和 <i>d<sub>t</sub></i>，再与静态参数组合成该模态的保留率和相位。</p>
      ${math(String.raw`\begin{aligned}
      \rho_{j,t} &= \exp\!\left(-\nu_j\exp(c_t)\right),\\
      \phi_{j,t} &= \theta_j+d_t,\\
      z_{j,t+1} &= \rho_{j,t}\exp(i\phi_{j,t})z_{j,t}+w_{j,t}.
      \end{aligned}`, "ρ 决定旧状态保留多少，φ 决定二维平面旋转多少，w 是当前词元的复数写入。")}
      <p>GPU 不直接计算复数乘法，而是把状态写成实部 <i>x</i> 和虚部 <i>y</i>：</p>
      ${math(String.raw`\begin{aligned}
      x' &= \rho\left(x\cos\phi-y\sin\phi\right)+w_{\mathrm R},\\
      y' &= \rho\left(x\sin\phi+y\cos\phi\right)+w_{\mathrm I}.
      \end{aligned}`)}
      ${steps([
        {title: "生成两个共享坐标", text: "把卷积输出分别投影到两个归一化方向，再经过有界非线性，得到当前词元的径向控制 c_t 与相位控制 d_t。当前实现用一次两列投影完成。", check: "一次两列投影必须逐值等价于原来的两次独立点积。"},
        {title: "按模态重建转移", text: "每个模态读取自己的静态 ν_j、θ_j，与共享坐标组合为保留率 ρ_j,t 和相位 φ_j,t。共享的是动态坐标，不是最终转移。", check: "所有指数、正弦和余弦使用原方程；主结果不使用近似特殊函数。"},
        {title: "执行二维旋转与衰减", text: "负责一个模态的线程读取旧实部和虚部，在 FP32 中完成旋转、缩放，并加入当前复数写入。", check: "实部、虚部交叉项的符号必须与复数乘法一致。"},
        {title: "输出与缓存", text: "完整序列输出转成 BF16 供下一层使用；最后一个实部、虚部保持 FP32，作为下一词元的状态缓存。", check: "对照显式 PyTorch 递推的逐元素输出、最终状态和输入梯度。"},
      ])}
      <p>这对实数状态形成一个受约束的 2×2 旋转—缩放块。相同的两个控制量作用于所有模态，但每个模态因 <i>ν<sub>j</sub></i>、<i>θ<sub>j</sub></i> 不同而产生不同响应。</p>`,
  },
  {
    title: "共享控制如何形成不同模态响应",
    context: ["两个共享控制", "多组静态谱参数", "相干选择性"],
    lead: "共享不等于所有模态做同一件事：c、d 相同，ν、θ 不同，因此每个模态得到不同的保留率和相位。",
    body: `
      <p><i>c<sub>t</sub></i> 同时改变所有模态的时间尺度，<i>d<sub>t</sub></i> 同时平移所有模态的相位。这种“共同移动、各自响应”的约束就是本报告所说的相干选择性。</p>
      <div class="breakout coherent-breakout" data-figure="coherent"><div class="figure-head"><div><h3>两个共享控制如何作用于十二个模态</h3></div></div><p class="figure-intro">左图显示 c 改变不同 ν 下的保留率；右图显示 d 对所有基础相位施加相同角度偏移。灰色是静态基准，彩色是当前结果。</p><div class="control-row"><label>衰减控制 c<input type="range" min="-0.8" max="0.8" value="0.25" step="0.01" data-coherent-c></label><label>相位控制 d<input type="range" min="-0.8" max="0.8" value="0.35" step="0.01" data-coherent-d></label></div><canvas data-coherent-canvas width="1100" height="500" aria-label="共享控制对十二个模态的影响"></canvas><div class="coherent-readout" data-coherent-readout></div></div>
      <p>这里的优势是“动态转移的描述只有两个数”，不是“整个递推只做两个运算”。内核仍要为 M 个模态重建保留率和旋转，并更新 2M 个实状态量。</p>`,
  },
  {
    title: "借鉴 RG‑LRU：短序列把状态留在片上",
    context: ["显存带宽", "片上状态", "顺序扫描"],
    lead: "短序列的主要问题不是乘法太多，而是每一步都把状态写回 HBM 会产生大量数据移动。",
    body: `
      <p>Griffin 第 4.2 节指出，RG‑LRU 的逐通道状态更新计算量很小，却需要反复读写状态，因此受显存带宽限制。官方实现为 TPU 编写 Pallas 顺序扫描，把隐藏状态保存在 VMEM 中，并按较大的数据块搬运。我们的 Triton 顺序内核采用同一系统原则：一个 GPU 内核沿时间循环，让当前状态尽可能停留在寄存器或片上存储中，序列结束后再写回 HBM。</p>
      ${steps([
        {title: "划分状态通道", text: "CUDA 网格的一个程序实例负责一个批次和一段连续状态通道。通道连续可以合并显存读取，避免大量离散访问。", check: "尾部不足一个分块的通道使用掩码，不能越界。"},
        {title: "把初始状态装入寄存器", text: "程序开始时只读取一次 FP32 初始状态，随后沿时间循环时持续复用。", check: "首词元若是文档起点，RG-LRU 必须按官方 segment_pos=0 语义切断旧状态。"},
        {title: "沿时间顺序递推", text: "每次读取当前转移和写入，更新寄存器中的状态，再把当前输出写到连续输出张量。", check: "前向扫描必须与显式 for 循环相同，不能交换词元顺序。"},
        {title: "写回最后状态", text: "全部 L 个词元处理完成后，再把寄存器状态写入 FP32 缓存。这样每一步不需要从 HBM 重新载入旧状态。", check: "分别测量逻辑状态流量与完整内核延迟，不能把逻辑字节当作性能计数器结果。"},
      ])}
      <div class="dataflow-diagram serial-flow" aria-label="顺序扫描数据移动">
        <div class="flow-node memory"><b>HBM</b><span>输入与初始状态</span></div><i>读取一次较大数据块</i><div class="flow-node compute"><b>GPU 线程块</b><span>状态保留在寄存器/片上存储</span></div><i>沿时间顺序更新</i><div class="flow-node memory"><b>HBM</b><span>输出与最终状态</span></div>
      </div>
      ${math(String.raw`h_t=a_t\odot h_{t-1}+b_t`, "同一状态通道的第 t 步依赖第 t−1 步，所以线程可以长期持有 h；不同批量和不同状态通道仍可并行。")}
      <p>这一思想不是 SAMU 发明的。官方 RG‑LRU 已经在 TPU 上使用优化过的 Pallas 顺序扫描；本项目做的是 NVIDIA GPU 上的 Triton 对应实现，并给 SAMU 使用同等级路径。</p>`,
  },
  {
    title: "长序列问题：顺序依赖让 GPU 空闲",
    context: ["时间依赖", "并行度不足", "结合律"],
    lead: "序列很长时，一个线程沿时间逐步循环会拉长关键路径；这时需要把时间轴拆成多个可以同时工作的分块。",
    body: `
      <p>每个时间步都可表示为仿射变换。两个相邻变换可以先组合，再作用到状态；组合满足结合律，因此多个分块可以同时生成摘要，然后计算每个分块的入口状态。</p>
      ${math(String.raw`\begin{aligned}
      T_t(z)&=A_tz+w_t,\\
      (A_2,w_2)\circ(A_1,w_1)&=\left(A_2A_1,\;A_2w_1+w_2\right),\\
      \bigl(T_3\circ T_2\bigr)\circ T_1&=T_3\circ\bigl(T_2\circ T_1\bigr).
      \end{aligned}`, "结合律允许改变计算分组，但不允许交换词元顺序。")}
      <p>当前 Triton 长序列路径分三步：第一步并行计算每个分块的摘要；第二步按顺序计算各分块的入口状态；第三步在每个分块内重新执行局部递推并写出所有词元的状态。第二步仍有顺序依赖，但它只处理“分块数”，不再处理全部词元。</p>
      ${steps([
        {title: "块内摘要", text: "把时间轴切成 C=16 或 32 个词元的块。所有块并行计算‘入口状态乘什么、再加什么’，得到每块的一对仿射摘要。", check: "随机输入下，摘要作用于任意入口状态时必须等于该块逐步递推。"},
        {title: "块间前缀", text: "按原始时间顺序组合块摘要，得到每块开始前的真实入口状态。当前两级实现只顺序处理块数 K。", check: "组合满足结合律但不满足交换律，块顺序不得改变。"},
        {title: "块内局部回放", text: "每块拿到入口状态后独立重放 C 个词元，写出所有逐词元输出。大量块可以同时占用不同计算单元。", check: "回放输出与完整顺序扫描逐值比较。"},
        {title: "逆序反向三阶段", text: "反向先概括每块对伴随状态的作用，再从末块向前求入口伴随量，最后各块并行重放反向递推并累积参数梯度。", check: "FP32 与 BF16 两种精度分别对照 PyTorch 自动微分。"},
      ])}
      ${math(String.raw`L=K\,C\qquad\Longrightarrow\qquad\text{顺序深度从 }L\text{ 降为约 }C+K`, "L 是序列长度，C 是每块词元数，K 是分块数。这里描述的是当前摘要—边界—回放实现，不把它误写成理想的对数深度。")}`,
  },
  {
    title: "图 3：两条执行路径的数据移动",
    context: ["矩阵投影", "顺序扫描", "分块扫描"],
    lead: "这张图只画实际发生的数据读写，避免把 HBM、缓存、寄存器和计算单元混在同一层。",
    body: `
      <div class="memory-map" aria-label="SAMU 预填充的数据移动">
        <div class="memory-map-row"><b>阶段一：共同输入分支</b><span class="flow-node memory">HBM：块输入与权重</span><em>→</em><span class="flow-node tensor">Tensor Core：BF16 矩阵乘</span><em>→</em><span class="flow-node compute">因果深度卷积</span></div>
        <div class="memory-map-row"><b>短序列：控制—转移—扫描融合</b><span class="flow-node memory">HBM：卷积输出</span><em>→</em><span class="flow-node compute">两个共享控制量</span><em>→</em><span class="flow-node compute">寄存器：重建转移并更新 FP32 状态</span><em>→</em><span class="flow-node memory">HBM：序列输出与最终状态</span></div>
        <div class="memory-map-row"><b>长序列：分块扫描</b><span class="flow-node memory">HBM：卷积输出</span><em>→</em><span class="flow-node compute">并行生成分块摘要</span><em>→</em><span class="flow-node memory">HBM：摘要与分块入口</span><em>→</em><span class="flow-node compute">并行局部回放</span><em>→</em><span class="flow-node memory">HBM：序列输出</span></div>
      </div>
      <div class="term-note"><b>为什么分块不总是更快</b><p>顺序扫描只需一次递推内核；分块扫描要额外保存摘要和分块入口，并启动三个递推内核。长序列获得更多并行度，短序列却可能被额外读写和启动时间抵消，所以切换点必须在目标 GPU 上实测。</p></div>`,
  },
  {
    title: "预填充和单步解码需要不同内核",
    context: ["预填充", "单步解码", "计时边界"],
    lead: "预填充一次处理整段输入，单步解码每次只处理一个新词元；两者不能用同一条性能结论。",
    body: `
      <p><b>预填充</b>是根据完整提示序列建立状态缓存。它有较大的 B×L 矩阵乘法，也有较长时间轴，适合 Tensor Core 投影和分块扫描。<b>单步解码</b>每次只有 L=1，时间轴不能并行；主要成本变成权重读取、状态读写、特殊函数和内核启动。</p>
      ${math(String.raw`\text{预填充}=\text{一次 BF16 投影}+\text{递推扫描}`)}
      ${math(String.raw`\text{单步解码}=\text{当前词元投影}+\text{一次状态更新}`)}
      <p>“打包投影”表示把写入和两个控制器的输出列拼成一次矩阵乘法；“分块摘要”表示用一对仿射参数概括一段词元；“分块入口”表示该段开始前的真实状态；“局部回放”表示拿到入口状态后在该段内重新递推并写出所有结果。这些词在页面中不再以英文缩写代替解释。</p>`,
  },
  {
    title: "RG‑LRU 的官方方程和实现边界",
    context: ["两组门", "16 个分块", "分段重置"],
    lead: "RG‑LRU 对手直接从固定版本的 RecurrentGemma 创建，并逐值对照官方 PyTorch 输出。",
    body: `
      <p>官方 RG‑LRU 先用两组块对角矩阵生成输入门和递推门。论文在模型并行部分使用 16 个门分块；本报告主对比也固定为 16。每个实状态通道都有自己的两个动态门。</p>
      ${math(String.raw`\begin{aligned}
      i_t &= \sigma(W_xx_t+b_x),\\
      r_t &= \sigma(W_ax_t+b_a),\\
      \log a_t &= -8\,r_t\odot\operatorname{softplus}(a_{\mathrm{param}}),\\
      h_t &= a_t\odot h_{t-1}+\sqrt{1-a_t^2}\odot(i_t\odot x_t).
      \end{aligned}`, "当 segment_pos=0 时，官方语义令 a=0，并把输入乘子改为 1。Triton 正确性测试同时覆盖重置和普通位置。")}
      <p>官方 JAX 代码已经包含为 TPU 编写的 Pallas 顺序扫描。因此，“把状态留在片上”不是我们对 RG‑LRU 的新发明；Triton 串行、分块和融合解码是 NVIDIA GPU 实现工作。公平比较要求双方都获得合理实现，但不能把通用扫描或融合重新命名为 SAMU 的算法贡献。</p>`,
  },
  {
    title: "SAMU 可以利用的独特结构",
    context: ["共享转移坐标", "复数配对模态", "直接写入"],
    lead: "SAMU 真正适合 GPU 的地方，是用两个共享坐标控制全部模态，并让模型已有的输入分支直接提供复数写入。",
    body: `
      <div class="advantage-list">
        <div><b>两个共享转移坐标</b><p>RG‑LRU 为每个实状态通道生成输入门和递推门；SAMU 每个词元只生成 <i>c</i>、<i>d</i>，再与静态 <i>ν</i>、<i>θ</i> 重建全部模态的转移。控制器输出从随状态宽度增长变为两个标量。</p></div>
        <div><b>复数配对模态</b><p>两个实状态通道绑定为一个旋转—缩放块。每个模态只存一个半径尺度和一个基础相位，运行时用四次乘加完成二维更新；状态容量仍按两个实数计算，不靠少算缓存获得优势。</p></div>
        <div><b>共享量只计算一次</b><p>当前词元的径向公共因子，以及相位偏移的正弦和余弦，对所有模态都相同。内核先计算一次，再广播给负责不同模态的线程；若每个模态重复计算，就浪费了共享控制带来的结构约束。</p></div>
        <div><b>模型已有输入分支可直接写入</b><p>Griffin 的递推块本来就有 <code>linear_x</code>。把它的 <code>D_RNN</code> 个输出拆为实部和虚部，便得到 SAMU 写入；这样不再追加一张 <code>D_RNN×D_RNN</code> 稠密矩阵。</p></div>
      </div>
      ${math(String.raw`\begin{aligned}
      [u_t^{\mathrm R},u_t^{\mathrm I}]&=\operatorname{linear\_x}(x_t),\\
      w_{j,t}&=\gamma_j\left(u_{j,t}^{\mathrm R}+i u_{j,t}^{\mathrm I}\right).
      \end{aligned}`, "训练模型采用这一定义：外层结构不变，linear_x 同时承担输入变换和 SAMU 复数写入。")}`,
  },
  {
    title: "消除隔离微基准中的八倍投影差距",
    context: ["完整递推块", "参数来源", "公平口径"],
    lead: "旧表中 656 万对 81.9 万参数比较的是两个不同边界：SAMU 计入了完整写入矩阵，RG‑LRU 却只计入门矩阵，因此不适合解释完整模型。",
    body: `
      <p>在隔离单层微基准里，SAMU 从一个 <code>D_RNN</code> 宽向量再次投影到 <code>2M+2</code> 个值；RG‑LRU 只列两张 16 分块门矩阵。取 <code>D_RNN=2560</code>、<code>M=1280</code> 时，前者约有 656 万权重，后者约有 81.9 万权重，正好相差约八倍。这个数字解释了旧微基准的大批量瓶颈，却不是完整递推块的总参数对比。</p>
      <p>完整 Griffin 递推块的双方都具有三张共同矩阵：输入分支 <code>linear_x</code>、门控分支 <code>linear_y</code> 和输出矩阵 <code>linear_out</code>。新的小模型训练让 SAMU 直接使用 <code>linear_x</code> 的输出作为写入，只额外添加两个归一化控制方向和每模态的静态谱参数；RG‑LRU 则在 <code>linear_x</code> 之后添加两张块对角门矩阵。</p>
      ${math(String.raw`\begin{aligned}
      N_{\mathrm{shared}}&=2D D_{\mathrm{RNN}}+D_{\mathrm{RNN}}D,\\
      N_{\mathrm{RG\!-​LRU}}^{\mathrm{extra}}&=\frac{2D_{\mathrm{RNN}}^2}{16}+O(D_{\mathrm{RNN}}),\\
      N_{\mathrm{SAMU}}^{\mathrm{extra}}&=O(D_{\mathrm{RNN}}).
      \end{aligned}`, "第一行是双方共同的三张矩阵；后两行才是递推方法带来的额外参数。")}
      <div class="term-note"><b>这项修改为什么必须重新训练</b><p>虽然递推方程没有删除，但写入来源从“额外稠密矩阵”改为“复用外层输入分支”，模型参数化发生了变化。只有完成同数据、同预算训练并验证损失，才能证明这种集成没有以质量下降换速度。</p></div>`,
  },
  {
    title: "官方 TPU 路径与本项目 H800 路径的对应关系",
    context: ["官方方程", "自定义反向", "设备映射"],
    lead: "我们迁移的是方程、精度和反向递推，不是把 TPU Pallas 源码逐行改成 Triton。",
    body: `
      <p>RecurrentGemma 官方仓库明确说明：Flax/Pallas 路径为 TPU 优化，PyTorch 路径主要作为参考。官方 Pallas 前向让每个程序持有一块状态并沿时间顺序更新；自定义梯度在反向时间方向执行另一条线性递推，并用前向输出计算转移系数的梯度。</p>
      ${math(String.raw`\begin{aligned}
      h_t&=a_t h_{t-1}+b_t,\\
      r_t&=g_t+a_{t+1}r_{t+1},\\
      \frac{\partial\mathcal L}{\partial b_t}&=r_t,\qquad
      \frac{\partial\mathcal L}{\partial a_t}=r_t h_{t-1}.
      \end{aligned}`, "g_t 是上游对当前输出的梯度，r_t 是包含未来时间步影响的伴随状态。")}
      <p>H800 上的 Triton 前向和反向使用同一递推，但把状态宽度切成 128 通道左右的线程块，由 CUDA 线程束共同处理。前向状态和反向伴随量都以 FP32 留在寄存器中，序列输出和梯度按 BF16 写回。这个映射已经分别与显式 PyTorch 前向和自动微分逐值对照。</p>
      <div class="term-note"><b>能否称为最优</b><p>不能只凭设计声称全局最优。当前能证明的是：候选块宽、线程束数、顺序和分块路径经过目标形状搜索，且完整轨迹复测稳定。真正的最优性还要继续用寄存器数、溢出、占用率和显存流量解释，并与新的融合候选复测。</p></div>`,
  },
  {
    title: "双方都获得精确分块训练后，差异还剩什么",
    context: ["同等级对手", "精确分块", "动态中间量"],
    lead: "仿射结合律不是 SAMU 独占的能力，因此 RG‑LRU 也必须获得同样的时间分块前向和反向。",
    body: `
      <p>RG‑LRU 的实数递推和 SAMU 的二维旋转—缩放递推都属于仿射变换。两边现在都按“分块摘要、块间入口、块内回放”执行；反向也都先概括每块的伴随变换，再从序列末端向前传播块边界。每一步仍使用原方程，没有缩短序列、跳过梯度或换成近似函数。</p>
      <div class="advantage-list">
        <div><b>RG‑LRU 的分块摘要</b><p>每个实状态通道保存一个乘法系数和一个加法系数。扫描前已经生成形状为 <code>[B,L,D]</code> 的保留系数与写入，因此块内只需实数乘加，计算很轻。</p></div>
        <div><b>SAMU 的分块摘要</b><p>每个复数模态保存二维旋转—缩放的乘法系数和复数加法项。逐词元动态描述只有 <code>[B,L,2]</code> 的径向、相位控制；逐模态转移在块内由静态谱参数重建。</p></div>
        <div><b>共同得到的收益</b><p>当序列增长而批量下降时，时间块为 H800 提供更多并行任务。顺序依赖从全部词元缩短为“块内长度 + 块数”，代价是额外的摘要与边界读写。</p></div>
        <div><b>仍然不同的代价</b><p>RG‑LRU 需要两张块对角门矩阵和随状态宽度展开的动态门；SAMU 的门投影很小，但每个模态要计算指数与旋转。哪一侧更快取决于状态宽度，不能只由公式计数决定。</p></div>
      </div>`,
  },
  {
    title: "SAMU 的 GPU 优势在哪些形状出现",
    context: ["宽度拐点", "共享控制", "实测边界"],
    lead: "共享控制不是让所有形状都变快，而是让控制成本不随递推宽度同步增长。",
    body: `
      <div class="profile-table-wrap"><table class="profile-table"><thead><tr><th>比较项目</th><th>RG‑LRU</th><th>SAMU</th></tr></thead><tbody>
        <tr><td>动态控制形状</td><td>输入门和保留门随实状态宽度展开，为 <code>[B,L,2D]</code>。</td><td>径向和相位控制在所有复数模态间共享，为 <code>[B,L,2]</code>。</td></tr>
        <tr><td>状态容量口径</td><td><i>D</i> 个 FP32 实状态。</td><td><i>D/2</i> 个复数模态，即 <i>D</i> 个 FP32 实标量；状态字节相同。</td></tr>
        <tr><td>短序列路径</td><td>顺序扫描把状态留在寄存器中。</td><td>同样使用顺序扫描；避免分块启动与摘要开销。</td></tr>
        <tr><td>长序列路径</td><td>32 词元精确分块，实数乘加便宜，但门张量随宽度增长。</td><td>16 词元精确分块，在扫描内重建旋转；控制张量不随宽度增长。</td></tr>
        <tr><td>单步解码</td><td>低批量融合两组块对角门与状态更新；高批量把两组门合并为一次分组矩阵乘，使用 Tensor Core。</td><td>在融合、预打包静态谱和不同模态分块之间按批量实测选择。</td></tr>
      </tbody></table></div>
      ${math(String.raw`\begin{aligned}
      \text{RG-LRU 动态门输出}&:\ [B,L,2D_{\mathrm{RNN}}],\\
      \text{SAMU 动态控制}&:\ [B,L,2].
      \end{aligned}`, "这里比较随词元变化、为状态转移服务的动态描述。输入、状态输出等双方都必须处理的数据没有从口径中删除。")}
      <p>H800 完整训练步显示了宽度拐点：536 万参数的小模型中 RG‑LRU 快约 3.1%–3.3%；1543 万参数的中模型中 SAMU 快约 4.3%–5.1%；递推宽度增至 768、完整模型约 4265 万参数时，SAMU 快约 22.5%–25.8%。因此当前证据支持“状态变宽后，共享控制的收益逐渐超过复数运算成本”，不支持“所有规模全面领先”。</p>`,
  },
  {
    title: "训练实验：从扫描内核到完整小模型",
    context: ["自定义反向", "配对训练", "固定每步词元数"],
    lead: "训练侧同时测递推扫描、完整前向与反向、优化器更新和验证损失，不再用前向微基准替代完整训练。",
    body: `
      <div class="coverage-table"><div><b>递推扫描对照</b><p>沿用 Griffin 附录图 8(a) 的 B=8、1024 个实状态量和 L=2K/4K/8K/16K，比较顺序与分块前向。这一轨道隔离数据移动和递推本身。</p></div>
      <div><b>训练反向对照</b><p>双方都使用自定义 CUDA 反向扫描：RG‑LRU 保存实数转移和前向状态；SAMU 只保存两个共享控制序列与前向状态，并在反向中重算复数转移。两边都把伴随状态留在 FP32 寄存器中沿时间倒序更新。</p></div>
      <div><b>enwik8 配对训练</b><p>质量实验使用 6 层纯 Hawk 递推语言模型：宽度 256、递推宽度 384、门分块 16、卷积宽度 4。原始 enwik8 的一个字节就是一个词元；前 9000 万字节训练、中间 500 万验证、最后 500 万测试。</p></div>
      <div><b>质量与随机性</b><p>数据窗口、优化器、学习率日程、梯度裁剪、每步 8192 字节及验证窗口完全相同。使用 17、29、43 三个随机种子，每个模型训练 3000 步；六条验证曲线都在最后一个检查点继续下降。</p></div></div>
      <p>速度实验分为两层。第一层用小、中、大配置定位递推宽度拐点；第二层直接使用 Griffin 表 2 的 400M 与 1.3B Hawk 配置和 32K 词表。两层都固定每步 8192 个词元，序列长度取 2K、4K、8K，批量相应为 4、2、1；计时包含完整模型前向、交叉熵、反向传播、梯度裁剪和 AdamW 更新。</p>
      ${steps([
        {title: "固定训练负载", text: "每个速度点都处理 8192 个词元。L 从 2K 增到 8K 时，B 从 4 降到 1，避免把更多训练数据误当成序列长度成本。", check: "记录每个点的 B、L、总词元和模型参数量。"},
        {title: "完成一次真实优化器更新", text: "依次执行嵌入、全部 Hawk 递推块、32K 词表投影、交叉熵、反向传播、梯度裁剪和 AdamW 更新。训练驻留 FP32 主参数与 Adam 状态，矩阵和激活进入 BF16 自动混合精度。", check: "优化器确实修改参数，峰值显存在预热后重新清零统计。"},
        {title: "平衡测量顺序", text: "一轮按 SAMU→RG-LRU 与短→长执行，另一轮反向执行；每个点先编译和预热，再保留七个完整步骤样本。", check: "最终耗时取两轮中位数的平均，原始样本全部写入 JSON。"},
        {title: "独立验证训练质量", text: "三个随机种子从头训练；定期计算固定验证集 BPC，用验证集最低点选择检查点，再且仅再测试一次。", check: "若验证曲线反弹，必须报告过拟合并更换合理数据规模，不能裁图伪装收敛。"},
      ])}
      ${math(String.raw`B\times L=8192\qquad (B,L)\in\{(4,2048),(2,4096),(1,8192)\}`, "这样长度变化不会同时改变每步训练词元数。")}`,
  },
  {
    title: "推理实验：1.3B 配置完整模型连续生成",
    context: ["B=16 延迟", "空提示与 4K 提示", "CUDA 图"],
    lead: "推理主结果采用 Griffin 表 2 的 1.3B 系统形状，包括 24 个完整 Hawk 递推块、32K 词表输出和下一个词元选择。随机权重只用于速度，质量由 enwik8 训练轨道单独回答。",
    body: `
      <p>延迟轨道固定 B=16，从空状态以及 4096 字符提示建立的状态开始，连续生成 128、256、512、1024、2048、4096 步。提示处理时间不计入生成延迟，但后续每一步都使用提示产生的真实卷积缓存和递推状态。</p>
      <p>吞吐轨道先在共同的 B=1、4、16、32、64、96、128、192、256、384、512 候选上运行 64 步探针，再让每轮探针吞吐最高的两个批量完成 512、1024、2048、4096 步完整轨迹。每个批量预先捕获一张 CUDA 图；图内执行整模型并把预测词元和每层缓存写回静态缓冲区，因此测到的不是孤立递推函数。</p>
      ${steps([
        {title: "建立真实起始缓存", text: "空提示使用全零状态；4K 提示先执行完整预填充，生成每层卷积缓存和 FP32 递推状态。预填充时间与后续生成时间分开。", check: "4K 轨道的 segment_pos 从 4096 开始，不能错误触发首词元重置。"},
        {title: "捕获完整单步 CUDA 图", text: "图内包含词嵌入、24 个递推块中的归一化、输入投影、卷积、递推、输出投影和前馈网络，以及最终归一化、32K 词表投影、取最大概率词元和缓存回写。", check: "CUDA 图只消除 Python 提交噪声，不允许删掉模型算子。"},
        {title: "测量连续轨迹延迟", text: "固定 B=16，连续执行 128–4096 次图重放；每个长度从同一类起始缓存重置，记录三条完整轨迹。", check: "报告整条轨迹中位数，而不是用一次单步短探针乘长度。"},
        {title: "搜索最高吞吐", text: "先在共同的十一个批量候选上运行 64 步探针，再让每轮最高的两个候选完成 512、1K、2K 与 4K 轨迹；最终值只从完成整条轨迹的候选中选择。", check: "双方候选集合相同，并同时报告获胜批量、显存上限和低批量延迟。"},
      ])}
      ${math(String.raw`t_{\mathrm{step}}\approx\frac{\text{参数字节数}+B\times\text{每个序列的缓存字节数}}{\text{有效显存带宽}}`, "这是 Griffin 公式 (5)。1.3B 配置中每层双方都缓存 2560 个 FP32 实状态量，并额外保存宽度 4 卷积所需的三个 BF16 历史值。")}
      <p>上式用于解释低批量解码的主要瓶颈。双方公共矩阵、卷积、前馈网络和词表投影完全相同；差异只来自 RG‑LRU 的两张 16 组块对角门矩阵，以及 SAMU 的两个共享控制方向和复数转移。</p>`,
  },
  {
    title: "如何证明 SAMU 方法真正有效",
    context: ["速度", "质量", "端到端"],
    lead: "最终论文结论必须同时经过方程正确性、硬件速度和训练质量三道门槛。",
    body: `
      <ol class="research-checklist">
        <li><b>方程正确：</b>标准 SAMU 对照自己的 PyTorch 参考；RG‑LRU 对照固定提交的官方 RecurrentGemma，覆盖普通位置和分段重置。</li>
        <li><b>内核公平：</b>相同真实状态字节数、精度、批量、序列长度和计时边界；双方分别调优，但不能改变模型参数化。</li>
        <li><b>系统完整：</b>单卡训练实验必须包含前向传播、反向传播和优化器；只有在声称多卡收益时才必须进一步计入设备间通信。推理实验必须包含真实模型全部层、连续采样和内存上限。</li>
        <li><b>质量不下降：</b>在相同数据、词元数、优化器和调参预算下训练 SAMU 与 RG‑LRU，比较验证损失及下游任务。只有速度和质量同时成立，才能证明方法有效。</li>
      </ol>
      <p>enwik8 三随机种子训练中，SAMU 与 RG‑LRU 的六条验证曲线都持续下降；测试集结果会如实报告均值与标准差，不能只展示速度。系统侧同时报告递推内核、小中大模型、400M/1.3B 完整优化器步骤和 1.3B 完整生成轨迹。只有这些层次的结果方向一致时，才把共享控制写成端到端优势。</p>`,
  },
];
