export const parts = {
  0: { id: "learn", roman: "I", title: "从方程到 GPU 数据", text: "明确 GPU 实际读取什么、保存什么，以及哪些量随时间或模态变化。" },
  4: { id: "gpu-explorer", roman: "II", title: "针对瓶颈选择执行方法", text: "短序列减少显存往返，长序列利用结合律拆开时间依赖。" },
  8: { id: "compare", roman: "III", title: "与 RG‑LRU 对比", text: "对手固定为 Griffin 方程、RecurrentGemma 官方语义和 16 个门分块。" },
  10: { id: "benchmark-course", roman: "IV", title: "论文实验轴与结论", text: "按照 Griffin 第 4、5 节及附录的实验问题组织 H800 测试。" },
};

const math = (tex, note = "") => `<div class="formula"><div class="math-display" data-katex>${tex}</div>${note ? `<small>${note}</small>` : ""}</div>`;

export const lessons = [
  {
    title: "结论与证据范围",
    context: ["标准 SAMU", "官方 RG‑LRU", "性能与模型质量分开"],
    lead: "这份汇报只回答实现正确性和硬件效率，不把未经训练的结构候选当成方法优势。",
    body: `
      <p>主对比中的 SAMU 固定为单组共享控制的标准方程；RG‑LRU 固定为 RecurrentGemma 官方方程和论文使用的 16 个门分块。16 组 SAMU、直接控制器、低阶函数近似和压缩分块转移都不进入主结论。</p>
      <p>当前能严谨回答两件事：第一，投影完成后，两种递推在论文长序列形状上的运行成本；第二，在相同递推状态字节数下，把各自方程必需的投影也计入后，单层连续解码的真实延迟。没有反向传播内核，就不能把前向结果称为训练速度；没有训练好的同规模模型参数文件，也不能声称模型质量领先。</p>
      <div class="term-note"><b>本页怎样使用“公平”</b><p>同一张 H800、相同批量和序列长度、相同 BF16 输入输出、相同 FP32 状态字节数；双方都包含本方程完成该实验所需的计算。任何改变 SAMU 参数化并需要重训的候选都单独留档，不参与主图。</p></div>`,
  },
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
      \text{输入投影:}\quad &[B,L,D]\,[D,2M+2]\longrightarrow[B,L,2M+2],\\
      \text{递推状态:}\quad &[B,M,2]_{\mathrm{FP32}}.
      \end{aligned}`, "前 2M 列是复数写入，最后 2 列生成共享控制。标准 SAMU 的写入投影是稠密矩阵，不能从成本中删掉。")}
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
      ${math(String.raw`L=K\,C\qquad\Longrightarrow\qquad\text{顺序深度从 }L\text{ 降为约 }C+K`, "L 是序列长度，C 是每块词元数，K 是分块数。这里描述的是当前摘要—边界—回放实现，不把它误写成理想的对数深度。")}`,
  },
  {
    title: "图 3：两条执行路径的数据移动",
    context: ["矩阵投影", "顺序扫描", "分块扫描"],
    lead: "这张图只画实际发生的数据读写，避免把 HBM、缓存、寄存器和计算单元混在同一层。",
    body: `
      <div class="memory-map" aria-label="SAMU 预填充的数据移动">
        <div class="memory-map-row"><b>阶段一：稠密投影</b><span class="flow-node memory">HBM：输入与权重</span><em>→</em><span class="flow-node tensor">Tensor Core：BF16 矩阵乘</span><em>→</em><span class="flow-node memory">HBM：写入与两个控制量</span></div>
        <div class="memory-map-row"><b>短序列：顺序扫描</b><span class="flow-node memory">HBM：投影结果</span><em>→</em><span class="flow-node compute">寄存器：逐步更新 FP32 状态</span><em>→</em><span class="flow-node memory">HBM：序列输出与最终状态</span></div>
        <div class="memory-map-row"><b>长序列：分块扫描</b><span class="flow-node memory">HBM：投影结果</span><em>→</em><span class="flow-node compute">并行生成分块摘要</span><em>→</em><span class="flow-node memory">HBM：摘要与分块入口</span><em>→</em><span class="flow-node compute">并行局部回放</span><em>→</em><span class="flow-node memory">HBM：序列输出</span></div>
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
    context: ["共享转移坐标", "复数配对模态", "稠密写入瓶颈"],
    lead: "SAMU 的硬件机会来自转移结构，而不是把写入成本凭空消失。",
    body: `
      <div class="advantage-list">
        <div><b>两个共享转移坐标</b><p>RG‑LRU 为每个实状态通道生成两个门；SAMU 每个词元只生成 c、d，再与静态 ν、θ 重建全部模态的转移。控制描述从随状态宽度增长变为固定两个标量。</p></div>
        <div><b>复数配对模态</b><p>两个实状态通道被绑定为一个旋转—缩放块。一次相位计算同时决定 2×2 变换中的四个系数关系，不需要为四个矩阵元素分别生成动态参数；代价是每个模态仍要计算指数和正弦余弦。</p></div>
        <div><b>分块转移可由共享统计重建</b><p>一段词元的乘法转移只需 Σexp(c)、Σd 和长度就可与静态谱重建。但是当前压缩转移原型没有快过直接保存每个模态的转移，因此它没有实测收益，也不列为现有优势。</p></div>
        <div><b>当前最大的限制：稠密写入</b><p>标准 SAMU 仍需把输入稠密投影为 2M 个写入值。到 Griffin 的 D_RNN=2560 宽度时，这部分可能比 RG‑LRU 的 16 组块对角门更贵。新的 H800 主测会把这项投影完整计入。</p></div>
      </div>
      ${math(String.raw`P_j^{(C)}=\exp\!\left(-\nu_j\sum_{t=1}^{C}\exp(c_t)\right)\exp\!\left(i\left(C\theta_j+\sum_{t=1}^{C}d_t\right)\right)`, "这是准确的结构恒等式；当前实现没有因它获得速度提升，所以只作为负结果和后续方向。")}`,
  },
  {
    title: "Training Recurrent Models Efficiently on Device：对照项目",
    context: ["附录图 8(a)", "完整训练步骤", "多卡并行"],
    lead: "原论文第 4 节包含三个层次；本报告逐项对应，不用一个前向微基准代替整章。",
    body: `
      <div class="coverage-table"><div><b>论文第 4.1 节：模型并行</b><p>论文讨论 Megatron 风格切分、前向和反向各一次 all-reduce（跨卡求和通信）、卷积按通道独立切分，以及 ZeRO（把优化器和参数分散到多卡）。当前租用节点只有一张 H800，不能实测跨卡通信；该项标记为未覆盖。</p></div>
      <div><b>论文第 4.2 节与附录图 8(a)：扫描内核</b><p>主测严格采用 B=8、1024 个实状态量、L=2K/4K/8K/16K。双方比较顺序 Triton 和分块 Triton；计时从投影完成后开始，只回答递推扫描成本。</p></div>
      <div><b>论文附录图 8(b)：扫描对完整 Hawk 的影响</b><p>论文报告 400M、1B、7B Hawk 的完整训练步骤。当前 SAMU Triton 没有反向内核，也没有标准 SAMU Hawk 模型，因此不能复制这一图。旧的随机权重前向外壳缺少官方 Conv1D、双分支和 D_RNN，已从主证据删除。</p></div>
      <div><b>论文第 4.3 节：完整训练步骤随长度变化</b><p>论文固定每批词元总数，比较 400M、1B、7B、L=2K/4K/8K 的完整训练步骤。要完成对应实验，仍需标准 SAMU 语言模型定义、反向 Triton、训练模型参数、多卡切分和相同优化器；这些不能由前向扫描数字替代。</p></div></div>
      <p>因此，当前“训练侧优势”只能写成“投影后的前向递推扫描优势”，不能写成“完整训练快若干倍”。页面会直接展示已覆盖和未覆盖项目。</p>`,
  },
  {
    title: "Inference Speed：严格对应的测量轴",
    context: ["B=16 延迟", "空提示与 4K 提示", "最佳吞吐"],
    lead: "原论文第 5 节的核心不是单次微内核，而是连续生成延迟、可容纳批量和完整轨迹吞吐。",
    body: `
      <p>H800 对比采用论文 1.3B 配置的递推宽度 D_RNN=2560。延迟轨道固定 B=16，从空提示和 4096 词元提示建立的状态开始，连续执行 128、256、512、1024、2048、4096 步；提示处理时间不计入生成延迟。</p>
      <p>吞吐轨道先在预先固定的 B=1…256 候选上运行短探针，各方法选出前两名，再使用两边候选的并集完成 512、1024、2048、4096 步整条轨迹。这样不会只给某一方测试更有利的批量。</p>
      ${math(String.raw`t_{\mathrm{step}}\approx\frac{\text{参数字节数}+B\times\text{每个序列的缓存字节数}}{\text{有效显存带宽}}`, "这是 Griffin 公式 (5)。在本对比中两边递推状态均为 2560 个 FP32 实数；参数读取和内核实现是主要差异。")}
      <p>本轮结果的范围是“单个递推层，包含各自方程必需的投影和状态更新”。它严格使用论文解码轴，但不是完整 1.3B 模型：仓库里没有训练好的标准 SAMU Griffin 模型，也不能自行发明 Conv1D、双分支、输出投影和层布局后称为官方规模。</p>`,
  },
  {
    title: "如何证明 SAMU 方法真正有效",
    context: ["速度", "质量", "端到端"],
    lead: "最终论文结论必须同时经过方程正确性、硬件速度和训练质量三道门槛。",
    body: `
      <ol class="research-checklist">
        <li><b>方程正确：</b>标准 SAMU 对照自己的 PyTorch 参考；RG‑LRU 对照固定提交的官方 RecurrentGemma，覆盖普通位置和分段重置。</li>
        <li><b>内核公平：</b>相同真实状态字节数、精度、批量、序列长度和计时边界；双方分别调优，但不能改变模型参数化。</li>
        <li><b>系统完整：</b>训练实验必须包含前向传播、反向传播、优化器和多卡通信；推理实验必须包含真实模型全部层、连续采样和内存上限。</li>
        <li><b>质量不下降：</b>在相同数据、词元数、优化器和调参预算下训练 SAMU 与 RG‑LRU，比较验证损失及下游任务。只有速度和质量同时成立，才能证明方法有效。</li>
      </ol>
      <p>当前最值得做的 SAMU 专属优化不是继续缩短两个控制量的计算，而是重新设计“稠密写入投影如何与上游线性层合并”，同时保持标准方程和可训练性。若不能在模型定义中合法合并，扫描微内核的优势不会自动变成完整模型优势。</p>`,
  },
];
