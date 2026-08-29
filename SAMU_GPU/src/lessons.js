export const parts = {
  0: { id: "learn", roman: "I", title: "研究对象与方程", text: "先把一次递推拆成旧状态、状态转移和新写入，并标明每个量的形状、数据类型与生命周期。" },
  5: { id: "gpu-explorer", roman: "II", title: "GPU 执行路径", text: "沿数据路径说明显存、缓存、线程束、寄存器、线程块、分块与扫描；动画只表达实际运算关系。" },
  20: { id: "compare", roman: "III", title: "SAMU 与 RG‑LRU 的结构对照", text: "先按官方方程构造同等级 Triton RG‑LRU，再区分数学结构、内核资源和实际延迟。" },
  22: { id: "benchmark-course", roman: "IV", title: "实验方法与结论", text: "最后比较实际运行时间。所有主结论同时匹配宽度、参数量、递推状态字节数和实现等级。" }
};

const tags = (...items) => `<div class="evidence-row">${items.map(([c,t]) => `<span class="evidence ${c}">${t}</span>`).join("")}</div>`;
const math = (tex, note="") => `<div class="formula"><div class="math-display" data-katex>${tex}</div>${note?`<small>${note}</small>`:""}</div>`;

export const lessons = [
  {
    title: "从模型结构到可验证的性能结论",
    context: ["三段计算", "写入 → 状态转移 → 新状态", "数学结构不等于内核性能"],
    lead: "本报告从方程、数据布局、GPU 内核和实测结果四个层面，检查 SAMU 的结构优势是否真的转化为速度。",
    body: `${tags(["verified","代码已核对"],["hypothesis","待验证假设"])}
      <p>给定当前 token 的输入 <span class="var">u<sub>t</sub></span>，控制器先产生两个有界标量，再把输入投影成所有模态的复数写入。这里的“模态”是递推状态的一对实部与虚部；“控制器”是从输入生成动态控制量的小型投影。旧状态经过每个模态各自的衰减和旋转，最后加上新写入。计算因此分成三部分：<b>可并行的稠密矩阵乘法</b>、<b>沿时间顺序执行的递推</b>，以及<b>两者之间的数据传递</b>。</p>
      ${math(String.raw`\begin{aligned}u_t&\longrightarrow(c_t,d_t),\\ Bu_t&\longrightarrow w_t,\\ z_{t+1}&=A_tz_t+w_t.\end{aligned}`, "三段在数学上相邻，在 GPU 上却可能属于不同 kernel。")}
      <div class="lesson-grid"><div class="micro-example"><strong>最小例子</strong>4 个复数模态等于 8 个实数状态量。一个 token 只产生共享的 c,d，但每个模态仍有自己的静态参数 ν,θ。</div><div class="meaning"><strong>要验证的结论</strong>共享控制是否减少动态控制开销，以及每个模态仍需执行的指数和旋转会消耗多少计算。</div></div>
      <div class="terminology"><strong>本报告的术语约定</strong><dl><div><dt>kernel（内核）</dt><dd>一次在 GPU 上启动并执行的函数。</dd></div><div><dt>launch（内核启动）</dt><dd>CPU 或运行时向 GPU 提交一次内核的固定开销。</dd></div><div><dt>warp（线程束）</dt><dd>32 个以同一指令节奏执行的 GPU 线程。</dd></div><div><dt>CTA（线程块）</dt><dd>一组可通过共享内存和屏障协作的线程；一个线程块可含多个线程束。</dd></div><div><dt>register（寄存器）</dt><dd>线程私有的片上高速存储；过多会降低可同时运行的线程数。</dd></div><div><dt>occupancy（占用率）</dt><dd>一个 SM 上实际或理论活跃线程束数相对硬件上限的比例。</dd></div><div><dt>HBM / DRAM（显存）</dt><dd>容量大、跨内核持久，但延迟高于片上缓存和寄存器。</dd></div><div><dt>Tensor Core（张量核心）</dt><dd>GPU 中专门加速低精度稠密矩阵乘法的计算单元。</dd></div><div><dt>FMA（融合乘加）</dt><dd>用一条指令完成 a×b+c，是矩阵乘和状态更新的基本运算。</dd></div><div><dt>SFU（特殊函数单元）</dt><dd>执行指数、正弦、余弦等特殊数学函数的硬件路径。</dd></div><div><dt>prefill（预填充）</dt><dd>一次处理完整输入序列并建立递推缓存。</dd></div><div><dt>decode（单步解码）</dt><dd>读取已有缓存，生成一个新时间步并更新缓存。</dd></div></dl></div>
      <p>“适合 GPU”不能只看理论字节数，还要同时检查并行度、内核启动数、寄存器压力、特殊函数和数据复用。后面的主实测只比较 SAMU 与严格按官方方程实现的 RG‑LRU Triton 路径。</p>`
  },
  {
    title: "SAMU 的单步递推方程",
    context: ["复数递推", "ρ = exp(−ν·exp(c))", "φ = θ + d"],
    lead: "先忘掉 chunk 和 scan。一个 timestep 只是：读旧记忆、衰减并旋转、加入新写入。",
    body: `${tags(["verified","代码已核对"],["derived","由方程推导"])}
      <p>对第 j 个模态，SAMU 的状态是复数 <span class="var state">z<sub>j,t</sub>=x+iy</span>。静态参数 <span class="var">ν<sub>j</sub>, θ<sub>j</sub></span> 描述该模态的基础时间尺度与相位；动态标量 c,d 则由当前 token 调整整组静态频谱。</p>
      ${math(String.raw`\boxed{z_{j,t+1}=\underbrace{e^{-\nu_j e^{c_t}}}_{\rho_{j,t}}\,\underbrace{e^{i(\theta_j+d_t)}}_{e^{i\phi_{j,t}}}\,z_{j,t}+w_{j,t}}`, "ρ 控制幅度保留，φ 控制二维平面旋转，w 是当前 token 的复数写入。")}
      <p>把它写成实数运算，一个 GPU 线程需要计算指数函数、正弦与余弦，再执行少量乘加：<span class="var">x′=ρ(x cosφ−y sinφ)+w<sub>R</sub></span>，<span class="var">y′=ρ(x sinφ+y cosφ)+w<sub>I</sub></span>。这正是后面“一个线程处理一个模态”的来源。</p>
      <div class="lesson-grid"><div class="micro-example"><strong>极小例子</strong>若 ρ=.9、φ=π/2、z=1+0i、w=.1+.2i，则旋转后为 0+.9i，更新后是 .1+1.1i。</div><div class="misconception"><strong>常见误解</strong>复数状态不是“免费表达力”。M 个复数模态占 2M 个实数状态量；匹配状态缓存字节数时必须按 2M 计算。</div></div>
      <div class="breakout" data-figure="step"><div class="figure-head"><div><span class="figure-number">图 2 · 分步查看</span><h3>SAMU 单步递推</h3></div><span class="evidence verified">数据路径已核对</span></div><div class="figure-toolbar"><div class="control-row"><label>显示内容<select data-step-view><option value="full">完整更新</option><option value="retention">只看保留率</option><option value="phase">只看相位</option><option value="write">只看写入</option></select></label><label>模态数量<select data-step-modes><option>4</option><option>8</option><option selected>32</option></select></label></div><span data-step-status class="mono">第 1 / 11 步</span></div><canvas data-step-canvas width="1100" height="420" aria-label="单个时间步分步播放器"></canvas><div class="data-badges" data-step-badges></div><div class="transport"><button data-step-action="back">上一步</button><button data-step-action="toggle" class="primary">播放</button><button data-step-action="step">下一步</button><button data-step-action="reset">重置</button><select data-step-speed><option value=".25">0.25×</option><option value=".5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select></div></div>`
  },
  {
    title: "单个时间步的数据与形状",
    context: ["每个 token 一组：u、c、d", "每个模态一组：ν、θ、z、w", "张量形状决定内核的数据约定"],
    lead: "在画 kernel 之前先给每个量贴 shape、dtype、生命周期与变化轴标签。否则“共享”只是模糊形容词。",
    body: `${tags(["verified","代码已核对"])}
      <p><span class="var dynamic">c<sub>t</sub>, d<sub>t</sub></span> 是 token-wise 控制；<span class="var static">ν<sub>j</sub>, θ<sub>j</sub></span> 是 mode-wise 静态 spectrum；<span class="var state">z<sub>j,t</sub></span> 与 <span class="var write">w<sub>j,t</sub></span> 都沿 mode 变化。批处理后，token 量带 B×L 轴，mode 量再带 M 轴。</p>
      ${math(String.raw`\begin{aligned}\operatorname{controller}:&\quad [B,L,d]\longrightarrow[B,L,2],\\ \operatorname{write}:&\quad [B,L,d]\,[d,2M]\longrightarrow[B,L,2M].\end{aligned}`, "canonical PPO 使用 raw observation 作为 controller 输入；formal PyTorch wrapper 可有独立 γ_log，不能混称同一实现。")}
      <p>对 kernel 来说，shape 决定连续内存、向量化和 CTA 分工。c,d 很小，不表示整个 recurrence summary 是 O(1)：每个 token 的 write 仍是 O(M)，每个 chunk 的 zero-state output q 也仍是 O(M)。</p>
      <div class="lesson-grid"><div class="meaning"><strong>GPU 数据</strong>BF16 下一个 complex write 占 4 bytes；B×L×M 的 write tensor 的逻辑体积是 4BLM bytes。</div><div class="misconception"><strong>常见误解</strong>“两个 dynamic scalars”描述控制维度，不自动等于两个 HBM load。fused kernel 可以广播或重算，实际 traffic 必须 profile。</div></div>`
  },
  {
    title: "写入向量可以一次矩阵乘法生成",
    context: ["各时间步可独立投影", "[BL,d] × [d,2M]", "适合张量核心矩阵乘"],
    lead: "write 只依赖当前输入，不依赖上一步状态。把 batch 与 time 合并后，所有 token 可以同时进入矩阵乘。",
    body: `${tags(["verified","代码已核对"],["paper","GPU 基础事实"])}
      <p>将 U reshape 成 <span class="var">[B·L,d]</span>，将复数投影拆成实部与虚部权重，便得到普通 dense GEMM。这里没有时间箭头：第 900 个 token 的 write 不需要等待第 899 个 state。</p>
      ${math(String.raw`[B\!\cdot\!L,d]\,[d,2M]\longrightarrow[B\!\cdot\!L,2M]=[\,w_R\mid w_I\,]`, "矩阵维度足够大时，可由高吞吐 matmul/Tensor Core 路径处理。")}
      <div class="lesson-grid"><div class="micro-example"><strong>极小例子</strong>B=2,L=4,d=3,M=2：一次 [8,3]×[3,4] 产生 8 个 token、2 个 complex modes 的写入。</div><div class="meaning"><strong>实际意义</strong>训练/prefill 可以先批量预计算 write；decode 的 L=1 很小，launch 与融合策略更重要。</div></div>
      <div class="meaning"><strong>并行范围</strong>所有时间步的写入向量可以并行生成，因为 w<sub>t</sub> 只依赖各自的 u<sub>t</sub> 和共享权重。时间依赖从 z<sub>t</sub> 更新到 z<sub>t+1</sub> 时才出现。</div>`
  },
  {
    title: "递推带来的时间依赖",
    context: ["zₜ₊₁ 依赖 zₜ", "总工作量与串行深度", "可结合，但不是矩阵乘"],
    lead: "同一 mode 上，下一步读的是刚刚产生的 state。把所有时间点一起启动，会缺少真实输入。",
    body: `${tags(["verified","代码已核对"],["derived","由方程推导"])}
      <p>recurrence 的 work 是 O(LM)，串行 span 也是 O(L)：每个 mode 可以并行，但每个 mode 内沿时间排成链。短序列时，M 个并行 job 也许够喂饱 GPU；当 M 很小而 L 很长，数十个活跃 lane 无法覆盖 82 个 SM。</p>
      ${math(String.raw`z_0\longrightarrow z_1\longrightarrow z_2\longrightarrow\cdots\longrightarrow z_L`, "把 write 一次 GEMM 出来，并没有移除 state chain。")}
      <p>关联 affine map 提供另一条路：每步是 <span class="var">T<sub>t</sub>(z)=A<sub>t</sub>z+w<sub>t</sub></span>，而 affine map 可结合。结合律允许 tree/prefix scan 把 span 降到 O(log L)，代价是中间 summary、更多 work 和不同的数据移动。</p>
      <div class="misconception"><strong>常见误解</strong>“scan 是并行的”不等于“scan 一定更快”。短 L 或小 batch 下，串行 state-stationary kernel 可能以更少 materialization 胜出；crossover 必须测。</div>`
  },
  {
    title: "HBM 显存、L2 缓存、共享内存与寄存器",
    context: ["容量与延迟的取舍", "公式流量不等于硬件实测", "状态尽量留在片上"],
    lead: "GPU 不是一块均匀的内存。递推是否高效，常常取决于同一个 state 在时间循环里停在哪里。",
    body: `${tags(["paper","CUDA 基础事实"],["derived","逻辑模型"])}
      <p>HBM 容量大、跨 kernel 持久，但访问昂贵；L2 由硬件缓存；shared memory 属于 CTA，适合线程协作；register 是 lane-local 的最快存储。最差的串行实现每一步把 z 写回 HBM、下一步再读。更合理的单 kernel 循环让 lane 把 x,y 留在 register，直到处理完负责的时间段。</p>
      <p>但普通 kernel 结束后，register 状态就消失。不要声称多个独立 launch 能“保持寄存器 state”；跨 launch 必须落到可见内存，除非使用 persistent kernel 等不同执行模型。</p>
      <div class="breakout" data-figure="memory"><div class="figure-head"><div><span class="figure-number">图 3 · 数据移动</span><h3>GPU 存储层级中的数据路径</h3></div><span class="evidence derived">逻辑模型</span></div><div class="figure-toolbar"><div class="control-row"><label>执行路径<select data-memory-track><option value="gemm">写入矩阵乘法</option><option value="serial">串行递推</option><option value="scan">分块扫描</option></select></label></div><div class="transport"><button data-memory-action="toggle" class="primary">播放</button><button data-memory-action="step">推进</button><button data-memory-action="reset">重置</button></div></div><div class="memory-stage" data-memory-stage><div class="memory-level hbm">HBM 显存<br>全局张量</div><div class="memory-level l2">L2<br>硬件缓存</div><div class="memory-level shared">共享内存<br>线程块分片</div><div class="memory-level register">寄存器<br>x,y</div><div class="memory-level tensor">Tensor Core<br>稠密矩阵乘</div><div class="memory-level alu">算术/特殊函数<br>exp · sincos · FMA</div></div><div class="traffic-readout"><div><span>逻辑流量（由张量形状计算）</span><strong data-logical-traffic>—</strong></div><div><span>DRAM 流量硬件实测</span><strong>未测</strong></div></div></div>`
  },
  {
    title: "一个 GPU 线程负责一个模态",
    context: ["lane-local x,y", "exp + sincos + FMA", "静态 ν,θ"],
    lead: "把一个复数模态分配给一个 GPU 线程：线程只保存自己的 ν、θ、x、y、w，并接收共享的 c,d。",
    body: `${tags(["proposed","拟议映射"])}
      <p>lane 先算 g=exp(c)，再算 ρ<sub>j</sub>=exp(−ν<sub>j</sub>g) 与 φ<sub>j</sub>=θ<sub>j</sub>+d。接着用二维旋转更新 x,y。只要 register 压力允许，状态在时间循环中可保持 lane-local。</p>
      ${math(String.raw`\operatorname{lane}_j:\ (\nu_j,\theta_j,x_j,y_j,w_{R,j},w_{I,j})\;\oplus\;\operatorname{shared}(c_t,d_t)`, "新 Triton kernel 已实现这个数据分工：c,d 每个 token/program 读取一次，mode state 以 FP32 累积。")}
      <div class="lesson-grid"><div class="meaning"><strong>已实现收益</strong>transition 不需要先 materialize 成 [B,L,M] gate tensor；kernel 从共享 c,d 与静态 ν,θ 现场重建。</div><div class="misconception"><strong>仍需硬件计数</strong>exp/sincos 会占用特殊函数路径，register 增长也可能压低 occupancy。H800 上已安装 Nsight Compute，但宿主机权限禁止读取计数器，所以硬件 SFU、DRAM 与 achieved occupancy 保持 N/A。</div></div>`
  },
  {
    title: "一个线程束并行处理 32 个模态",
    context: ["一个线程束含 32 个线程", "线程 7 处理模态 7", "同一组 c,d 产生不同响应"],
    lead: "一个线程束的 32 个线程以同一指令节奏执行。SAMU 各模态的公式相同、参数不同，因此可以形成规则的并行映射。",
    body: `${tags(["paper","CUDA 基础事实"],["proposed","拟议映射"])}
      <p>点击任意线程。上方的 c,d 对整个线程束相同；各线程保存的 ν、θ、x、y 与写入值不同，因此算出的 ρ、cosφ、sinφ 和新状态也不同。共享控制没有消除模态差异，只是把动态控制限制在低维坐标中。</p>
      <div class="breakout" data-figure="warp"><div class="figure-head"><div><span class="figure-number">图 4 · 线程映射</span><h3>一个线程束并行处理 32 个模态</h3></div><span class="evidence proposed">拟议内核映射</span></div><div class="control-row"><label>c<sub>t</sub><input data-warp-c type="range" min="-2" max="2" value="0" step=".05"></label><label>d<sub>t</sub><input data-warp-d type="range" min="-3.14" max="3.14" value="0.35" step=".05"></label><span class="mono" data-warp-control>c=0.00 · d=0.35</span></div><div class="broadcast-line"></div><div class="warp-layout"><div class="lane-grid" data-lane-grid></div><aside class="lane-detail" data-lane-detail>选择一个线程</aside></div></div>
      <div class="meaning"><strong>控制量的作用范围</strong><span class="var">c<sub>t</sub></span> 和 <span class="var">d<sub>t</sub></span> 由当前 token 产生，并由全部模态共享；静态参数 <span class="var">ν<sub>j</sub></span> 和 <span class="var">θ<sub>j</sub></span> 再把共享控制转换成每个模态不同的衰减 <span class="var">ρ<sub>j,t</sub></span> 与相位 <span class="var">φ<sub>j,t</sub></span>。</div>`
  },
  {
    title: "共享控制形成一致而不相同的模态响应",
    context: ["相干选择性", "共享动态坐标", "静态模态频谱"],
    lead: "共享不表示 32 个模态行为相同；同一个全局控制经过不同的静态敏感度后，会产生不同响应。",
    body: `${tags(["verified","方程已核对"],["hypothesis","系统假设"])}
      <p>c 通过 g=exp(c) 同时改变所有模态的保留率，但不同 ν 使各模态的变化幅度不同；d 给整个频谱增加相同的相位偏移，而每个模态仍保留不同的基础相位 θ。这就是“相干选择性”：少量共享动态坐标以一致方向调节一个静态谱，但不会让所有模态得到相同结果。</p>
      <div class="breakout coherent-breakout" data-figure="coherent"><div class="figure-head"><div><span class="figure-number">图 5 · 方程驱动</span><h3>共享控制如何产生不同的模态响应</h3></div><span class="evidence verified">按公式计算</span></div><p class="figure-intro">拖动 c 观察不同 ν 对保留率的敏感度；拖动 d 观察整个频谱如何保持间距并整体平移。</p><div class="control-row"><label>衰减控制 c<sub>t</sub><input data-coherent-c type="range" min="-2" max="2" value="0" step=".02"></label><label>相位控制 d<sub>t</sub><input data-coherent-d type="range" min="-3.14" max="3.14" value="0" step=".02"></label></div><canvas data-coherent-canvas width="1100" height="520" aria-label="共享控制量广播到不同模态，并产生不同保留率与相位的图"></canvas><div class="coherent-readout" data-coherent-readout></div><p data-coherent-note class="fairness-inline">这张图只解释 SAMU 方程，不是性能曲线。每个点都由当前 c,d 和该模态的静态 ν,θ 直接计算。</p></div>
      <p>这套约束可能降低 controller 输出维度，也可能限制选择性。最终问题不只是吞吐，还包括质量—效率 Pareto frontier；本站没有重新训练大型任务，因此把质量优势保留为未验证问题。</p>`
  },
  {
    title: "多个线程束组成一个线程块",
    context: ["warp 合作", "shared tile", "barrier 只在 CTA 内"],
    lead: "M 大于 32 时，一个 warp 不够。多个 warp 可覆盖 mode tiles，并在 CTA 内共享 token 控制或 chunk 元数据。",
    body: `${tags(["paper","CUDA 基础事实"],["proposed","拟议映射"])}
      <p>例如 4 个 warp 覆盖 128 modes：每个 warp 处理连续 32 modes，c,d 只需进入每个 warp 的 broadcast 路径。若算法需要 time×mode tile 交换 partial summary，shared memory 是 CTA 内显式协作区。</p>
      ${math(String.raw`\operatorname{CTA}=N_{\mathrm{warp}}\times32\ \text{lanes},\qquad \operatorname{scope}(\mathrm{barrier})=1\ \mathrm{CTA}`, "CTA 之间不能用普通 barrier 同步；全局 prefix 往往需要多阶段 launch 或专门 persistent/cooperative 设计。")}
      <p>更多 warp 不保证更快。H800 上的独立校准表明，SAMU 的长序列 C32 kernel 使用 2 warps 最合适；RG‑LRU 在 L≤2048 使用 4 warps，长序列回到 2 warps。校准数据不进入最终中位数，也不外推到其他 GPU。</p>`
  },
  {
    title: "长序列下的并行度不足",
    context: ["parallel jobs ≈ B×M", "serial depth = L", "82 SM"],
    lead: "串行 recurrence 的总工作很多，却可能只有 B×M 条独立链。工作量大不等于同时可调度的工作多。",
    body: `${tags(["derived","并行度推导模型"])}
      <p>在 B=1,M=64 时，理想化地只有 64 条 mode chains；这张 H800 有 114 个 SM。即便一个 warp 负责多个 mode，时间维仍串行，硬件无法把未来 timestep 提前执行来隐藏当前指令的延迟。</p>
      ${math(String.raw`N_{\mathrm{parallel}}\approx B\!\cdot\!M,\qquad \operatorname{span}_{\mathrm{serial}}=L`, "这只是调度直觉，不是 occupancy profiler 数字。")}
      <p>官方 RecurrentGemma 的 <span class="var">rnn_scan</span> 对 L&gt;1 在 Python 中逐 timestep 循环，FP32 accumulator 则跨步更新。它是可信的公式/source reference，却不是专用 GPU scan kernel；因此长序列的巨大 wall-clock 差距首先是执行路径差距。</p>`
  },
  {
    title: "分块把长递推拆成可组合区间",
    context: ["局部串行", "chunk 独立 summary", "边界 state"],
    lead: "把长度 L 切成 K 个 chunk，每块先在 zero input state 下形成自己的 affine summary，再并行接回真实边界状态。",
    body: `${tags(["derived","由方程推导"])}
      <p>chunk 内仍有顺序，但 K 个 chunk 的参数只依赖各自 token，可以并行生成。之后 scan 求出每个 chunk 的输入 state，再进行 local replay 或 correction。这把一个长度 L 的依赖链变成 chunk_size 的局部深度加 log K 的全局组合。</p>
      ${math(String.raw`L=K\,C_{\mathrm{chunk}}\quad\Longrightarrow\quad \operatorname{span}\approx C_{\mathrm{chunk}}+\log_2K`, "work 与 materialization 会增加；chunking 不是 SAMU 独有，也不是无条件加速。")}
      <div class="micro-example"><strong>极小例子</strong>16 tokens 切成 4×4：先同时总结 [0…3]、[4…7]、[8…11]、[12…15]，再 scan 4 个 summaries。</div>`
  },
  {
    title: "仿射算子的定义与结合律",
    context: ["T(z)=Pz+q", "组合有结合律", "顺序不可交换"],
    lead: "每个 timestep 不是一个 state 数值，而是一个‘给我输入 state，我返回输出 state’的函数。函数可以先组合，再作用。",
    body: `${tags(["derived","代数推导"])}
      <p>令 T₁(z)=P₁z+q₁，T₂(z)=P₂z+q₂。先做 T₁ 再做 T₂，结果仍是 affine map：P 相乘，q 被后一个 P 变换后相加。组合有结合律，所以可以用树；但通常不可交换，token 顺序不能打乱。</p>
      ${math(String.raw`\boxed{(P_2,q_2)\circ(P_1,q_1)=\bigl(P_2P_1,\;P_2q_1+q_2\bigr)}`, "(T₃∘T₂)∘T₁ = T₃∘(T₂∘T₁)，这就是 parallel scan 的代数入口。")}
      <div class="lesson-grid"><div class="micro-example"><strong>极小例子</strong>T₁(z)=2z+1，T₂(z)=3z+4；组合为 6z+7。</div><div class="misconception"><strong>常见误解</strong>“可结合”不是“可交换”。2→3 的 transition 与 3→2 的 q 通常不同。</div></div>`
  },
  {
    title: "分块摘要中的零状态输出 q",
    context: ["q = T(0)", "zero-state output", "O(M)"],
    lead: "q 是一个 segment 在零输入状态下产生的输出。它把 segment 内所有 write 经过后续 transition 的影响累积起来。",
    body: `${tags(["derived","由方程推导"])}
      <p>若 segment summary 是 T(z)=Pz+q，那么代入 z=0 立刻得到 q=T(0)。真实输入 z_in 到来时，不必重做全部代数，只需算 Pz_in+q。q 的尺寸与 state 相同，因为每个 mode 都可能有不同的累积写入。</p>
      ${math(String.raw`q=w_{n}+A_{n}w_{n-1}+A_{n}A_{n-1}w_{n-2}+\cdots`, "q 仍有 M complex entries，即 O(M)；只有 transition P 的描述在 SAMU 中可进一步压缩。")}
      <div class="meaning"><strong>摘要大小</strong>(C,G,D) 只有常数个状态转移标量，但 q 与递推状态同宽，因此完整摘要 (C,G,D,q) 仍然是 O(M)，不能写成常数空间。</div>`
  },
  {
    title: "SAMU 状态转移的 C、G、D 压缩",
    context: ["P_j = exp(−ν_jG)·exp(i(Cθ_j+D))", "3 scalars + q", "exact structure"],
    lead: "SAMU 的每步 transition 共用 c,d，并通过静态 ν,θ 生成 mode 响应。一个 segment 的对角 transition 因此保留同一函数族。",
    body: `${tags(["derived","精确推导"],["proposed","待实现内核"])}
      <p>把 segment 内保留因子相乘：指数里的 −ν<sub>j</sub>exp(c<sub>t</sub>) 相加，因此只需 G=Σ exp(c<sub>t</sub>)。相位相加后得到 C·θ<sub>j</sub>+D；若每步都有一次基础 θ，则 C 是 token count，D=Σd。</p>
      ${math(String.raw`\boxed{P_{j,\mathcal S}=\exp(-\nu_jG_{\mathcal S})\,\exp\!\bigl(i(C_{\mathcal S}\theta_j+D_{\mathcal S})\bigr)}`, "compressed transition = (C,G,D)；完整 affine summary = (C,G,D,q)，其中 q 仍是 mode vector。")}
      <p>这不是近似压缩，而是由 canonical 结构推导的 exact representation。本站 PyTorch prototype 已与直接 recurrence 做数值正确性检查；但高性能 advanced scan custom kernel 尚不存在，所以性能仍是 proposal。</p>`
  },
  {
    title: "前缀扫描连接各个分块",
    context: ["offset 1 → 2 → 4", "exclusive prefix", "chunk inputs"],
    lead: "每个 chunk 先只知道自己的 summary。prefix scan 计算它之前所有 chunks 的组合，于是得到它的真实输入 state。",
    body: `${tags(["derived","由方程推导"])}
      <p>对 8 chunks，Hillis–Steele 风格演示有三轮：offset 1、2、4。每轮只有满足索引条件的节点参与 combine。工程实现可以选择更节省 work 的 scan，但结合律和前缀含义不变。</p>
      <div class="breakout" data-figure="scan"><div class="figure-head"><div><span class="figure-number">图 6 · 分块组合</span><h3>SAMU 分块前缀扫描</h3></div><span class="evidence derived">精确代数</span></div><div class="control-row"><button data-scan-mode="tokens" class="primary">组合 4 个 token</button><button data-scan-mode="compressed">把 P 折叠为 C,G,D</button><button data-scan-mode="prefix">8 个分块的前缀</button></div><div class="scan-track" data-scan-track></div><div class="scan-equation" data-scan-equation></div><div class="transport"><button data-scan-action="back">上一轮</button><button data-scan-action="toggle" class="primary">自动播放</button><button data-scan-action="step">下一轮</button><button data-scan-action="reset">重置</button><select data-scan-speed><option value=".5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select></div></div>
      <p>exclusive prefix 给 chunk k 之前的总 map；把初始 state 作用到这个 map，就得到 chunk k 的 boundary state。之后 chunk 内可以以真实边界 state 生成全部 token states。</p>`
  },
  {
    title: "chunk 内 time scan",
    context: ["intra-chunk prefix", "two levels", "materialization tradeoff"],
    lead: "全局 scan 只给边界；要输出每个 timestep，还要在 chunk 内恢复 prefix，或重放局部 recurrence。",
    body: `${tags(["derived","由方程推导"],["proposed","原型实现"])}
      <p>一种方案先保存每步 affine pair，在 chunk 内做 local scan；另一种方案只保存 chunk summary，边界到达后串行 replay。前者增加临时 tensor，后者增加串行深度。最优选择依赖 chunk size、M、dtype 与是否需要 backward。</p>
      ${math(String.raw`\operatorname{prefill}=\operatorname{scan}(K\ \mathrm{summaries})+\operatorname{replay}(C_{\mathrm{chunk}}\ \mathrm{steps})`, "当前实现是 1 次 packed GEMM + summary / prefix / replay 三个 Triton kernels。")}
      <p>H800 校准显示 crossover 很明确：SAMU 在 L≤256 使用 C8，L=512 使用 C16，更长使用 C32；RG‑LRU 也在长序列使用 C32，但 warp 数的切换点不同。serial kernel 仍保留作研究基线；最终结果只使用预先校准并固定的分派。</p>`
  },
  {
    title: "时间 × 模态：二维并行面",
    context: ["横轴是模态", "纵轴是时间", "线程块形状必须实测"],
    lead: "模态并行沿横向展开，时间扫描沿纵向展开。内核需要把二维平面切成 GPU 可以调度的线程块分片。",
    body: `${tags(["proposed","待测试设计空间"])}
      <p>串行 recurrence 点亮少数 mode columns，并沿时间向下走；time-parallel scan 同时点亮多个 chunk tiles。CTA tile 可以是 32×8、32×16、64×8，但这些只是候选布局。</p>
      <div class="breakout" data-figure="timemode"><div class="figure-head"><div><span class="figure-number">图 7 · 二维分片</span><h3>时间轴与模态轴的并行分片</h3></div><span class="evidence hypothesis">需要实测</span></div><div class="control-row"><button data-tm-kind="serial" class="primary">串行递推</button><button data-tm-kind="scan">时间并行扫描</button><label>线程块分片<select data-tm-tile><option value="32x8">32 个时间步 × 8 个模态</option><option value="32x16">32 × 16</option><option value="64x8">64 × 8</option></select></label></div><div class="time-mode"><canvas id="time-mode-canvas" data-tm-canvas width="850" height="560" aria-label="时间乘模态矩阵"></canvas><aside class="tile-stats" data-tm-stats></aside></div></div>
      <p>旁边的 CTA/warp/job 数是按几何形状计算的 logical schedule，不是 occupancy。编译器给出的 registers/thread、spill 与共享内存会单独展示；由资源上限推导的 occupancy 也不是硬件 achieved occupancy。图中的 tile 仍是设计候选，不等同于 profiler 结论。</p>`
  },
  {
    title: "训练与预填充的执行路径",
    context: ["large L", "dense projections", "chunk + scan"],
    lead: "训练与 prefill 一次看到完整序列：它们可以批量 GEMM，也有机会沿时间做并行 scan。",
    body: `${tags(["verified","Triton 已实现"],["measured","前向已实测"])}
      <p>新 prefill 路径先把 write 的 2M 列与 controller 的 2 列打包为一次 BF16 GEMM，再由 summary、exclusive prefix 与 local replay 以 FP32 state 执行 recurrence。shape dispatch 在计时前解析，双方都计入 1 次 projection 与 3 个 Triton launches。</p>
      ${math(String.raw`\operatorname{prefill}=1\ \mathrm{packed\ GEMM}+\begin{cases}\mathrm{C8},&L\le256,\\ \mathrm{C16},&256<L\le512,\\ \mathrm{C32},&L>512.\end{cases}`, "L≤128 且解析参数界通过时启用 bounded-exp；其他 shape 使用通用 exp。阈值来自 H800 的独立校准，不外推其他 GPU。")}
      <p>该路径是 inference-only；还没有 backward kernel。训练需要为中间状态选择保存、重算或反向 scan，因此网页不再用 forward 数字暗示训练速度。</p>`
  },
  {
    title: "单步解码的执行路径",
    context: ["L = 1", "launch dominated", "persistent state is separate design"],
    lead: "autoregressive decode 每次只有一个新 token。没有长 time axis 可 scan，核心问题转为小批量投影、state load/store 与 kernel launch。",
    body: `${tags(["verified","单次启动 Triton"],["measured","融合微内核已实测"])}
      <p>新 decode kernel 从显存读 token 与 FP32 state，在同一个 launch 内完成 write dot products、两个 controller dot products、有界 c,d、复数 transition 和 state store。没有把 projection 偷移到计时外。</p>
      ${math(String.raw`1\ \mathrm{launch}=\mathrm{write}+\mathrm{control}+\mathrm{transition}+\mathrm{state\ update}`, "H800，B=1,d=128,M=64：SAMU 为 0.008576 ms，RG‑LRU 为 0.006112 ms；这个极小 decode 点由 RG‑LRU 获胜。")}
      <p>SAMU auto-short BF16 output 最大绝对误差为 0.00390625、最终 cache 最大误差为 1.31e-6；相位增量限制在 ±0.077114 rad，degree-6/7 cos/sin 多项式的 float32 最大误差分别为 5.96e-8 与 7.45e-9。</p>
      <div class="misconception"><strong>常见误解</strong>普通多次 kernel invocation 不能让 register state 跨 step 存活。persistent decode 是值得实现的新 backend，但必须单独设计和测量。</div>`
  },
  {
    title: "RG-LRU：独立 gate 与共享坐标的差别",
    context: ["每个通道有动态门", "按官方方程实现的 Triton 内核", "控制维度不等于显存流量"],
    lead: "RG-LRU 为每个通道形成更独立的动态门；SAMU 用 c,d 经过静态频谱产生方向一致、幅度不同的模态响应。前者更自由，后者约束更强。",
    body: `${tags(["verified","官方公式 1–4"],["measured","同等级 Triton 实测"])}
      <p>比较要分两层。数学层：RG-LRU 的 dynamic quantities 随 channel 展开，SAMU 只有少量共享 coordinates。实现层：fused RG-LRU 可以现场生成 gates、复用 cache，未必 materialize O(LM) gate tensor 到 HBM。</p>
      ${math(String.raw`\begin{aligned}i_t&=\sigma(W_xx_t+b_x),\\ r_t&=\sigma(W_ax_t+b_a),\\ \log a_t&=-8\,r_t\odot\operatorname{softplus}(a_{\mathrm{param}}),\\ h_t&=a_t\odot h_{t-1}+\sqrt{1-a_t^2}\odot(i_t\odot x_t).\end{aligned}`, "这是 Griffin / RecurrentGemma 的 RG-LRU 语义；segment_pos=0 时官方实现令 aₜ=0，并把输入 multiplier 置为 1。")}
      <div class="provenance-grid"><div><span class="evidence paper">官方定义</span><strong>方程与语义</strong><p>两组分块对角门、−8·softplus 参数化、平方根归一化、分段重置、BF16 激活与 FP32 缓存。</p></div><div><span class="evidence derived">本项目实现</span><strong>加速实现</strong><p>双门合并为一次打包分块矩阵乘、串行/分块摘要-前缀-回放、单次启动解码和 H800 形状分派。</p></div><div><span class="evidence derived">通用代数</span><strong>不是官方专属</strong><p>仿射结合律和分块扫描是通用方法；我们把它严谨适配到官方 RG‑LRU，而不声称发明扫描算法。</p></div></div>
      <div class="breakout" data-figure="compare"><div class="figure-head"><div><span class="figure-number">图 8 · 同一输入</span><h3>两种递推的动态计算</h3></div><span class="evidence paper">只比较结构</span></div><div class="compare-three two-models" data-compare-cards></div><div class="formula" data-compare-formula>选择模型查看同一 u<sub>t</sub> 经过哪些动态量与状态更新。</div></div>
      <p>本站的同等级对照严格复现官方逐项运算次序。fused decode 的 FP32 dot 会先在 projection 边界舍入到 BF16，再加 BF16 bias，从而与官方 BlockDiagonalLinear 的数值路径一致。固定 commit 的官方 PyTorch 层仍保留为 source reference，但不再承担架构性能对手。</p>`
  },
  {
    title: "从 RG-LRU 到更快的 SAMU 执行路径",
    context: ["packed projection", "on-chip transition", "shape-aware dispatch"],
    lead: "SAMU 的优势不是省掉 state-sized write，而是把动态 transition 的描述压到 c,d，使 projection、重建与递推能形成更紧凑的 kernel 边界。",
    body: `${tags(["verified","已经实现"],["derived","结构到内核的映射"])}
      <p>RG-LRU Triton 每个 token 生成 input gate 与 recurrence gate，再进入 FP32 scan。SAMU 把 2M 维 complex write 与 2 维 selector 合成一次 padded BF16 projection；recurrence kernel 只接收 packed rows、静态 ν/θ/γ 和 FP32 state。</p>
      <div class="lesson-grid"><div class="meaning"><strong>独特可利用点</strong>同一 token 的 c,d 对全部 modes 共享；相位增量与衰减指数都有解析界。bounded-exp 只在 L≤128、pack-time bound check 通过且实测获益时启用，否则回退通用 exp。</div><div class="misconception"><strong>没有消失的成本</strong>write 仍是 2M 维；长序列 chunk 仍需 O(M) 的 q。优化数学函数不等于改变方程。</div></div>
      <div class="misconception"><strong>部署开销仍待优化</strong>当前系统代理把四个控制常数编译进每层的专用 Triton 变体。稳态代码更专用，但 24 层冷启动会产生多份 cubin；JIT 时间不计入延迟，也不能被当作不存在。</div>
      ${math(String.raw`|c_t|\le\frac{|s_c|}{2},\qquad 0\le\nu_j e^{c_t}\le\nu_{\max}e^{|s_c|/2}`, "这两个解析界决定 bounded-exp fast path 是否安全；不是从 benchmark 输入分布拟合出来的阈值。")}
      ${math(String.raw`P_j^{(C)}=\prod_{t=1}^{C}\rho_{j,t}e^{i(\theta_j+d_t)}=\exp\!\left(-\nu_j\sum_{t=1}^{C}e^{c_t}\right)e^{i\left(C\theta_j+\sum_{t=1}^{C}d_t\right)}`, "这是由 SAMU 当前共享控制结构精确导出的可压缩块转移：共享保存 Σexp(cₜ)、Σdₜ 与 C，即可结合静态 νⱼ、θⱼ 重建所有 Pⱼ。当前 compressed-P 原型尚未快过直接 C32，因此不计入已测加速；写入摘要 qⱼ 仍是 O(M)。")}
      <div class="breakout" data-figure="transition"><div class="figure-head"><div><span class="figure-number">图 9 · 动态量对照</span><h3>两种状态转移的表示方式</h3></div><span class="evidence paper">按公式计算</span></div><div class="control-row"><button data-transition-span="token" class="primary">1 个 token</button><button data-transition-span="tokens">32 个 token</button><button data-transition-span="chunk">1 个分块</button></div><div class="transition-view two-models" data-transition-view></div></div>`
  },
  {
    title: "公平测试协议与数据范围",
    context: ["38 个公平实测点", "同宽度、参数量与状态", "只比较 SAMU 与 RG‑LRU"],
    lead: "性能曲线必须先说明实现对象、匹配条件和计时边界，才能支持结构结论。",
    body: `${tags(["measured","38 个实测点"],["verified","保留原始 JSON"])}
      <p id="fairness">公平对照固定 <b>d=128，M=64</b>：SAMU 有 16,772 个参数，RG‑LRU 有 16,768 个参数；双方都使用每个样本 512 字节的 FP32 递推状态、BF16 输入输出，并分别拥有分块预填充和融合单步解码。</p>
      <p>实验覆盖 L=128…65,536、预填充 B=1…64、解码 B=1…64，以及双方 C=8/16/32 三种分块长度，共 38 个公平实测点。</p>
      ${math(String.raw`\operatorname{timing}=\operatorname{median}\!\left(\mathrm{Triton\ driver\ event\ samples}\right),\qquad \mathrm{compile,dispatch,warmup}\notin\mathrm{samples}`, "每个 timed call 前清 L2；保留 p10/p90/p95 与 raw samples。launch、register、spill 来自 profiler/compiler metadata；宿主机权限阻止的硬件 counters 保持 N/A。")}
      <p>下方实验区可切换序列长度、批量大小、单步解码和分块长度，并从每个点追溯到原始计时样本。</p>`
  },
  {
    title: "结论不是口号，而是一张未完成的研究地图",
    context: ["当前实测结论", "下一项内核工作", "尚未回答的问题"],
    lead: "当前证据已经足以淘汰几种过强说法，也足以确定下一步工程优先级。",
    body: `${tags(["measured","已经实测"],["hypothesis","仍待验证"])}
      <p><b>已测结论：</b>H800 双顺序合并结果存在清楚的 crossover。B=1 时，L=128 由 RG‑LRU 快约 1.25×；L=512 与 2048 相差不到 1%；L=8192 与 32768 由 SAMU 快约 1.16×，L=65536 快 1.33×。在 L=512 的批量扫描中，SAMU 到 B=16/64 时扩大为 1.46×/2.18×；但 B=1/4/16/64 的极小 fused decode 均由 RG‑LRU 快约 1.31×–1.40×。</p>
      <p><b>整模型规模结果：</b>放入共同的约 10.7 亿参数 decoder 外壳后，矩阵乘和其余层占据了大部分时间，递推微内核的差距被稀释。双顺序原始样本合并后，B=16、空前缀的六条连续生成轨迹中，SAMU 延迟低 0.8%–1.3%；4K 前缀后低 2.1%–2.7%。在预先限定的 B≤128 完整轨迹中，SAMU 吞吐为 RG‑LRU 的 1.003×–1.013×。这是一项小而一致的系统收益，不写成数量级提升。</p>
      <p><b>资源解释：</b>C32 prefill 两边都是 4 launches；SAMU 的静态 exp/sigmoid 计数为 134/token，RG‑LRU 为 1024/token 且另有 256 sqrt。寄存器、spill 与资源占用上界由 H800 编译器元数据给出；真实 SFU、DRAM 和 achieved occupancy 因宿主机计数权限受限而保持 N/A。</p>
      <div class="lesson-grid"><div class="meaning"><strong>继续验证</strong>H800 已完成主测；下一步需要在有性能计数权限的 Hopper 节点补充 DRAM、L2、SFU 与 achieved occupancy，并实现反向传播和训练质量对照。</div><div class="misconception"><strong>尚未验证的假设</strong>共享控制可能形成更好的质量—效率折中；当前实验没有重新训练模型，也没有测量模型质量。</div></div>
      <p>当前结论的边界很明确：我们验证的是长序列或较大批量 prefill 的优势，不是所有形状都赢；极小 decode 仍需继续优化。训练质量与真实训练好模型的收益也没有被这组随机权重系统实验回答。每条说法都必须能够追溯到公式、代码或测量中的一种。</p>`
  }
];
