export const parts = {
  0: { id: "learn", roman: "I", title: "Learn the Operator", text: "先把一次递推拆成 old memory、transition 与 new write。每一课只增加一个概念。" },
  5: { id: "gpu-explorer", roman: "II", title: "See It Run on GPU", text: "沿数据路径进入 HBM、warp、register、CTA、chunk 与 scan；图中的运动对应运算状态，而非装饰。" },
  20: { id: "compare", roman: "III", title: "From RG-LRU to SAMU", text: "先按官方方程构造同等级 Triton RG-LRU，再分清数学结构、kernel 资源与实际延迟。" },
  22: { id: "benchmark-course", roman: "IV", title: "Benchmark Lab", text: "最后看 wall-clock。主轨道同时匹配 width、参数量、recurrent-state bytes 与实现等级；Mamba 单列为同宽官方背景线。" }
};

const tags = (...items) => `<div class="evidence-row">${items.map(([c,t]) => `<span class="evidence ${c}">${t}</span>`).join("")}</div>`;
const quiz = (q, a, b, correct, why) => `<div class="quiz" data-correct="${correct}"><p>${q}</p><div class="quiz-options"><button data-answer="a">A · ${a}</button><button data-answer="b">B · ${b}</button></div><p class="quiz-feedback" aria-live="polite">选择后展开原因。</p><template>${why}</template></div>`;
const math = (tex, note="") => `<div class="formula"><div class="math-display" data-katex>${tex}</div>${note?`<small>${note}</small>`:""}</div>`;

export const lessons = [
  {
    title: "五分钟总览：把 hypothesis 变成可测对象",
    context: ["三段计算", "write → transition → state", "数学结构不等于 kernel 性能"],
    lead: "SAMU 不只是一个公式。要判断它是否适合 GPU，必须沿着从 token 到数据布局、从 warp 到 benchmark 的完整链路走一遍。",
    body: `${tags(["verified","Verified code"],["hypothesis","Hypothesis"])}
      <p>给定输入 <span class="var">u<sub>t</sub></span>，canonical 路径先由 controller 产生两个有界标量，再把输入投影为所有 mode 的复数写入。旧状态经过 mode-specific 的保留与旋转，最后加上 write。这个分解把问题切成三块：<b>可并行的 dense projection</b>、<b>带时间依赖的 recurrence</b>、以及<b>两者之间的数据交接</b>。</p>
      ${math(String.raw`\begin{aligned}u_t&\longrightarrow(c_t,d_t),\\ Bu_t&\longrightarrow w_t,\\ z_{t+1}&=A_tz_t+w_t.\end{aligned}`, "三段在数学上相邻，在 GPU 上却可能属于不同 kernel。")}
      <div class="lesson-grid"><div class="micro-example"><strong>极小例子</strong>4 个 complex modes 就是 8 个 real state scalars。一个 token 只提供共享的 c,d，但每个 mode 有自己的 ν,θ。</div><div class="meaning"><strong>实际意义</strong>这给出一个可证伪问题：共享控制能否降低控制开销，而 mode-wise exp/rotation 又会花掉多少算力？</div></div>
      <p>本课程中的“适合 GPU”从不等于“理论 bytes 更少”。它至少同时包含并行度、kernel 数、launch 开销、寄存器压力、数学特殊函数和数据重用。后面的主实测把 SAMU 与严格按官方方程实现的 RG-LRU Triton 路径放在同一等级，官方 RecurrentGemma 与 Mamba-3 只作为独立背景线。</p>`
  },
  {
    title: "SAMU 到底算什么？",
    context: ["复数递推", "ρ = exp(−ν·exp(c))", "φ = θ + d"],
    lead: "先忘掉 chunk 和 scan。一个 timestep 只是：读旧记忆、衰减并旋转、加入新写入。",
    body: `${tags(["verified","Verified code"],["derived","Derived"])}
      <p>对 mode j，SAMU 的状态是复数 <span class="var state">z<sub>j,t</sub>=x+iy</span>。静态参数 <span class="var">ν<sub>j</sub>, θ<sub>j</sub></span> 描述该 mode 的基础时间尺度与相位；动态标量 c,d 则由当前 token 调整整组 spectrum。</p>
      ${math(String.raw`\boxed{z_{j,t+1}=\underbrace{e^{-\nu_j e^{c_t}}}_{\rho_{j,t}}\,\underbrace{e^{i(\theta_j+d_t)}}_{e^{i\phi_{j,t}}}\,z_{j,t}+w_{j,t}}`, "ρ 控制幅度保留，φ 控制二维平面旋转，w 是当前 token 的复数写入。")}
      <p>把它写成实数运算，GPU lane 需要一个 exp、一个 sincos 和少量乘加：<span class="var">x′=ρ(x cosφ−y sinφ)+w<sub>R</sub></span>，<span class="var">y′=ρ(x sinφ+y cosφ)+w<sub>I</sub></span>。这正是后面“一个 lane 处理一个 mode”的来源。</p>
      <div class="lesson-grid"><div class="micro-example"><strong>极小例子</strong>若 ρ=.9、φ=π/2、z=1+0i、w=.1+.2i，则旋转后为 0+.9i，更新后是 .1+1.1i。</div><div class="misconception"><strong>常见误解</strong>complex 不是“免费表达力”。M 个 complex modes 占 2M 个 real scalars，做 state-byte matching 时必须按 2M 计。</div></div>
      <div class="breakout" data-figure="step"><div class="figure-head"><div><span class="figure-number">Interactive 1</span><h3>SAMU Step Explorer</h3></div><span class="evidence verified">Verified dataflow</span></div><div class="figure-toolbar"><div class="control-row"><label>View<select data-step-view><option value="full">完整 update</option><option value="retention">只看 retention</option><option value="phase">只看 phase</option><option value="write">只看 write</option></select></label><label>Mode count<select data-step-modes><option>4</option><option>8</option><option selected>32</option></select></label></div><span data-step-status class="mono">STEP 1 / 11</span></div><canvas data-step-canvas width="1100" height="420" aria-label="单 timestep 分步播放器"></canvas><div class="data-badges" data-step-badges></div><div class="transport"><button data-step-action="back">Previous step</button><button data-step-action="toggle" class="primary">Play</button><button data-step-action="step">Next step</button><button data-step-action="reset">Reset</button><select data-step-speed><option value=".25">0.25×</option><option value=".5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select></div></div>`
  },
  {
    title: "一个 timestep 里有哪些数据？",
    context: ["token-wise: u,c,d", "mode-wise: ν,θ,z,w", "shape 是 kernel 合同"],
    lead: "在画 kernel 之前先给每个量贴 shape、dtype、生命周期与变化轴标签。否则“共享”只是模糊形容词。",
    body: `${tags(["verified","Verified code"])}
      <p><span class="var dynamic">c<sub>t</sub>, d<sub>t</sub></span> 是 token-wise 控制；<span class="var static">ν<sub>j</sub>, θ<sub>j</sub></span> 是 mode-wise 静态 spectrum；<span class="var state">z<sub>j,t</sub></span> 与 <span class="var write">w<sub>j,t</sub></span> 都沿 mode 变化。批处理后，token 量带 B×L 轴，mode 量再带 M 轴。</p>
      ${math(String.raw`\begin{aligned}\operatorname{controller}:&\quad [B,L,d]\longrightarrow[B,L,2],\\ \operatorname{write}:&\quad [B,L,d]\,[d,2M]\longrightarrow[B,L,2M].\end{aligned}`, "canonical PPO 使用 raw observation 作为 controller 输入；formal PyTorch wrapper 可有独立 γ_log，不能混称同一实现。")}
      <p>对 kernel 来说，shape 决定连续内存、向量化和 CTA 分工。c,d 很小，不表示整个 recurrence summary 是 O(1)：每个 token 的 write 仍是 O(M)，每个 chunk 的 zero-state output q 也仍是 O(M)。</p>
      <div class="lesson-grid"><div class="meaning"><strong>GPU 数据</strong>BF16 下一个 complex write 占 4 bytes；B×L×M 的 write tensor 的逻辑体积是 4BLM bytes。</div><div class="misconception"><strong>常见误解</strong>“两个 dynamic scalars”描述控制维度，不自动等于两个 HBM load。fused kernel 可以广播或重算，实际 traffic 必须 profile。</div></div>`
  },
  {
    title: "write 为什么可以一次 GEMM？",
    context: ["所有 timestep 独立", "[BL,d] × [d,2M]", "Tensor Core 友好"],
    lead: "write 只依赖当前输入，不依赖上一步状态。把 batch 与 time 合并后，所有 token 可以同时进入矩阵乘。",
    body: `${tags(["verified","Verified code"],["paper","GPU fact"])}
      <p>将 U reshape 成 <span class="var">[B·L,d]</span>，将复数投影拆成实部与虚部权重，便得到普通 dense GEMM。这里没有时间箭头：第 900 个 token 的 write 不需要等待第 899 个 state。</p>
      ${math(String.raw`[B\!\cdot\!L,d]\,[d,2M]\longrightarrow[B\!\cdot\!L,2M]=[\,w_R\mid w_I\,]`, "矩阵维度足够大时，可由高吞吐 matmul/Tensor Core 路径处理。")}
      <div class="lesson-grid"><div class="micro-example"><strong>极小例子</strong>B=2,L=4,d=3,M=2：一次 [8,3]×[3,4] 产生 8 个 token、2 个 complex modes 的写入。</div><div class="meaning"><strong>实际意义</strong>训练/prefill 可以先批量预计算 write；decode 的 L=1 很小，launch 与融合策略更重要。</div></div>
      ${quiz("write_t 能不能对所有 timestep 并行计算？", "可以", "不可以", "a", "可以。它只依赖各自的 u_t 与共享权重；时间依赖从 z_t → z_{t+1} 才开始。")}`
  },
  {
    title: "为什么 recurrence 是另一种问题？",
    context: ["z_{t+1} depends on z_t", "work/span", "关联而非 GEMM"],
    lead: "同一 mode 上，下一步读的是刚刚产生的 state。把所有时间点一起启动，会缺少真实输入。",
    body: `${tags(["verified","Verified code"],["derived","Derived"])}
      <p>recurrence 的 work 是 O(LM)，串行 span 也是 O(L)：每个 mode 可以并行，但每个 mode 内沿时间排成链。短序列时，M 个并行 job 也许够喂饱 GPU；当 M 很小而 L 很长，数十个活跃 lane 无法覆盖 82 个 SM。</p>
      ${math(String.raw`z_0\longrightarrow z_1\longrightarrow z_2\longrightarrow\cdots\longrightarrow z_L`, "把 write 一次 GEMM 出来，并没有移除 state chain。")}
      <p>关联 affine map 提供另一条路：每步是 <span class="var">T<sub>t</sub>(z)=A<sub>t</sub>z+w<sub>t</sub></span>，而 affine map 可结合。结合律允许 tree/prefix scan 把 span 降到 O(log L)，代价是中间 summary、更多 work 和不同的数据移动。</p>
      <div class="misconception"><strong>常见误解</strong>“scan 是并行的”不等于“scan 一定更快”。短 L 或小 batch 下，串行 state-stationary kernel 可能以更少 materialization 胜出；crossover 必须测。</div>`
  },
  {
    title: "HBM、L2、Shared Memory 与 Register",
    context: ["容量 ↔ 延迟", "logical ≠ measured", "state-stationary"],
    lead: "GPU 不是一块均匀的内存。递推是否高效，常常取决于同一个 state 在时间循环里停在哪里。",
    body: `${tags(["paper","CUDA fact"],["derived","Logical model"])}
      <p>HBM 容量大、跨 kernel 持久，但访问昂贵；L2 由硬件缓存；shared memory 属于 CTA，适合线程协作；register 是 lane-local 的最快存储。最差的串行实现每一步把 z 写回 HBM、下一步再读。更合理的单 kernel 循环让 lane 把 x,y 留在 register，直到处理完负责的时间段。</p>
      <p>但普通 kernel 结束后，register 状态就消失。不要声称多个独立 launch 能“保持寄存器 state”；跨 launch 必须落到可见内存，除非使用 persistent kernel 等不同执行模型。</p>
      <div class="breakout" data-figure="memory"><div class="figure-head"><div><span class="figure-number">Interactive 2</span><h3>GPU Memory Journey</h3></div><span class="evidence derived">Logical model</span></div><div class="figure-toolbar"><div class="control-row"><label>Track<select data-memory-track><option value="gemm">Write GEMM</option><option value="serial">Serial recurrence</option><option value="scan">Chunk scan</option></select></label></div><div class="transport"><button data-memory-action="toggle" class="primary">Play</button><button data-memory-action="step">Step</button><button data-memory-action="reset">Restart</button></div></div><div class="memory-stage" data-memory-stage><div class="memory-level hbm">HBM<br>global tensors</div><div class="memory-level l2">L2<br>cache</div><div class="memory-level shared">Shared<br>CTA tile</div><div class="memory-level register">Registers<br>x,y</div><div class="memory-level tensor">Tensor Cores<br>dense MMA</div><div class="memory-level alu">ALU / SFU<br>exp · sincos · FMA</div></div><div class="traffic-readout"><div><span>Logical traffic（公式计数）</span><strong data-logical-traffic>—</strong></div><div><span>Profiler-measured DRAM traffic</span><strong>N/A</strong></div></div></div>`
  },
  {
    title: "一个 GPU thread 怎样算一个 mode？",
    context: ["lane-local x,y", "exp + sincos + FMA", "静态 ν,θ"],
    lead: "把一个 complex mode 分配给一个 lane：lane 只需自己的 ν、θ、x、y、w，并接收共享的 c,d。",
    body: `${tags(["proposed","Proposed mapping"])}
      <p>lane 先算 g=exp(c)，再算 ρ<sub>j</sub>=exp(−ν<sub>j</sub>g) 与 φ<sub>j</sub>=θ<sub>j</sub>+d。接着用二维旋转更新 x,y。只要 register 压力允许，状态在时间循环中可保持 lane-local。</p>
      ${math(String.raw`\operatorname{lane}_j:\ (\nu_j,\theta_j,x_j,y_j,w_{R,j},w_{I,j})\;\oplus\;\operatorname{shared}(c_t,d_t)`, "新 Triton kernel 已实现这个数据分工：c,d 每个 token/program 读取一次，mode state 以 FP32 累积。")}
      <div class="lesson-grid"><div class="meaning"><strong>已实现收益</strong>transition 不需要先 materialize 成 [B,L,M] gate tensor；kernel 从共享 c,d 与静态 ν,θ 现场重建。</div><div class="misconception"><strong>仍需 profile</strong>exp/sincos 使用特殊函数路径；register 增长可能压低 occupancy。环境没有 Nsight Compute，硬件 counters 仍保持 N/A。</div></div>`
  },
  {
    title: "一个 warp 怎样算 32 个 modes？",
    context: ["warp = 32 lanes", "lane 7 → mode 7", "同 c,d，不同响应"],
    lead: "warp 的 32 个 lane 以 lockstep 执行同一指令。SAMU 的 mode 公式相同，参数不同，天然形成规则的 SIMD 映射。",
    body: `${tags(["paper","CUDA fact"],["proposed","Proposed mapping"])}
      <p>点击任意 lane。上方的 c,d 对整个 warp 相同，lane 内的 ν,θ,x,y 与 write 不同，因此算出的 ρ、cosφ、sinφ 和新状态不同。共享控制没有消除 mode diversity；它只限制动态控制位于一个低维流形上。</p>
      <div class="breakout" data-figure="warp"><div class="figure-head"><div><span class="figure-number">Interactive 3</span><h3>Warp Viewer · mode parallel</h3></div><span class="evidence proposed">Proposed kernel</span></div><div class="control-row"><label>c<sub>t</sub><input data-warp-c type="range" min="-2" max="2" value="0" step=".05"></label><label>d<sub>t</sub><input data-warp-d type="range" min="-3.14" max="3.14" value="0.35" step=".05"></label><span class="mono" data-warp-control>c=0.00 · d=0.35</span></div><div class="broadcast-line"></div><div class="warp-layout"><div class="lane-grid" data-lane-grid></div><aside class="lane-detail" data-lane-detail>选择一个 lane</aside></div></div>
      ${quiz("c_t 是 mode-specific 还是 token-shared？", "mode-specific", "token-shared", "b", "在 canonical SAMU 中 c_t,d_t 是 token-wise 共享坐标；静态 ν_j,θ_j 把共享坐标变成各 mode 不同的 ρ_j,φ_j。")}`
  },
  {
    title: "c,d 为什么可以共享？",
    context: ["coherent selectivity", "shared coordinates", "static spectrum"],
    lead: "共享不是因为 32 个 modes 行为相同，而是因为同一个全局动作经过不同静态敏感度后会产生不同响应。",
    body: `${tags(["verified","Verified formula"],["hypothesis","Systems hypothesis"])}
      <p>c 改变 g=exp(c)，所有 retention 一起变，但较大的 ν 对 g 更敏感；d 给整个 frequency spectrum 加相同相位偏移，而各 mode 仍保留不同 θ。数学上这是 coherent control：少量动态坐标调制一个静态基底。</p>
      <div class="breakout" data-figure="coherent"><div class="figure-head"><div><span class="figure-number">Interactive 4</span><h3>Coherent Selectivity</h3></div><span class="evidence verified">Formula driven</span></div><div class="control-row"><label>c<sub>t</sub><input data-coherent-c type="range" min="-2" max="2" value="0" step=".02"></label><label>d<sub>t</sub><input data-coherent-d type="range" min="-3.14" max="3.14" value="0" step=".02"></label><button data-coherent-mode="samu" class="primary">SAMU shared c,d</button><button data-coherent-mode="rglru">RG-LRU-like gates</button></div><canvas data-coherent-canvas width="1100" height="430" aria-label="共享控制与 mode 响应图"></canvas><p data-coherent-note class="fairness-inline">概念可视化，不是 benchmark；RG-LRU 的 optimized memory traffic 不由控制维度直接推出。</p></div>
      <p>这套约束可能降低 controller 输出维度，也可能限制选择性。最终问题不只是吞吐，还包括质量—效率 Pareto frontier；本站没有重新训练大型任务，因此把质量优势保留为未验证问题。</p>`
  },
  {
    title: "多个 warp 如何形成 CTA？",
    context: ["warp 合作", "shared tile", "barrier 只在 CTA 内"],
    lead: "M 大于 32 时，一个 warp 不够。多个 warp 可覆盖 mode tiles，并在 CTA 内共享 token 控制或 chunk 元数据。",
    body: `${tags(["paper","CUDA fact"],["proposed","Proposed mapping"])}
      <p>例如 4 个 warp 覆盖 128 modes：每个 warp 处理连续 32 modes，c,d 只需进入每个 warp 的 broadcast 路径。若算法需要 time×mode tile 交换 partial summary，shared memory 是 CTA 内显式协作区。</p>
      ${math(String.raw`\operatorname{CTA}=N_{\mathrm{warp}}\times32\ \text{lanes},\qquad \operatorname{scope}(\mathrm{barrier})=1\ \mathrm{CTA}`, "CTA 之间不能用普通 barrier 同步；全局 prefix 往往需要多阶段 launch 或专门 persistent/cooperative 设计。")}
      <p>更多 warp 不保证更快。新 C32 kernel 的 A/B 测试恰好显示，在 M=64、L=65536 上 2 warps 比 4 warps 约快 12%；因此默认对 64-mode block 使用 2 warps，更宽 block 仍保留 4 warps。这个选择只对当前 RTX 3090 测量负责。</p>`
  },
  {
    title: "长 sequence 为什么让 GPU 不够忙？",
    context: ["parallel jobs ≈ B×M", "serial depth = L", "82 SM"],
    lead: "串行 recurrence 的总工作很多，却可能只有 B×M 条独立链。工作量大不等于同时可调度的工作多。",
    body: `${tags(["derived","Derived occupancy model"])}
      <p>在 B=1,M=64 时，理想化地只有 64 条 mode chains；RTX 3090 有 82 SM。即便一个 warp 负责多个 mode，时间维仍串行，硬件无法把未来 timestep 提前执行来隐藏当前指令的延迟。</p>
      ${math(String.raw`N_{\mathrm{parallel}}\approx B\!\cdot\!M,\qquad \operatorname{span}_{\mathrm{serial}}=L`, "这只是调度直觉，不是 occupancy profiler 数字。")}
      <p>官方 RecurrentGemma 的 <span class="var">rnn_scan</span> 对 L&gt;1 在 Python 中逐 timestep 循环，FP32 accumulator 则跨步更新。它是可信的公式/source reference，却不是专用 GPU scan kernel；因此长序列的巨大 wall-clock 差距首先是执行路径差距。</p>`
  },
  {
    title: "为什么可以切 chunk？",
    context: ["局部串行", "chunk 独立 summary", "边界 state"],
    lead: "把长度 L 切成 K 个 chunk，每块先在 zero input state 下形成自己的 affine summary，再并行接回真实边界状态。",
    body: `${tags(["derived","Derived"])}
      <p>chunk 内仍有顺序，但 K 个 chunk 的参数只依赖各自 token，可以并行生成。之后 scan 求出每个 chunk 的输入 state，再进行 local replay 或 correction。这把一个长度 L 的依赖链变成 chunk_size 的局部深度加 log K 的全局组合。</p>
      ${math(String.raw`L=K\,C_{\mathrm{chunk}}\quad\Longrightarrow\quad \operatorname{span}\approx C_{\mathrm{chunk}}+\log_2K`, "work 与 materialization 会增加；chunking 不是 SAMU 独有，也不是无条件加速。")}
      <div class="micro-example"><strong>极小例子</strong>16 tokens 切成 4×4：先同时总结 [0…3]、[4…7]、[8…11]、[12…15]，再 scan 4 个 summaries。</div>`
  },
  {
    title: "什么是 affine operator？",
    context: ["T(z)=Pz+q", "组合有结合律", "顺序不可交换"],
    lead: "每个 timestep 不是一个 state 数值，而是一个‘给我输入 state，我返回输出 state’的函数。函数可以先组合，再作用。",
    body: `${tags(["derived","Algebraic derivation"])}
      <p>令 T₁(z)=P₁z+q₁，T₂(z)=P₂z+q₂。先做 T₁ 再做 T₂，结果仍是 affine map：P 相乘，q 被后一个 P 变换后相加。组合有结合律，所以可以用树；但通常不可交换，token 顺序不能打乱。</p>
      ${math(String.raw`\boxed{(P_2,q_2)\circ(P_1,q_1)=\bigl(P_2P_1,\;P_2q_1+q_2\bigr)}`, "(T₃∘T₂)∘T₁ = T₃∘(T₂∘T₁)，这就是 parallel scan 的代数入口。")}
      <div class="lesson-grid"><div class="micro-example"><strong>极小例子</strong>T₁(z)=2z+1，T₂(z)=3z+4；组合为 6z+7。</div><div class="misconception"><strong>常见误解</strong>“可结合”不是“可交换”。2→3 的 transition 与 3→2 的 q 通常不同。</div></div>`
  },
  {
    title: "q 到底是什么？",
    context: ["q = T(0)", "zero-state output", "O(M)"],
    lead: "q 是一个 segment 在零输入状态下产生的输出。它把 segment 内所有 write 经过后续 transition 的影响累积起来。",
    body: `${tags(["derived","Derived"])}
      <p>若 segment summary 是 T(z)=Pz+q，那么代入 z=0 立刻得到 q=T(0)。真实输入 z_in 到来时，不必重做全部代数，只需算 Pz_in+q。q 的尺寸与 state 相同，因为每个 mode 都可能有不同的累积写入。</p>
      ${math(String.raw`q=w_{n}+A_{n}w_{n-1}+A_{n}A_{n-1}w_{n-2}+\cdots`, "q 仍有 M complex entries，即 O(M)；只有 transition P 的描述在 SAMU 中可进一步压缩。")}
      ${quiz("(C,G,D,q) 整体是不是 O(1)？", "是", "不是", "b", "不是。C,G,D 是常数个 transition scalars，但 q 是 state-sized 的 zero-state output，仍是 O(M)。")}`
  },
  {
    title: "transition 为什么能压缩成 C,G,D？",
    context: ["P_j = exp(−ν_jG)·exp(i(Cθ_j+D))", "3 scalars + q", "exact structure"],
    lead: "SAMU 的每步 transition 共用 c,d，并通过静态 ν,θ 生成 mode 响应。一个 segment 的对角 transition 因此保留同一函数族。",
    body: `${tags(["derived","Exact derivation"],["proposed","Kernel proposal"])}
      <p>把 segment 内保留因子相乘：指数里的 −ν<sub>j</sub>exp(c<sub>t</sub>) 相加，因此只需 G=Σ exp(c<sub>t</sub>)。相位相加后得到 C·θ<sub>j</sub>+D；若每步都有一次基础 θ，则 C 是 token count，D=Σd。</p>
      ${math(String.raw`\boxed{P_{j,\mathcal S}=\exp(-\nu_jG_{\mathcal S})\,\exp\!\bigl(i(C_{\mathcal S}\theta_j+D_{\mathcal S})\bigr)}`, "compressed transition = (C,G,D)；完整 affine summary = (C,G,D,q)，其中 q 仍是 mode vector。")}
      <p>这不是近似压缩，而是由 canonical 结构推导的 exact representation。本站 PyTorch prototype 已与直接 recurrence 做数值正确性检查；但高性能 advanced scan custom kernel 尚不存在，所以性能仍是 proposal。</p>`
  },
  {
    title: "prefix scan 怎样接回 chunks？",
    context: ["offset 1 → 2 → 4", "exclusive prefix", "chunk inputs"],
    lead: "每个 chunk 先只知道自己的 summary。prefix scan 计算它之前所有 chunks 的组合，于是得到它的真实输入 state。",
    body: `${tags(["derived","Derived"])}
      <p>对 8 chunks，Hillis–Steele 风格演示有三轮：offset 1、2、4。每轮只有满足索引条件的节点参与 combine。工程实现可以选择更节省 work 的 scan，但结合律和前缀含义不变。</p>
      <div class="breakout" data-figure="scan"><div class="figure-head"><div><span class="figure-number">Interactive 5</span><h3>Chunked SAMU Scan Explorer</h3></div><span class="evidence derived">Exact algebra</span></div><div class="control-row"><button data-scan-mode="tokens" class="primary">4 tokens · compose</button><button data-scan-mode="compressed">Fold P → C,G,D</button><button data-scan-mode="prefix">8 chunks · prefix</button></div><div class="scan-track" data-scan-track></div><div class="scan-equation" data-scan-equation></div><div class="transport"><button data-scan-action="back">Previous round</button><button data-scan-action="toggle" class="primary">Autoplay</button><button data-scan-action="step">Next round</button><button data-scan-action="reset">Restart</button><select data-scan-speed><option value=".5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select></div></div>
      <p>exclusive prefix 给 chunk k 之前的总 map；把初始 state 作用到这个 map，就得到 chunk k 的 boundary state。之后 chunk 内可以以真实边界 state 生成全部 token states。</p>`
  },
  {
    title: "chunk 内 time scan",
    context: ["intra-chunk prefix", "two levels", "materialization tradeoff"],
    lead: "全局 scan 只给边界；要输出每个 timestep，还要在 chunk 内恢复 prefix，或重放局部 recurrence。",
    body: `${tags(["derived","Derived"],["proposed","Prototype"])}
      <p>一种方案先保存每步 affine pair，在 chunk 内做 local scan；另一种方案只保存 chunk summary，边界到达后串行 replay。前者增加临时 tensor，后者增加串行深度。最优选择依赖 chunk size、M、dtype 与是否需要 backward。</p>
      ${math(String.raw`\operatorname{prefill}=\operatorname{scan}(K\ \mathrm{summaries})+\operatorname{replay}(C_{\mathrm{chunk}}\ \mathrm{steps})`, "当前实现是 1 次 packed GEMM + summary / prefix / replay 三个 Triton kernels。")}
      <p>新实测显示 crossover 很明确：L≤256 使用 C8，L=512 使用 C16，更长使用 C32；L=65536 时 C32 明显优于 C16/C8。serial kernel 仍保留作研究基线，但在这组 M=64、RTX 3090 shape 上不再进入 auto。</p>`
  },
  {
    title: "time × mode：二维并行面",
    context: ["horizontal = mode", "vertical = time", "tile needs benchmark"],
    lead: "mode parallelism 横向展开，time scan 纵向展开。kernel 的工作是把二维平面切成可调度 tiles。",
    body: `${tags(["proposed","Design space"])}
      <p>串行 recurrence 点亮少数 mode columns，并沿时间向下走；time-parallel scan 同时点亮多个 chunk tiles。CTA tile 可以是 32×8、32×16、64×8，但这些只是候选布局。</p>
      <div class="breakout" data-figure="timemode"><div class="figure-head"><div><span class="figure-number">Interactive 6</span><h3>Time × Mode Tiling</h3></div><span class="evidence hypothesis">Needs benchmark</span></div><div class="control-row"><button data-tm-kind="serial" class="primary">Serial recurrence</button><button data-tm-kind="scan">Time-parallel scan</button><label>CTA tile<select data-tm-tile><option value="32x8">32 time × 8 modes</option><option value="32x16">32 × 16</option><option value="64x8">64 × 8</option></select></label></div><div class="time-mode"><canvas id="time-mode-canvas" data-tm-canvas width="850" height="560" aria-label="时间乘 mode 矩阵"></canvas><aside class="tile-stats" data-tm-stats></aside></div></div>
      <p>旁边的 CTA/warp/job 数是按几何形状计算的 logical schedule，不是 occupancy。当前 kernel 已覆盖 M=64/128/256 的 equal-state sweep，但尚未收集 registers/thread 与 occupancy counters；图中的 tile 仍是教学候选，不等同于 profiler 结论。</p>`
  },
  {
    title: "training / prefill backend",
    context: ["large L", "dense projections", "chunk + scan"],
    lead: "训练与 prefill 一次看到完整序列：它们可以批量 GEMM，也有机会沿时间做并行 scan。",
    body: `${tags(["verified","Implemented Triton"],["measured","Forward measured"])}
      <p>新 prefill 路径先把 write 的 2M 列与 controller 的 2 列打包为一次 BF16 GEMM，再由 summary、exclusive prefix 与 local replay 以 FP32 state 执行 recurrence。shape dispatch 在计时前解析，双方都计入 1 次 projection 与 3 个 Triton launches。</p>
      ${math(String.raw`\operatorname{prefill}=1\ \mathrm{packed\ GEMM}+\begin{cases}\mathrm{C8},&L\le256,\\ \mathrm{C16},&256<L\le512,\\ \mathrm{C32},&L>512.\end{cases}`, "L≤128 且解析参数界通过时启用 bounded-exp；其他 shape 使用通用 exp。阈值来自 RTX 3090 实测，不外推其他 GPU。")}
      <p>该路径是 inference-only；还没有 backward kernel。训练需要为中间状态选择保存、重算或反向 scan，因此网页不再用 forward 数字暗示训练速度。</p>`
  },
  {
    title: "decode backend",
    context: ["L = 1", "launch dominated", "persistent state is separate design"],
    lead: "autoregressive decode 每次只有一个新 token。没有长 time axis 可 scan，核心问题转为小批量投影、state load/store 与 kernel launch。",
    body: `${tags(["verified","Single-launch Triton"],["measured","Total decode measured"])}
      <p>新 decode kernel 从显存读 token 与 FP32 state，在同一个 launch 内完成 write dot products、两个 controller dot products、有界 c,d、复数 transition 和 state store。没有把 projection 偷移到计时外。</p>
      ${math(String.raw`1\ \mathrm{launch}=\mathrm{write}+\mathrm{control}+\mathrm{transition}+\mathrm{state\ update}`, "B=1,d=128,M=64：SAMU 与 RG-LRU 的 median 都是 0.007168 ms，等于当前事件分辨率；不能据此宣称 decode 胜出。")}
      <p>SAMU auto-short BF16 output 最大绝对误差为 0.00390625、最终 cache 最大误差为 1.31e-6；相位增量限制在 ±0.077114 rad，degree-6/7 cos/sin 多项式的 float32 最大误差分别为 5.96e-8 与 7.45e-9。</p>
      <div class="misconception"><strong>常见误解</strong>普通多次 kernel invocation 不能让 register state 跨 step 存活。persistent decode 是值得实现的新 backend，但必须单独设计和测量。</div>`
  },
  {
    title: "RG-LRU：独立 gate 与共享坐标的差别",
    context: ["per-channel dynamic gate", "official-equation Triton", "control dimension ≠ HBM traffic"],
    lead: "RG-LRU 为 channel/mode 形成更独立的动态 gating；SAMU 用 c,d 经过静态 spectrum 产生 coherent response。前者更自由，后者更受约束。",
    body: `${tags(["verified","Official Eq. 1–4"],["measured","Equal Triton measured"])}
      <p>比较要分两层。数学层：RG-LRU 的 dynamic quantities 随 channel 展开，SAMU 只有少量共享 coordinates。实现层：fused RG-LRU 可以现场生成 gates、复用 cache，未必 materialize O(LM) gate tensor 到 HBM。</p>
      ${math(String.raw`\begin{aligned}i_t&=\sigma(W_xx_t+b_x),\\ r_t&=\sigma(W_ax_t+b_a),\\ \log a_t&=-8\,r_t\odot\operatorname{softplus}(a_{\mathrm{param}}),\\ h_t&=a_t\odot h_{t-1}+\sqrt{1-a_t^2}\odot(i_t\odot x_t).\end{aligned}`, "这是 Griffin / RecurrentGemma 的 RG-LRU 语义；segment_pos=0 时官方实现令 aₜ=0，并把输入 multiplier 置为 1。")}
      <div class="provenance-grid"><div><span class="evidence paper">Official</span><strong>方程与语义</strong><p>两组 block-diagonal gates、−8·softplus 参数化、平方根归一化、segment reset、BF16 activation 与 FP32 cache。</p></div><div><span class="evidence derived">This work</span><strong>加速实现</strong><p>双 gate 合并为一次 packed BMM、serial/chunk summary-prefix-replay、单 launch decode 与 RTX 3090 shape dispatch。</p></div><div><span class="evidence derived">Generic algebra</span><strong>不是官方专属</strong><p>affine 结合律与 chunk scan 是通用方法；我们把它严谨适配到官方 RG-LRU，而不声称发明 scan。</p></div></div>
      <div class="breakout" data-figure="compare"><div class="figure-head"><div><span class="figure-number">Interactive 7</span><h3>Same Token, Two Recurrences</h3></div><span class="evidence paper">Structure, not speed</span></div><div class="compare-three two-models" data-compare-cards></div><div class="formula" data-compare-formula>选择模型查看同一 u<sub>t</sub> 经过哪些动态量与 state update。</div></div>
      <p>本站的同等级对照严格复现官方逐项运算次序。fused decode 的 FP32 dot 会先在 projection 边界舍入到 BF16，再加 BF16 bias，从而与官方 BlockDiagonalLinear 的数值路径一致。固定 commit 的官方 PyTorch 层仍保留为 source reference，但不再承担架构性能对手。</p>`
  },
  {
    title: "从 RG-LRU 到更快的 SAMU 执行路径",
    context: ["packed projection", "on-chip transition", "shape-aware dispatch"],
    lead: "SAMU 的优势不是省掉 state-sized write，而是把动态 transition 的描述压到 c,d，使 projection、重建与递推能形成更紧凑的 kernel 边界。",
    body: `${tags(["verified","Implemented"],["derived","Architecture mapping"])}
      <p>RG-LRU Triton 每个 token 生成 input gate 与 recurrence gate，再进入 FP32 scan。SAMU 把 2M 维 complex write 与 2 维 selector 合成一次 padded BF16 projection；recurrence kernel 只接收 packed rows、静态 ν/θ/γ 和 FP32 state。</p>
      <div class="lesson-grid"><div class="meaning"><strong>独特可利用点</strong>同一 token 的 c,d 对全部 modes 共享；相位增量与衰减指数都有解析界。bounded-exp 只在 L≤128、pack-time bound check 通过且实测获益时启用，否则回退通用 exp。</div><div class="misconception"><strong>没有消失的成本</strong>write 仍是 2M 维；长序列 chunk 仍需 O(M) 的 q。优化数学函数不等于改变方程。</div></div>
      ${math(String.raw`|c_t|\le\frac{|s_c|}{2},\qquad 0\le\nu_j e^{c_t}\le\nu_{\max}e^{|s_c|/2}`, "这两个解析界决定 bounded-exp fast path 是否安全；不是从 benchmark 输入分布拟合出来的阈值。")}
      <div class="breakout" data-figure="transition"><div class="figure-head"><div><span class="figure-number">Interactive 8</span><h3>Transition Representation</h3></div><span class="evidence paper">Formula-level</span></div><div class="control-row"><button data-transition-span="token" class="primary">1 token</button><button data-transition-span="tokens">32 tokens</button><button data-transition-span="chunk">1 chunk</button></div><div class="transition-view two-models" data-transition-view></div></div>`
  },
  {
    title: "真实 benchmark：先看实现等级，再看折线",
    context: ["54 measured + 4 unsupported", "same width + params + state", "two-track scope"],
    lead: "一张曲线只有在回答清楚‘测了谁的什么实现’后才有意义。主轨道使用同等级 Triton kernel；官方实现另起一条背景轨道。",
    body: `${tags(["measured","54 measured"],["verified","Raw JSON retained"])}
      <p id="fairness">Track A 固定 <b>d=128, M=64</b>：SAMU 16,772 参数、RG-LRU 16,768 参数；两边都使用 512 B FP32 recurrent state、BF16 input/output，并拥有 chunk prefill 与 fused decode。Track B 的 Mamba-3 只匹配 width，不匹配参数或 state。</p>
      <p>sweep 覆盖 L=128…65536、prefill B=1…64、decode B=1…64，以及两边 C8/C16/C32。58 行中 54 measured；官方 Mamba-3 CuTe decode 在 RTX 3090 上不可用，4 行明确标记 unsupported。</p>
      ${math(String.raw`h_t=e^{\Delta_tA_t}h_{t-1}+(1-\lambda_t)\Delta_t e^{\Delta_tA_t}B_{t-1}x_{t-1}+\lambda_t\Delta_tB_tx_t`, "Mamba-3 论文的 exponential-trapezoidal update。本站 Track B 直接调用固定 commit 的官方 Mamba3 forward / SISO Triton 路径。")}
      <p><b>Mamba 审计边界：</b>官方 <code>Mamba3.step()</code> 在该 commit 中直接依赖 CuTe <code>mamba3_step_fn</code>；仓库虽有独立 Triton SISO step 文件，但没有接入模块的 <code>step()</code>。把它手动包起来会变成我们的集成，不再是“未修改官方 module decode”，因此 RTX 3090 上保持 unsupported 而不是偷换 backend。</p>
      ${math(String.raw`\operatorname{timing}=\operatorname{median}\!\left(\mathrm{Triton\ driver\ event\ samples}\right),\qquad \mathrm{compile,dispatch,warmup}\notin\mathrm{samples}`, "每个 timed call 前清 L2；保留 p10/p90/p95 与 raw samples。launch、register、spill 来自 profiler/compiler metadata；无 Nsight Compute 的硬件 counters 保持 N/A。")}
      <p>向下进入 Benchmark Lab，可切换 length、batch、state、decode 与 kernel crossover，并点击每个点追到原始样本。</p>`
  },
  {
    title: "结论不是口号，而是一张未完成的研究地图",
    context: ["measured today", "next kernel", "open questions"],
    lead: "当前证据已经足以淘汰几种过强说法，也足以确定下一步工程优先级。",
    body: `${tags(["measured","Measured"],["hypothesis","Open questions"])}
      <p><b>已测结论：</b>同等级 Triton、冷 L2 driver-event 计时下，SAMU 在六个 B1 prefill 点全部领先：L=128/512/2048 分别约 1.06×/1.04×/1.07×，L=8192/32768/65536 约 1.36×/2.03×/1.55×。B1/B4/B16/B64 decode 的 median 均与 RG-LRU 持平，受限于 1.024 µs 事件分辨率；这里不写成“decode 胜出”。</p>
      <p><b>资源解释：</b>C32 L2048 两边都是 4 launches；SAMU 的静态 exp/sigmoid 计数为 134/token，RG-LRU 为 1024/token 且另有 256 sqrt。SAMU summary/prefix/replay 是 64/30/28 registers，RG-LRU 是 207/27/30；前者 summary 的 derived occupancy 为 66.7%，后者为 16.7%。硬件 SFU/DRAM utilization 仍需 Nsight Compute 才能确认。</p>
      <div class="lesson-grid"><div class="meaning"><strong>继续验证</strong>在 Hopper/Ada 上复测；补硬件 DRAM/SFU counters；runtime autotune；backward；训练质量—效率 Pareto。</div><div class="misconception"><strong>仍是 hypothesis</strong>coherent control 会形成更好的质量—效率 Pareto frontier；当前实验没有重新训练模型质量。</div></div>
      <p>课程到这里不要求相信 SAMU 更快，只要求你现在能指出：一次 timestep 做什么、数据可能停在哪里、为什么 write 与 recurrence 不同、scan 怎样成立，以及每条性能结论究竟来自公式、代码还是测量。</p>`
  }
];
