export const parts = {
  0: { id: "learn", roman: "I", title: "Learn the Operator", text: "先把一次递推拆成 old memory、transition 与 new write。每一课只增加一个概念。" },
  5: { id: "gpu-explorer", roman: "II", title: "See It Run on GPU", text: "沿数据路径进入 HBM、warp、register、CTA、chunk 与 scan；图中的运动对应运算状态，而非装饰。" },
  20: { id: "compare", roman: "III", title: "Compare the Architectures", text: "让同一个 token 分别通过 SAMU、RG-LRU 与 Mamba-3，区分数学控制维度与实际 kernel traffic。" },
  22: { id: "implementation", roman: "IV", title: "Benchmark Lab", text: "最后才看 wall-clock。实现等级、公平协议、失败项和硬件计数器缺口全部保留。" }
};

const tags = (...items) => `<div class="evidence-row">${items.map(([c,t]) => `<span class="evidence ${c}">${t}</span>`).join("")}</div>`;
const quiz = (q, a, b, correct, why) => `<div class="quiz" data-correct="${correct}"><p>${q}</p><div class="quiz-options"><button data-answer="a">A · ${a}</button><button data-answer="b">B · ${b}</button></div><p class="quiz-feedback" aria-live="polite">选择后展开原因。</p><template>${why}</template></div>`;

export const lessons = [
  {
    title: "五分钟总览：把 hypothesis 变成可测对象",
    context: ["三段计算", "write → transition → state", "数学结构不等于 kernel 性能"],
    lead: "SAMU 不只是一个公式。要判断它是否适合 GPU，必须沿着从 token 到数据布局、从 warp 到 benchmark 的完整链路走一遍。",
    body: `${tags(["verified","Verified code"],["hypothesis","Hypothesis"])}
      <p>给定输入 <span class="var">u<sub>t</sub></span>，canonical 路径先由 controller 产生两个有界标量，再把输入投影为所有 mode 的复数写入。旧状态经过 mode-specific 的保留与旋转，最后加上 write。这个分解把问题切成三块：<b>可并行的 dense projection</b>、<b>带时间依赖的 recurrence</b>、以及<b>两者之间的数据交接</b>。</p>
      <div class="formula">u<sub>t</sub> → {c<sub>t</sub>, d<sub>t</sub>} &nbsp; | &nbsp; B u<sub>t</sub> → w<sub>t</sub> &nbsp; | &nbsp; z<sub>t+1</sub> = A<sub>t</sub>z<sub>t</sub> + w<sub>t</sub><small>三段在数学上相邻，在 GPU 上却可能属于不同 kernel。</small></div>
      <div class="lesson-grid"><div class="micro-example"><strong>极小例子</strong>4 个 complex modes 就是 8 个 real state scalars。一个 token 只提供共享的 c,d，但每个 mode 有自己的 ν,θ。</div><div class="meaning"><strong>实际意义</strong>这给出一个可证伪问题：共享控制能否降低控制开销，而 mode-wise exp/rotation 又会花掉多少算力？</div></div>
      <p>本课程中的“适合 GPU”从不等于“理论 bytes 更少”。它至少同时包含并行度、kernel 数、launch 开销、寄存器压力、数学特殊函数和数据重用。后面的实测会表明，当前 PyTorch prototype 与官方 fused Mamba-3 之间仍有巨大的实现鸿沟。</p>`
  },
  {
    title: "SAMU 到底算什么？",
    context: ["复数递推", "ρ = exp(−ν·exp(c))", "φ = θ + d"],
    lead: "先忘掉 chunk 和 scan。一个 timestep 只是：读旧记忆、衰减并旋转、加入新写入。",
    body: `${tags(["verified","Verified code"],["derived","Derived"])}
      <p>对 mode j，SAMU 的状态是复数 <span class="var state">z<sub>j,t</sub>=x+iy</span>。静态参数 <span class="var">ν<sub>j</sub>, θ<sub>j</sub></span> 描述该 mode 的基础时间尺度与相位；动态标量 c,d 则由当前 token 调整整组 spectrum。</p>
      <div class="formula">z<sub>j,t+1</sub> = exp[−ν<sub>j</sub> exp(c<sub>t</sub>)] · exp[i(θ<sub>j</sub>+d<sub>t</sub>)] · z<sub>j,t</sub> + w<sub>j,t</sub><small>ρ 控制幅度保留，φ 控制二维平面旋转，w 是当前 token 的复数写入。</small></div>
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
      <div class="formula">controller: [B,L,d] → [B,L,2] &nbsp;&nbsp; write: [B,L,d] × [d,2M] → [B,L,2M]<small>canonical PPO 使用 raw observation 作为 controller 输入；formal PyTorch wrapper 可有独立 γ_log，不能混称同一实现。</small></div>
      <p>对 kernel 来说，shape 决定连续内存、向量化和 CTA 分工。c,d 很小，不表示整个 recurrence summary 是 O(1)：每个 token 的 write 仍是 O(M)，每个 chunk 的 zero-state output q 也仍是 O(M)。</p>
      <div class="lesson-grid"><div class="meaning"><strong>GPU 数据</strong>BF16 下一个 complex write 占 4 bytes；B×L×M 的 write tensor 的逻辑体积是 4BLM bytes。</div><div class="misconception"><strong>常见误解</strong>“两个 dynamic scalars”描述控制维度，不自动等于两个 HBM load。fused kernel 可以广播或重算，实际 traffic 必须 profile。</div></div>`
  },
  {
    title: "write 为什么可以一次 GEMM？",
    context: ["所有 timestep 独立", "[BL,d] × [d,2M]", "Tensor Core 友好"],
    lead: "write 只依赖当前输入，不依赖上一步状态。把 batch 与 time 合并后，所有 token 可以同时进入矩阵乘。",
    body: `${tags(["verified","Verified code"],["paper","GPU fact"])}
      <p>将 U reshape 成 <span class="var">[B·L,d]</span>，将复数投影拆成实部与虚部权重，便得到普通 dense GEMM。这里没有时间箭头：第 900 个 token 的 write 不需要等待第 899 个 state。</p>
      <div class="formula">[B·L,d] · [d,2M] → [B·L,2M] = [w<sub>R</sub> | w<sub>I</sub>]<small>矩阵维度足够大时，可由高吞吐 matmul/Tensor Core 路径处理。</small></div>
      <div class="lesson-grid"><div class="micro-example"><strong>极小例子</strong>B=2,L=4,d=3,M=2：一次 [8,3]×[3,4] 产生 8 个 token、2 个 complex modes 的写入。</div><div class="meaning"><strong>实际意义</strong>训练/prefill 可以先批量预计算 write；decode 的 L=1 很小，launch 与融合策略更重要。</div></div>
      ${quiz("write_t 能不能对所有 timestep 并行计算？", "可以", "不可以", "a", "可以。它只依赖各自的 u_t 与共享权重；时间依赖从 z_t → z_{t+1} 才开始。")}`
  },
  {
    title: "为什么 recurrence 是另一种问题？",
    context: ["z_{t+1} depends on z_t", "work/span", "关联而非 GEMM"],
    lead: "同一 mode 上，下一步读的是刚刚产生的 state。把所有时间点一起启动，会缺少真实输入。",
    body: `${tags(["verified","Verified code"],["derived","Derived"])}
      <p>recurrence 的 work 是 O(LM)，串行 span 也是 O(L)：每个 mode 可以并行，但每个 mode 内沿时间排成链。短序列时，M 个并行 job 也许够喂饱 GPU；当 M 很小而 L 很长，数十个活跃 lane 无法覆盖 82 个 SM。</p>
      <div class="formula">z<sub>0</sub> → z<sub>1</sub> → z<sub>2</sub> → … → z<sub>L</sub><small>把 write 一次 GEMM 出来，并没有移除 state chain。</small></div>
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
      <div class="formula">lane j: (ν<sub>j</sub>,θ<sub>j</sub>,x<sub>j</sub>,y<sub>j</sub>,w<sub>R,j</sub>,w<sub>I,j</sub>) + broadcast(c,d)<small>映射是 proposal；当前仓库没有被审计通过的 production SAMU custom kernel。</small></div>
      <div class="lesson-grid"><div class="meaning"><strong>可能收益</strong>g 与 d 可 warp-broadcast，避免每个 lane 从大 control tensor 读取独立 gate。</div><div class="misconception"><strong>可能瓶颈</strong>exp/sincos 使用特殊函数路径；register 增长会压低 occupancy。两者当前没有 Nsight counters，保持假设。</div></div>`
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
      <div class="formula">CTA = warps × 32 lanes &nbsp;&nbsp; | &nbsp;&nbsp; barrier scope = one CTA<small>CTA 之间不能用普通 barrier 同步；全局 prefix 往往需要多阶段 launch 或专门 persistent/cooperative 设计。</small></div>
      <p>更多 warp 不保证更快。每个 lane 保留 x,y、临时 sincos 和 write 会消耗 register；shared memory 也限制 resident CTAs。mode tile=64/128/256/512 哪个最好仍需 custom kernel 后实测。</p>`
  },
  {
    title: "长 sequence 为什么让 GPU 不够忙？",
    context: ["parallel jobs ≈ B×M", "serial depth = L", "82 SM"],
    lead: "串行 recurrence 的总工作很多，却可能只有 B×M 条独立链。工作量大不等于同时可调度的工作多。",
    body: `${tags(["derived","Derived occupancy model"])}
      <p>在 B=1,M=64 时，理想化地只有 64 条 mode chains；RTX 3090 有 82 SM。即便一个 warp 负责多个 mode，时间维仍串行，硬件无法把未来 timestep 提前执行来隐藏当前指令的延迟。</p>
      <div class="formula">parallel jobs ≈ B·M &nbsp;&nbsp; ; &nbsp;&nbsp; serial span = L<small>这只是调度直觉，不是 occupancy profiler 数字。</small></div>
      <p>当前实测的官方 RG-LRU PyTorch reference 正好暴露了这个问题，但它还是 Python serial loop，不能据此断言 RG-LRU 架构本身慢。Profiler 记录到 24,628 个事件；这首先说明实现层面存在循环与 launch 开销。</p>`
  },
  {
    title: "为什么可以切 chunk？",
    context: ["局部串行", "chunk 独立 summary", "边界 state"],
    lead: "把长度 L 切成 K 个 chunk，每块先在 zero input state 下形成自己的 affine summary，再并行接回真实边界状态。",
    body: `${tags(["derived","Derived"])}
      <p>chunk 内仍有顺序，但 K 个 chunk 的参数只依赖各自 token，可以并行生成。之后 scan 求出每个 chunk 的输入 state，再进行 local replay 或 correction。这把一个长度 L 的依赖链变成 chunk_size 的局部深度加 log K 的全局组合。</p>
      <div class="formula">L = K·C<sub>chunk</sub> &nbsp;&nbsp; → &nbsp;&nbsp; span ≈ C<sub>chunk</sub> + log₂K<small>work 与 materialization 会增加；chunking 不是 SAMU 独有，也不是无条件加速。</small></div>
      <div class="micro-example"><strong>极小例子</strong>16 tokens 切成 4×4：先同时总结 [0…3]、[4…7]、[8…11]、[12…15]，再 scan 4 个 summaries。</div>`
  },
  {
    title: "什么是 affine operator？",
    context: ["T(z)=Pz+q", "组合有结合律", "顺序不可交换"],
    lead: "每个 timestep 不是一个 state 数值，而是一个‘给我输入 state，我返回输出 state’的函数。函数可以先组合，再作用。",
    body: `${tags(["derived","Algebraic derivation"])}
      <p>令 T₁(z)=P₁z+q₁，T₂(z)=P₂z+q₂。先做 T₁ 再做 T₂，结果仍是 affine map：P 相乘，q 被后一个 P 变换后相加。组合有结合律，所以可以用树；但通常不可交换，token 顺序不能打乱。</p>
      <div class="formula">T₂∘T₁ = (P₂P₁, P₂q₁+q₂)<small>(T₃∘T₂)∘T₁ = T₃∘(T₂∘T₁)，这就是 parallel scan 的代数入口。</small></div>
      <div class="lesson-grid"><div class="micro-example"><strong>极小例子</strong>T₁(z)=2z+1，T₂(z)=3z+4；组合为 6z+7。</div><div class="misconception"><strong>常见误解</strong>“可结合”不是“可交换”。2→3 的 transition 与 3→2 的 q 通常不同。</div></div>`
  },
  {
    title: "q 到底是什么？",
    context: ["q = T(0)", "zero-state output", "O(M)"],
    lead: "q 是一个 segment 在零输入状态下产生的输出。它把 segment 内所有 write 经过后续 transition 的影响累积起来。",
    body: `${tags(["derived","Derived"])}
      <p>若 segment summary 是 T(z)=Pz+q，那么代入 z=0 立刻得到 q=T(0)。真实输入 z_in 到来时，不必重做全部代数，只需算 Pz_in+q。q 的尺寸与 state 相同，因为每个 mode 都可能有不同的累积写入。</p>
      <div class="formula">q = w<sub>last</sub> + A<sub>last</sub>w<sub>prev</sub> + A<sub>last</sub>A<sub>prev</sub>w<sub>prev−1</sub> + …<small>q 仍有 M complex entries，即 O(M)；只有 transition P 的描述在 SAMU 中可进一步压缩。</small></div>
      ${quiz("(C,G,D,q) 整体是不是 O(1)？", "是", "不是", "b", "不是。C,G,D 是常数个 transition scalars，但 q 是 state-sized 的 zero-state output，仍是 O(M)。")}`
  },
  {
    title: "transition 为什么能压缩成 C,G,D？",
    context: ["P_j = exp(−ν_jG)·exp(i(Cθ_j+D))", "3 scalars + q", "exact structure"],
    lead: "SAMU 的每步 transition 共用 c,d，并通过静态 ν,θ 生成 mode 响应。一个 segment 的对角 transition 因此保留同一函数族。",
    body: `${tags(["derived","Exact derivation"],["proposed","Kernel proposal"])}
      <p>把 segment 内保留因子相乘：指数里的 −ν<sub>j</sub>exp(c<sub>t</sub>) 相加，因此只需 G=Σ exp(c<sub>t</sub>)。相位相加后得到 C·θ<sub>j</sub>+D；若每步都有一次基础 θ，则 C 是 token count，D=Σd。</p>
      <div class="formula">P<sub>j,segment</sub> = exp(−ν<sub>j</sub> G) · exp[i(C θ<sub>j</sub> + D)]<small>compressed transition = (C,G,D)；完整 affine summary = (C,G,D,q)，其中 q 仍是 mode vector。</small></div>
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
      <div class="formula">global: scan K summaries &nbsp;&nbsp; + &nbsp;&nbsp; local: scan/replay C<sub>chunk</sub> steps<small>当前 compressed PyTorch prototype 不是 fused kernel，测到的 chunk-size 趋势主要反映 Python/Torch operator 结构。</small></div>
      <p>RTX 3090 快速 sweep 中，L=2048 的 prototype 在 tested set 里 C=16 最快，median 8.05 ms；C=512 增至 134.21 ms。这是“当前 prototype 的最佳 chunk”，不是未来 custom kernel 的普适最优值。</p>`
  },
  {
    title: "time × mode：二维并行面",
    context: ["horizontal = mode", "vertical = time", "tile needs benchmark"],
    lead: "mode parallelism 横向展开，time scan 纵向展开。kernel 的工作是把二维平面切成可调度 tiles。",
    body: `${tags(["proposed","Design space"])}
      <p>串行 recurrence 点亮少数 mode columns，并沿时间向下走；time-parallel scan 同时点亮多个 chunk tiles。CTA tile 可以是 32×8、32×16、64×8，但这些只是候选布局。</p>
      <div class="breakout" data-figure="timemode"><div class="figure-head"><div><span class="figure-number">Interactive 6</span><h3>Time × Mode Tiling</h3></div><span class="evidence hypothesis">Needs benchmark</span></div><div class="control-row"><button data-tm-kind="serial" class="primary">Serial recurrence</button><button data-tm-kind="scan">Time-parallel scan</button><label>CTA tile<select data-tm-tile><option value="32x8">32 time × 8 modes</option><option value="32x16">32 × 16</option><option value="64x8">64 × 8</option></select></label></div><div class="time-mode"><canvas id="time-mode-canvas" data-tm-canvas width="850" height="560" aria-label="时间乘 mode 矩阵"></canvas><aside class="tile-stats" data-tm-stats></aside></div></div>
      <p>旁边的 CTA/warp/job 数是按几何形状计算的 logical schedule，不是 occupancy。当前没有 SAMU mode-tile custom kernel，也没有 registers/thread 数据，因此不能声称哪一块最快。</p>`
  },
  {
    title: "training / prefill backend",
    context: ["large L", "dense projections", "chunk + scan"],
    lead: "训练与 prefill 一次看到完整序列：它们可以批量 GEMM，也有机会沿时间做并行 scan。",
    body: `${tags(["proposed","Backend proposal"],["measured","Prototype measured"])}
      <p>理想路径把 controller 与 dense write 批量计算，构造或融合 per-token transition，再用 chunked prefix 求边界，最后产生 outputs。backward 还需保存或重算中间量，所以 forward 最快配置不一定训练最快。</p>
      <div class="formula">prefill = dense front-end + chunk summary + global prefix + local output<small>当前 SAMU 路径是 audited PyTorch reference/prototype；Mamba-3 对照是官方 fused Triton SISO kernel。</small></div>
      <p>实测 L=2048,B=1,d=128,M=64,BF16：SAMU materialized tree median 5.20 ms，compressed chunk(C=64) 19.82 ms，官方 Mamba-3 1.56 ms。结果明确否定“仅靠压缩 transition，当前实现已经胜过官方 fused kernel”。</p>`
  },
  {
    title: "decode backend",
    context: ["L = 1", "launch dominated", "persistent state is separate design"],
    lead: "autoregressive decode 每次只有一个新 token。没有长 time axis 可 scan，核心问题转为小批量投影、state load/store 与 kernel launch。",
    body: `${tags(["measured","Measured"],["proposed","Kernel proposal"])}
      <p>普通 decode step 从显存读当前 state，算 controller、write、每 mode update，再写回。把 controller/write 在计时外预计算可以隔离 recurrence core，但不能把它冒充 total decode。</p>
      <div class="formula">total step = controller + write projection + recurrence &nbsp;&nbsp; ≠ &nbsp;&nbsp; recurrence-only<small>两者在 raw JSON 中分别标注 decode_direct/factorized 与 decode_precomputed_factorized。</small></div>
      <p>B=1,d=128,M=64,BF16 实测：SAMU direct total median 0.448 ms；precomputed recurrence-only 0.213 ms；官方 RG-LRU PyTorch reference 0.568 ms。官方 Mamba-3 decode 依赖 CuTeDSL step，在该 RTX 3090 环境不支持，故保留 unsupported，不能比较谁赢。</p>
      <div class="misconception"><strong>常见误解</strong>普通多次 kernel invocation 不能让 register state 跨 step 存活。persistent decode 是值得实现的新 backend，但必须单独设计和测量。</div>`
  },
  {
    title: "RG-LRU：独立 gate 与共享坐标的差别",
    context: ["per-channel dynamic gate", "official PyTorch reference", "control dimension ≠ HBM traffic"],
    lead: "RG-LRU 为 channel/mode 形成更独立的动态 gating；SAMU 用 c,d 经过静态 spectrum 产生 coherent response。前者更自由，后者更受约束。",
    body: `${tags(["verified","Official source"],["measured","Reference measured"])}
      <p>比较要分两层。数学层：RG-LRU 的 dynamic quantities 随 channel 展开，SAMU 只有少量共享 coordinates。实现层：fused RG-LRU 可以现场生成 gates、复用 cache，未必 materialize O(LM) gate tensor 到 HBM。</p>
      <div class="breakout" data-figure="compare"><div class="figure-head"><div><span class="figure-number">Interactive 7</span><h3>Same Token, Three Recurrences</h3></div><span class="evidence paper">Structure, not speed</span></div><div class="compare-three" data-compare-cards></div><div class="formula" data-compare-formula>选择模型查看同一 u<sub>t</sub> 经过哪些动态量与 state update。</div></div>
      <p>本站运行的是 Google RecurrentGemma 官方 PyTorch RGLRU source，但该代码包含 Python serial loop，因此标注为 <b>official-source reference</b>，不是 production optimized kernel。它在 L=2048 的 187.9 ms 不能被用来声称 SAMU 架构比 RG-LRU 快 36×。</p>`
  },
  {
    title: "Mamba-3：为什么它是强 GPU baseline？",
    context: ["official fused Triton", "chunked hardware-aware", "implementation reality"],
    lead: "Mamba-3 的官方实现已经把 chunked formulation、matmul-oriented work 与硬件执行路径连接起来；它不是一个可用几行 Python 替代的弱 baseline。",
    body: `${tags(["paper","Official code fact"],["measured","Measured"])}
      <p>本站直接加载 state-spaces/mamba 的官方 Mamba-3 SISO Triton 路径，源码 commit 固定在环境 JSON。Profiler 的代表点仅记录 51 个 CUDA events，主时间落在 fused Mamba3 function 与 SISO forward kernel；对照 SAMU tree prototype 的 1,274 events，差距首先是 kernel maturity。</p>
      <p>SAMU 的研究问题不是预设胜利，而是：更丰富的 mode-specific complex transition，能否保持紧凑 transition representation，并最终达到竞争性的 GPU 效率？当前数据回答是“prototype 尚未达到”。</p>
      <div class="lesson-grid"><div class="meaning"><strong>谁赢</strong>L=128…8192、B=1、BF16 forward 的已测 best-native 点全部由官方 Mamba-3 获得最低 latency。</div><div class="misconception"><strong>不能推出</strong>这不是同等成熟 custom kernels 的架构终局；它也不提供 Mamba-3 decode 数字，因为 RTX 3090 上 CuTe path unsupported。</div></div>
      <div class="breakout" data-figure="transition"><div class="figure-head"><div><span class="figure-number">Interactive 8</span><h3>Transition Representation</h3></div><span class="evidence paper">Formula-level</span></div><div class="control-row"><button data-transition-span="token" class="primary">1 token</button><button data-transition-span="tokens">32 tokens</button><button data-transition-span="chunk">1 chunk</button></div><div class="transition-view" data-transition-view></div></div>`
  },
  {
    title: "真实 benchmark：先看实现等级，再看折线",
    context: ["Track A best-native", "Track B architectural prototype", "matching protocol"],
    lead: "一张曲线只有在回答清楚‘测了谁的什么实现’后才有意义。本页把 best available 与 apples-to-apples 结构实验分开。",
    body: `${tags(["measured","67 configurations"],["verified","Raw JSON retained"])}
      <p id="fairness"><b>Track A</b> 回答当前现实谁最快：官方 Mamba-3 fused Triton、可获得的官方 RG-LRU source、当前最快 SAMU prototype。<b>Track B</b> 使用同一 PyTorch/device/dtype 的 recurrence prototypes，隔离 direct/factorized/tree/chunk 的结构成本；但 competitor 若只有 serial reference，会明确警告。</p>
      <p>三类 matching protocol 同时记录：same model width；按 2M real scalars 计算的 state-byte matching；parameter/compute approximate matching。官方推荐 state structure 与 SAMU complex modes 不能强行一一等同。</p>
      <div class="formula">timing = CUDA events + device synchronization; warmup/compile excluded; median + p10/p90/p95 + raw samples retained<small>显存是 PyTorch peak allocated；DRAM/L2/occupancy/register/TC/SFU/SM counters 因无 Nsight Compute 而为 N/A。</small></div>
      <p>向下进入 Benchmark Lab，可按 workload、dtype、batch 与 metric 过滤，并点击每个点追到原始样本。unsupported 行仍在 coverage 内。</p>`
  },
  {
    title: "结论不是口号，而是一张未完成的研究地图",
    context: ["measured today", "next kernel", "open questions"],
    lead: "当前证据已经足以淘汰几种过强说法，也足以确定下一步工程优先级。",
    body: `${tags(["measured","Measured"],["hypothesis","Open questions"])}
      <p><b>已测结论：</b>在 RTX 3090 上，官方 Mamba-3 fused SISO 在 L=128…32768 的匹配 prefill points 领先或接近当前 SAMU PyTorch prototypes；到 L=65536，SAMU materialized tree 测得 7.54 ms，低于 Mamba-3 的 13.52 ms，出现一个需要复核和 custom-kernel 对照的真实 crossover。SAMU tree 也显著快于 official-source RG-LRU serial reference；但后者不是 optimized kernel，所以这只是 implementation result。SAMU decode total 在该 reference 对照中更低，但 Mamba-3 decode 缺失，结论不完整。</p>
      <p><b>最直接瓶颈：</b>SAMU 目前没有 fused custom kernel。Profiler 显示 tree prototype 有 1,274 个事件、chunk prototype 1,476 个；phase factorization 在 PyTorch 中反而因额外 operators 变慢。下一版最值得实现的是一个 fused warp-shared-control forward kernel：c,d 一次生成/广播，mode state lane-local，sincos/exp 与复数 FMA 融合，并记录 registers、occupancy、DRAM/L2。</p>
      <div class="lesson-grid"><div class="meaning"><strong>继续验证</strong>serial vs chunk crossover；C=16 是否在 fused kernel 仍最优；precompute vs recompute；mode tile；SFU 与 register pressure。</div><div class="misconception"><strong>仍是 hypothesis</strong>coherent control 会带来更好的 quality-efficiency frontier；SAMU 能在任何 regime 击败成熟 Mamba-3 kernel。</div></div>
      <p>课程到这里不要求相信 SAMU 更快，只要求你现在能指出：一次 timestep 做什么、数据可能停在哪里、为什么 write 与 recurrence 不同、scan 怎样成立，以及每条性能结论究竟来自公式、代码还是测量。</p>`
  }
];
