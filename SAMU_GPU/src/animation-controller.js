export class AnimationController {
  constructor({steps = 1, duration = 1100, onFrame = () => {}, onStep = () => {}} = {}) {
    this.steps = steps;
    this.duration = duration;
    this.onFrame = onFrame;
    this.onStep = onStep;
    this.stepIndex = 0;
    this.progress = 0;
    this.speed = 1;
    this.playing = !matchMedia("(prefers-reduced-motion: reduce)").matches;
    this.last = performance.now();
    this.raf = 0;
    this.tick = this.tick.bind(this);
    this.raf = requestAnimationFrame(this.tick);
  }

  tick(now) {
    const dt = Math.min(80, now - this.last);
    this.last = now;
    if (this.playing) {
      this.progress += dt * this.speed / this.duration;
      if (this.progress >= 1) {
        this.progress %= 1;
        this.stepIndex = (this.stepIndex + 1) % this.steps;
        this.onStep(this.stepIndex);
      }
    }
    this.onFrame(this.stepIndex, ease(this.progress));
    this.raf = requestAnimationFrame(this.tick);
  }

  toggle() { this.playing = !this.playing; return this.playing; }
  next() { this.stepIndex = (this.stepIndex + 1) % this.steps; this.progress = 0; this.onStep(this.stepIndex); }
  back() { this.stepIndex = (this.stepIndex - 1 + this.steps) % this.steps; this.progress = 0; this.onStep(this.stepIndex); }
  reset() { this.stepIndex = 0; this.progress = 0; this.onStep(0); }
  setSpeed(value) { this.speed = Number(value) || 1; }
  stop() { cancelAnimationFrame(this.raf); }
}

export const ease = x => x < .5 ? 2*x*x : 1 - Math.pow(-2*x+2, 2)/2;
export const lerp = (a,b,t) => a + (b-a)*t;
export const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

export function fitCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(2, devicePixelRatio || 1);
  const width = Math.max(320, Math.round(rect.width));
  const height = Math.max(220, Math.round(rect.height || width * .5));
  if (canvas.width !== width*dpr || canvas.height !== height*dpr) {
    canvas.width = width*dpr; canvas.height = height*dpr;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr,0,0,dpr,0,0);
  return {ctx,width,height,dpr};
}

export function roundedRect(ctx,x,y,w,h,r=7) {
  ctx.beginPath(); ctx.roundRect(x,y,w,h,r); return ctx;
}

export function bindTransport(root, controller, labels = {play:"播放", pause:"暂停"}) {
  if (!root) return;
  const toggle = root.querySelector('[data-action="toggle"], [data-step-action="toggle"], [data-scan-action="toggle"]');
  const update = () => { if (toggle) toggle.textContent = controller.playing ? labels.pause : labels.play; };
  root.addEventListener("click", e => {
    const action = e.target.dataset.action || e.target.dataset.stepAction || e.target.dataset.scanAction;
    if (action === "toggle") controller.toggle();
    if (action === "step") controller.next();
    if (action === "back") controller.back();
    if (action === "reset") controller.reset();
    update();
  });
  root.addEventListener("change", e => {
    if (e.target.matches('[data-action="speed"], [data-step-speed], [data-scan-speed]')) controller.setSpeed(e.target.value);
  });
  update();
}

export function activateWhenVisible(root, controller) {
  if (!root || matchMedia("(prefers-reduced-motion: reduce)").matches) {
    controller.playing = false;
    return;
  }
  const observer = new IntersectionObserver(entries => {
    const visible = entries.some(entry => entry.isIntersecting && entry.intersectionRatio > .12);
    controller.playing = visible;
    const toggle = root.querySelector('[data-action="toggle"], [data-step-action="toggle"], [data-scan-action="toggle"]');
    if (toggle) toggle.textContent = visible ? "暂停" : "播放";
  }, {threshold:[0,.12,.5]});
  observer.observe(root);
}
