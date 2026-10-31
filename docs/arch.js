// The Architecture section: one kernel through every stage of cimo (arch/arch.json, real
// compiler output from arch/gen.py), as a stepper with the stage's output beside it.
(function () {
  'use strict';
  const root = document.getElementById('arch');
  if (!root) return;
  const steps = root.querySelector('.arch-steps'), code = root.querySelector('.arch-code pre');
  const tab = root.querySelector('.arch-file'), title = root.querySelector('.arch-title');
  const origin = root.querySelector('.arch-origin'), desc = root.querySelector('.arch-desc');
  const stat = root.querySelector('.arch-stat'), alt = root.querySelector('.arch-alt');

  const esc = t => t.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  function tokens(text, re, cls) {
    let out = '', last = 0;
    text.replace(re, (m, ...g) => {
      const at = g[g.length - 2];
      out += esc(text.slice(last, at));
      const k = g.findIndex((x, i) => i < cls.length && x !== undefined);
      out += k < 0 || !cls[k] ? esc(m) : `<span class="t-${cls[k]}">${esc(m)}</span>`;
      last = at + m.length;
      return m;
    });
    return out + esc(text.slice(last));
  }
  const ASM = /(^\s*[.\w]+:)|(?<=^\s*)([a-z][a-z0-9.]*)\b|\b([xwsdvq]\d{1,2}|sp|xzr|wzr|lr)\b|(#-?\d+|\b-?\d+\b)|(\.L\w+)/gm;
  const HL = {
    c: t => tokens(t, /(\/\*[\s\S]*?\*\/|\/\/[^\n]*)|\b(for|if|return|const)\b|\b(void|int|float)\b|(\b\d+\b)|\b([A-Za-z_]\w*)(?=\s*\()/g, ['cm', 'kw', 'ty', 'num', 'fn']),
    qbe: t => tokens(t, /(#[^\n]*)|\b(export|function|data|type|phi|jnz|jmp|ret|call|par)\b|(@[\w.]+)|(\$\w+)|(%[\w.]+)|(=[wlsd]\b)|\b(load\w*|store\w*|alloc\d+|add|sub|mul|div|and|or|shl|shr|sar|ext\w+|c[a-z]+[wlsd]|fma|copy)\b|(-?\b\d+\b)/g, ['cm', 'kw', 'fn', 'fn', 'op', 'ty', 'kw', 'num']),
    asm: t => tokens(t, ASM, ['fn', 'kw', 'op', 'num', 'fn']),
    hex: t => t.split('\n').map(l => {
      const m = /^(\w{6})  (\w{8})   (.*)$/.exec(l);
      return m ? `<span class="t-cm">${m[1]}</span>  <span class="t-num">${m[2]}</span>   ${HL.asm(m[3])}` : HL.asm(l);
    }).join('\n'),
    wasm: t => tokens(t, /(;;[^\n]*)|\b((?:i32|i64|f32|f64)\.[a-z0-9_]+|local\.(?:get|set|tee)|global\.(?:get|set)|memory\.\w+)\b|\b(func|param|result|local|export)\b|(\$[\w.]+)|("[^"]*")|\b(block|loop|if|else|end|br_if|br_table|br|return|call|unreachable|drop|select)\b|(-?\b\d+(?:\.\d+)?\b)/g, ['cm', 'op', 'ty', 'fn', 'str', 'kw', 'num']),
    py: t => t.split('\n').map(l => {
      if (l.startsWith('#')) return `<span class="t-cm">${esc(l)}</span>`;
      if (!l.startsWith('>>>')) return `<span class="t-out">${esc(l)}</span>`;
      return '<span class="t-pr">&gt;&gt;&gt;</span>' + tokens(l.slice(3), /(#[^\n]*)|("[^"]*")|\b(import|as)\b|(\b\d+(?:\.\d+)?\b)|\b([A-Za-z_]\w*)(?=\()/g, ['cm', 'str', 'kw', 'num', 'fn']);
    }).join('\n'),
  };
  const FILES = { c: 'saxpy.c', ir: 'saxpy.ssa · from the front end', opt: 'saxpy.ssa · optimized', asm: 'saxpy.s · arm64', enc: 'saxpy · machine code', run: 'python' };

  let data = null, cur = 0, showAlt = false, timer = null, auto = true, inView = false;

  function show(k, user) {
    cur = (k + data.length) % data.length;
    if (user) { auto = false; clearTimeout(timer); }
    const s = data[cur];
    steps.querySelectorAll('.arch-step').forEach((b, i) => {
      b.classList.toggle('on', i === cur);
      b.classList.toggle('done', i < cur);
      b.setAttribute('aria-selected', i === cur);
    });
    steps.style.setProperty('--p', cur / (data.length - 1));
    const v = s.alt && showAlt ? s.alt : s;
    title.textContent = s.title;
    origin.textContent = s.origin;
    origin.className = 'arch-origin o-' + s.origin.replace(/[^a-z]/gi, '').toLowerCase();
    desc.textContent = s.desc;
    stat.textContent = v.stat;
    tab.textContent = s.alt && showAlt ? 'saxpy.wasm · as text' : FILES[s.id] || s.id;
    alt.hidden = !s.alt;
    if (s.alt) alt.querySelectorAll('button').forEach(b => b.classList.toggle('on', (b.dataset.v === 'alt') === showAlt));
    code.className = 'lang-' + v.lang;
    code.innerHTML = HL[v.lang](v.text);
    code.scrollTop = 0;
    code.parentNode.classList.remove('swap'); void code.offsetWidth; code.parentNode.classList.add('swap');
    if (auto) { clearTimeout(timer); timer = setTimeout(tick, 6000); }
  }
  function tick() { if (auto && inView && !root.matches(':hover')) show(cur + 1); else timer = setTimeout(tick, 1500); }

  fetch('arch/arch.json').then(r => r.json()).then(j => {
    data = j.stages;
    steps.innerHTML = '<i class="arch-line"><b></b></i>' + data.map((s, i) =>
      `<button type="button" class="arch-step" role="tab" data-i="${i}"><span class="n">${i + 1}</span><b>${s.title}</b><small class="o-${s.origin.replace(/[^a-z]/gi, '').toLowerCase()}">${s.origin}</small></button>`).join('');
    steps.addEventListener('click', e => { const b = e.target.closest('.arch-step'); if (b) show(+b.dataset.i, true); });
    alt.addEventListener('click', e => { const b = e.target.closest('button'); if (!b) return; showAlt = b.dataset.v === 'alt'; show(cur, true); });
    root.addEventListener('keydown', e => {
      if (e.key === 'ArrowRight') { show(cur + 1, true); e.preventDefault(); }
      if (e.key === 'ArrowLeft') { show(cur - 1, true); e.preventDefault(); }
    });
    if (matchMedia('(prefers-reduced-motion:reduce)').matches) auto = false;
    new IntersectionObserver(es => { inView = es[0].isIntersecting; }, { threshold: 0.3 }).observe(root);
    const pick = /[?&]arch=(\d)/.exec(location.search);  // (?arch=N: for screenshots)
    show(pick ? +pick[1] : 0, !!pick);
  }).catch(() => { desc.textContent = 'Could not load the stages.'; });
})();
