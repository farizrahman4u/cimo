// The Architecture section: one kernel through every stage of cimo (arch/arch.json, real
// compiler output from arch/gen.py), as a stepper with the stage's output beside it.
(function () {
  'use strict';
  const root = document.getElementById('arch');
  if (!root) return;
  const steps = root.querySelector('.arch-steps'), view = root.querySelector('.arch-view');

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
  // one block per source line (CSS numbers them); a highlight span crossing a line break is
  // closed at the end of the line and reopened on the next
  function numbered(h) {
    let open = null;
    return h.replace(/\n$/, '').split('\n').map(line => {
      const head = open ? `<span class="${open}">` : '';
      const re = /<span class="([^"]+)">|<\/span>/g;
      let m;
      while ((m = re.exec(line))) open = m[1] || null;
      return `<span class="ln">${head}${line}${open ? '</span>' : ''}</span>`;
    }).join('');
  }
  const FILES = { c: 'saxpy.c', ir: 'saxpy.ssa · from the front end', opt: 'saxpy.ssa · optimized', asm: 'saxpy.s · arm64', enc: 'saxpy · machine code', run: 'python' };
  const slug = o => o.replace(/[^a-z]/gi, '').toLowerCase();

  let data = null, cur = 0, timer = null, auto = true, inView = false, slides = [];
  const RM = matchMedia('(prefers-reduced-motion:reduce)').matches;

  // every slide is rendered once, side by side in a track; changing stage only moves the track
  function slideHTML(s) {
    // line numbers on every listing except the machine code, which carries its own offsets
    const v = (x, alt) => `<pre class="lang-${x.lang}${alt ? ' alt' : ''}${x.lang === 'hex' ? '' : ' numbered'}"${alt ? ' hidden' : ''}>${x.lang === 'hex' ? HL.hex(x.text) : numbered(HL[x.lang](x.text))}</pre>`;
    return `<div class="arch-slide" role="tabpanel">
      <div class="arch-text">
        <div class="arch-head"><h3>${s.title}</h3><span class="arch-origin o-${slug(s.origin)}">${s.origin}</span></div>
        <p class="arch-desc">${esc(s.desc)}</p>
        <span class="arch-stat">${s.stat}</span>
        ${s.alt ? '<div class="arch-alt"><button type="button" data-v="main" class="on">arm64</button><button type="button" data-v="alt">WebAssembly</button></div>' : ''}
      </div>
      <div class="code arch-code"><div class="bar"><span class="dots"><i></i><i></i><i></i></span><span class="arch-file">${FILES[s.id] || s.id}</span></div>${v(s)}${s.alt ? v(s.alt, true) : ''}</div>
    </div>`;
  }

  // coverflow: the current card in front, its neighbours peeking in at the sides, smaller,
  // turned away and dimmed; frac shifts everything during a drag (in cards)
  function layout(frac, animate) {
    const w = slides[0].offsetWidth, step = w * .94;
    slides.forEach((el, i) => {
      const off = i - cur + frac, a = Math.min(Math.abs(off), 1.6);
      el.style.transition = animate && !RM ? 'transform .5s cubic-bezier(.3,.7,.2,1), opacity .5s ease' : 'none';
      el.style.transform = `translateX(calc(-50% + ${off * step}px)) rotateY(${Math.max(-1.2, Math.min(1.2, off)) * -14}deg) scale(${1 - a * .08})`;
      el.style.opacity = a > 1.5 ? 0 : 1 - Math.min(a, 1) * .5;
      el.style.zIndex = 10 - Math.round(a * 4);
      el.classList.toggle('side', Math.abs(off) > .5);
    });
  }
  function fit() {  // the stage is as tall as the tallest card
    view.style.height = Math.max(...slides.map(el => el.offsetHeight)) + 'px';
  }

  function show(k, user, animate = true) {
    cur = Math.max(0, Math.min(data.length - 1, k));
    if (user) { auto = false; clearTimeout(timer); }
    steps.querySelectorAll('.arch-step').forEach((b, i) => {
      b.classList.toggle('on', i === cur);
      b.classList.toggle('done', i < cur);
      b.setAttribute('aria-selected', i === cur);
    });
    steps.style.setProperty('--p', cur / (data.length - 1));
    slides.forEach((el, i) => el.setAttribute('aria-hidden', i !== cur));
    layout(0, animate);
    if (auto) { clearTimeout(timer); timer = setTimeout(tick, 6000); }
  }
  function tick() {
    if (auto && inView && !root.matches(':hover')) show((cur + 1) % data.length);
    else timer = setTimeout(tick, 1500);
  }

  fetch('arch/arch.json').then(r => r.json()).then(j => {
    data = j.stages;
    steps.innerHTML = '<i class="arch-line"><b></b></i>' + data.map((s, i) =>
      `<button type="button" class="arch-step" role="tab" data-i="${i}"><span class="n">${i + 1}</span><b>${s.title}</b><small class="o-${slug(s.origin)}">${s.origin}</small></button>`).join('');
    view.innerHTML = data.map(slideHTML).join('');
    slides = [...view.querySelectorAll('.arch-slide')];
    fit();
    addEventListener('resize', () => { fit(); layout(0, false); });
    steps.addEventListener('click', e => { const b = e.target.closest('.arch-step'); if (b) show(+b.dataset.i, true); });
    view.addEventListener('click', e => {
      const side = e.target.closest('.arch-slide.side');  // a peeking card: bring it forward
      if (side && !moved) { show(slides.indexOf(side), true); return; }
      const b = e.target.closest('.arch-alt button');  // the encoder's arm64 / WebAssembly toggle
      if (!b) return;
      const slide = b.closest('.arch-slide'), alt = b.dataset.v === 'alt', s = data[cur];
      slide.querySelectorAll('.arch-alt button').forEach(x => x.classList.toggle('on', x === b));
      slide.querySelector('pre:not(.alt)').hidden = alt;
      slide.querySelector('pre.alt').hidden = !alt;
      slide.querySelector('.arch-stat').textContent = alt ? s.alt.stat : s.stat;
      slide.querySelector('.arch-file').textContent = alt ? 'saxpy.wasm · as text' : FILES[s.id];
      auto = false; clearTimeout(timer);
    });
    root.addEventListener('keydown', e => {
      if (e.key === 'ArrowRight') { show(cur + 1, true); e.preventDefault(); }
      if (e.key === 'ArrowLeft') { show(cur - 1, true); e.preventDefault(); }
    });

    // drag: the track follows the pointer, so the next slide comes in as you pull; a drag
    // only counts once it is clearly sideways, so vertical scrolling in the code still works
    let d = null, moved = false;
    view.addEventListener('pointerdown', e => {
      if (e.button || e.target.closest('button, a')) return;
      moved = false;
      d = { x: e.clientX, y: e.clientY, t: e.timeStamp, id: e.pointerId, on: false, dx: 0 };
    });
    view.addEventListener('pointermove', e => {
      if (!d || e.pointerId !== d.id) return;
      const dx = e.clientX - d.x, dy = e.clientY - d.y;
      if (!d.on) {
        if (Math.abs(dy) > 8 && Math.abs(dy) > Math.abs(dx)) { d = null; return; }
        if (Math.abs(dx) < 8) return;
        d.on = true; moved = true;
        view.setPointerCapture(d.id);
        view.classList.add('dragging');
        window.getSelection().removeAllRanges();
        auto = false; clearTimeout(timer);
      }
      const end = (dx > 0 && cur === 0) || (dx < 0 && cur === data.length - 1);
      d.dx = end ? dx / 3 : dx;
      layout(d.dx / (slides[0].offsetWidth * .94), false);
    });
    const release = e => {
      if (!d) return;
      const was = d; d = null;
      if (!was.on) return;
      view.classList.remove('dragging');
      const w = slides[0].offsetWidth, v = Math.abs(was.dx) / Math.max(1, e.timeStamp - was.t);
      let next = cur;
      if (Math.abs(was.dx) > w * .18 || (v > .45 && Math.abs(was.dx) > 20)) next = cur + (was.dx < 0 ? 1 : -1);
      show(next, true);
    };
    view.addEventListener('pointerup', release);
    view.addEventListener('pointercancel', release);

    if (RM) auto = false;
    new IntersectionObserver(es => { inView = es[0].isIntersecting; }, { threshold: 0.3 }).observe(root);
    const pick = /[?&]arch=(\d)/.exec(location.search);  // (?arch=N: for screenshots)
    show(pick ? +pick[1] : 0, !!pick, false);
  }).catch(() => { view.textContent = 'Could not load the stages.'; });
})();
