const css = (name) => getComputedStyle(document.body).getPropertyValue(name).trim();
const fmt = (v, digits = 2) =>
  v === null || v === undefined ? "—" : v.toLocaleString("ru-RU", { minimumFractionDigits: digits, maximumFractionDigits: digits });

let data = null;
let frame = 0;

fetch("data.json")
  .then((r) => r.json())
  .then((loaded) => {
    data = loaded;
    start();
  })
  .catch(() => {
    document.getElementById("subtitle").textContent =
      "не найден web/data.json — соберите его командой python -m hypermm.replay_ui.export";
  });

function start() {
  const { coin, date, hour } = data.meta;
  const pretty = `${date.slice(6, 8)}.${date.slice(4, 6)}.${date.slice(0, 4)}`;
  document.getElementById("subtitle").textContent =
    `${coin} · ${pretty}, ${String(hour).padStart(2, "0")}:00–${String(hour + 1).padStart(2, "0")}:00 UTC`;

  renderStats();
  renderQueue();

  const slider = document.getElementById("frame");
  slider.max = data.frames.length - 1;
  slider.addEventListener("input", () => setFrame(+slider.value));
  setFrame(Math.floor(data.frames.length / 2));
  slider.value = frame;

  window.addEventListener("resize", () => {
    drawMid();
    drawQueue();
  });
}

function setFrame(index) {
  frame = index;
  document.getElementById("frame-label").textContent = `${fmt(data.series.second[frame], 1)} с`;
  document.getElementById("ladder-time").textContent = `на ${fmt(data.series.second[frame], 1)}-й секунде часа`;
  renderLadder();
  drawMid();
}

/* ── сводка ─────────────────────────────────────────────────────────────── */

function renderStats() {
  const s = data.stats;
  const counts = Object.fromEntries(data.outcomes.map((o) => [o.outcome, o.orders]));
  const share = (n) => `${fmt((100 * n) / s.orders, 2)}<small> %</small>`;

  const tiles = [
    ["Заявок за час", s.orders.toLocaleString("ru-RU")],
    ["Изменений стакана", s.changes.toLocaleString("ru-RU")],
    ["Хоть раз исполнились", `${fmt(s.fillRate, 2)}<small> %</small>`],
    ["Отменены", share(counts.cancelled ?? 0)],
    ["Медиана до исполнения", `${fmt(s.secondsToFill, 2)}<small> с</small>`],
    ["Медиана до отмены", `${fmt(s.secondsToCancel, 2)}<small> с</small>`],
    ["Медианный спред", fmt(s.medianSpread, 3)],
    ["Медиана объёма впереди", fmt(s.medianAhead, 0)],
  ];
  document.getElementById("stats").innerHTML = tiles
    .map(([term, value]) => `<dl class="tile"><dt>${term}</dt><dd>${value}</dd></dl>`)
    .join("");
}

/* ── стакан ─────────────────────────────────────────────────────────────── */

function renderLadder() {
  const { scale } = data.meta;
  const { bids, asks } = data.frames[frame];
  const peak = Math.max(1, ...bids.map((b) => b[1]), ...asks.map((a) => a[1]));
  const rung = (row, side) => {
    const width = (100 * row[1]) / peak;
    return `<div class="rung ${side}">
      <span class="px">${fmt(row[0] / scale, 2)}</span>
      <span><span class="bar" style="width:${width.toFixed(1)}%"></span></span>
      <span class="size">${fmt(row[1] / scale, 2)}</span>
    </div>`;
  };

  const best = bids[0] && asks[0] ? (asks[0][0] - bids[0][0]) / scale : null;
  document.getElementById("ladder").innerHTML =
    asks.slice().reverse().map((a) => rung(a, "ask")).join("") +
    `<div class="gap"><span>спред</span><span class="mono">${fmt(best, 3)}</span></div>` +
    bids.map((b) => rung(b, "bid")).join("");
}

/* ── середина цены ──────────────────────────────────────────────────────── */

function drawMid() {
  const canvas = document.getElementById("mid");
  const { ctx, width, height } = prepare(canvas);
  const pad = { left: 54, right: 10, top: 10, bottom: 24 };
  const values = data.series.mid;
  const seconds = data.series.second;
  const known = values.filter((v) => v !== null);
  const lo = Math.min(...known);
  const hi = Math.max(...known);

  const x = (i) => pad.left + ((width - pad.left - pad.right) * i) / (values.length - 1);
  const y = (v) => pad.top + (height - pad.top - pad.bottom) * (1 - (v - lo) / (hi - lo || 1));

  grid(ctx, width, height, pad, lo, hi, (v) => fmt(v, 2));

  ctx.strokeStyle = css("--bid");
  ctx.lineWidth = 2;
  ctx.lineJoin = "round";
  ctx.beginPath();
  let pen = false;
  values.forEach((v, i) => {
    if (v === null) { pen = false; return; }
    pen ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v));
    pen = true;
  });
  ctx.stroke();

  if (values[frame] !== null) {
    ctx.strokeStyle = css("--text-muted");
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x(frame), pad.top);
    ctx.lineTo(x(frame), height - pad.bottom);
    ctx.stroke();
    ctx.fillStyle = css("--bid");
    ctx.beginPath();
    ctx.arc(x(frame), y(values[frame]), 4, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.fillStyle = css("--text-muted");
  ctx.font = "11px ui-monospace, monospace";
  ctx.textAlign = "left";
  ctx.fillText("0 с", pad.left, height - 8);
  ctx.textAlign = "right";
  ctx.fillText(`${Math.round(seconds.at(-1))} с`, width - pad.right, height - 8);

  hover(canvas, (position) => {
    const i = Math.round(((position - pad.left) / (width - pad.left - pad.right)) * (values.length - 1));
    if (i < 0 || i >= values.length || values[i] === null) return null;
    return { x: x(i), y: y(values[i]), text: `${fmt(seconds[i], 1)} с · ${fmt(values[i], 2)}` };
  }, "mid-tip");
}

/* ── исполняемость по очереди ───────────────────────────────────────────── */

function renderQueue() {
  document.getElementById("queue-table").innerHTML =
    `<tr><th>дециль</th><th>объём впереди</th><th>заявок</th><th>исполнилось, %</th></tr>` +
    data.queue
      .map((d) => `<tr><td>${d.bucket}</td><td>${fmt(d.ahead, 2)}</td><td>${d.orders.toLocaleString("ru-RU")}</td><td>${fmt(d.fillRate, 2)}</td></tr>`)
      .join("");
  drawQueue();
}

function drawQueue() {
  const canvas = document.getElementById("queue");
  const { ctx, width, height } = prepare(canvas);
  const pad = { left: 54, right: 10, top: 14, bottom: 42 };
  const bars = data.queue;
  const hi = Math.max(...bars.map((b) => b.fillRate));
  const span = (width - pad.left - pad.right) / bars.length;
  const y = (v) => pad.top + (height - pad.top - pad.bottom) * (1 - v / hi);

  grid(ctx, width, height, pad, 0, hi, (v) => `${fmt(v, 1)} %`);

  ctx.fillStyle = css("--bid");
  ctx.font = "11px ui-monospace, monospace";
  bars.forEach((bar, i) => {
    const left = pad.left + i * span + 2;
    const wide = span - 6;
    const top = y(bar.fillRate);
    ctx.fillStyle = css("--bid");
    round(ctx, left, top, wide, height - pad.bottom - top, 4);
    ctx.fillStyle = css("--text-secondary");
    ctx.textAlign = "center";
    ctx.fillText(fmt(bar.fillRate, 1), left + wide / 2, top - 5);
    ctx.fillStyle = css("--text-muted");
    ctx.fillText(fmt(bar.ahead, 0), left + wide / 2, height - pad.bottom + 15);
  });

  ctx.fillStyle = css("--text-muted");
  ctx.textAlign = "center";
  ctx.fillText("объём, стоявший впереди в момент постановки →", width / 2, height - 8);

  hover(canvas, (position) => {
    const i = Math.floor((position - pad.left) / span);
    if (i < 0 || i >= bars.length) return null;
    const bar = bars[i];
    return {
      x: pad.left + (i + 0.5) * span,
      y: y(bar.fillRate),
      text: `впереди ${fmt(bar.ahead, 0)} · ${bar.orders.toLocaleString("ru-RU")} заявок · ${fmt(bar.fillRate, 2)} %`,
    };
  }, "queue-tip");
}

/* ── общее ──────────────────────────────────────────────────────────────── */

function prepare(canvas) {
  const ratio = window.devicePixelRatio || 1;
  // canvas.height — отражаемый (reflected) атрибут: присвоение canvas.height
  // ниже переписывает и HTML-атрибут height. Если брать высоту из
  // getAttribute("height") при каждой перерисовке, при ratio > 1 (Retina)
  // она удваивается на каждый вызов — 260 → 520 → 1040 → …, и график
  // за несколько перерисовок (скрабинг ползунка, resize) уезжает вниз за
  // пределы панели. Поэтому CSS-высоту фиксируем один раз и дальше её не
  // трогаем; резолюшн буфера пересчитываем от неё, а не от атрибута.
  if (!canvas.dataset.cssHeight) {
    canvas.dataset.cssHeight = canvas.getAttribute("height");
    canvas.style.height = `${canvas.dataset.cssHeight}px`;
  }
  const width = canvas.clientWidth;
  const height = +canvas.dataset.cssHeight;
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  canvas.style.height = `${height}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  return { ctx, width, height };
}

function grid(ctx, width, height, pad, lo, hi, label) {
  ctx.strokeStyle = css("--grid");
  ctx.fillStyle = css("--text-muted");
  ctx.lineWidth = 1;
  ctx.font = "11px ui-monospace, monospace";
  ctx.textAlign = "right";
  for (let step = 0; step <= 4; step++) {
    const value = lo + ((hi - lo) * step) / 4;
    const at = Math.round(pad.top + (height - pad.top - pad.bottom) * (1 - step / 4)) + 0.5;
    ctx.beginPath();
    ctx.moveTo(pad.left, at);
    ctx.lineTo(width - pad.right, at);
    ctx.stroke();
    ctx.fillText(label(value), pad.left - 8, at + 4);
  }
}

function round(ctx, x, y, w, h, r) {
  const radius = Math.min(r, w / 2, Math.max(h, 0));
  ctx.beginPath();
  ctx.moveTo(x, y + h);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
  ctx.lineTo(x + w - radius, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + radius);
  ctx.lineTo(x + w, y + h);
  ctx.closePath();
  ctx.fill();
}

function hover(canvas, locate, tipId) {
  const tip = document.getElementById(tipId);
  canvas.onmousemove = (event) => {
    const box = canvas.getBoundingClientRect();
    const found = locate(event.clientX - box.left);
    if (!found) { tip.hidden = true; return; }
    tip.hidden = false;
    tip.textContent = found.text;
    tip.style.left = `${Math.min(found.x + 12, canvas.clientWidth - tip.offsetWidth)}px`;
    tip.style.top = `${found.y + canvas.offsetTop - 8}px`;
  };
  canvas.onmouseleave = () => { tip.hidden = true; };
}
