"""Self-contained HTML/JS canvas animation of a recorded robot trajectory."""
import json

from grid_sim import SENSOR_OFFSETS

TEMPLATE = r"""
<div id="card" style="font-family:system-ui,-apple-system,Segoe UI,sans-serif;color:#222;background:#fff;
     border:1px solid #d9d9d9;border-radius:10px;padding:12px;max-width:100%;box-sizing:border-box">
  <canvas id="c" style="display:block;max-width:100%;border:1px solid #e3e3e3;border-radius:6px"></canvas>
  <div style="display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:10px 0 6px">
    <button id="play" style="padding:6px 12px;cursor:pointer">Pause</button>
    <button id="step" style="padding:6px 12px;cursor:pointer">Step &#9654;</button>
    <button id="reset" style="padding:6px 12px;cursor:pointer">Restart</button>
    <label style="font-size:13px">Speed <input id="speed" type="range" min="1" max="10" value="6"></label>
    <input id="scrub" type="range" min="0" max="0" value="0" style="flex:1;min-width:160px">
  </div>
  <div id="hud" style="font-size:13px;line-height:1.5"></div>
  <div style="font-size:12px;color:#555;margin-top:4px">
    Rays: <b style="color:#e4572e">front</b> &middot; <b style="color:#2e86ab">left</b> &middot;
    <b style="color:#3bb273">right</b> &middot; <b style="color:#888">back</b>.
    Red flash = bumped into a wall.
  </div>
</div>
<script>
const D = __DATA__;
const S = D.steps, n = S.length, CP = D.cp;
const cv = document.getElementById('c'), ctx = cv.getContext('2d');
cv.width = D.w * CP; cv.height = D.h * CP;
const RAY_COL = {front:'#e4572e', left:'#2e86ab', right:'#3bb273', back:'#888888'};
const ACT_COL = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02','#a6761d','#666666'];
const OFF = D.offsets;
let i = 0, p = 0, playing = true, last = performance.now();
const scrub = document.getElementById('scrub'); scrub.max = n - 1;

function esc(t) { return String(t).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function lerp(a, b, t) { return a + (b - a) * t; }
function stepMs() { return 1100 - 100 * Number(document.getElementById('speed').value); }

function draw(i, p) {
  const a = S[i], b = S[Math.min(i + 1, n - 1)];
  ctx.clearRect(0, 0, cv.width, cv.height);
  ctx.fillStyle = '#f7f7f4'; ctx.fillRect(0, 0, cv.width, cv.height);
  // visited cells (accumulates -> heat effect)
  ctx.fillStyle = 'rgba(46,134,171,0.13)';
  for (let j = 0; j <= i; j++) ctx.fillRect(S[j].x * CP, S[j].y * CP, CP, CP);
  // grid lines
  ctx.strokeStyle = '#e1e1dc'; ctx.lineWidth = 1; ctx.beginPath();
  for (let x = 0; x <= D.w; x++) { ctx.moveTo(x * CP + .5, 0); ctx.lineTo(x * CP + .5, D.h * CP); }
  for (let y = 0; y <= D.h; y++) { ctx.moveTo(0, y * CP + .5); ctx.lineTo(D.w * CP, y * CP + .5); }
  ctx.stroke();
  // walls
  ctx.fillStyle = '#3d4451';
  for (let y = 0; y < D.h; y++) for (let x = 0; x < D.w; x++) if (D.walls[y][x]) ctx.fillRect(x * CP, y * CP, CP, CP);
  // pose (interpolated)
  const px = (lerp(a.x, b.x, p) + 0.5) * CP, py = (lerp(a.y, b.y, p) + 0.5) * CP;
  const dh = ((b.h - a.h + 4) % 8 + 8) % 8 - 4;
  const ang = (a.h + dh * p) * Math.PI / 4;
  // trail
  ctx.strokeStyle = 'rgba(228,87,46,0.55)'; ctx.lineWidth = 2; ctx.beginPath();
  for (let j = 0; j <= i; j++) { const X = (S[j].x + .5) * CP, Y = (S[j].y + .5) * CP; j ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); }
  ctx.lineTo(px, py); ctx.stroke();
  // sensor rays
  ctx.font = '11px system-ui, sans-serif'; ctx.textAlign = 'center';
  ['front', 'left', 'right', 'back'].forEach((k, idx) => {
    const L = lerp(a.c[idx], b.c[idx], p) * CP;
    const th = ang + OFF[k] * Math.PI / 4;
    const ex = px + Math.sin(th) * L, ey = py - Math.cos(th) * L;
    ctx.strokeStyle = RAY_COL[k]; ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(ex, ey); ctx.stroke(); ctx.setLineDash([]);
    const m = lerp(a.m[idx], b.m[idx], p);
    const lx = px + Math.sin(th) * Math.min(L, CP * 1.1), ly = py - Math.cos(th) * Math.min(L, CP * 1.1);
    ctx.fillStyle = RAY_COL[k]; ctx.fillText(m.toFixed(1), lx, ly - 4);
  });
  // robot
  ctx.save(); ctx.translate(px, py); ctx.rotate(ang);
  const r = CP * 0.36;
  ctx.fillStyle = '#111'; ctx.beginPath();
  ctx.moveTo(0, -r); ctx.lineTo(r * .8, r * .8); ctx.lineTo(-r * .8, r * .8); ctx.closePath(); ctx.fill();
  ctx.restore();
  if (a.x_ && p < 0.7) { ctx.strokeStyle = 'rgba(214,39,40,' + (1 - p / 0.7) + ')'; ctx.lineWidth = 4;
    ctx.beginPath(); ctx.arc(px, py, CP * 0.5, 0, 2 * Math.PI); ctx.stroke(); }
}

function hud(i) {
  const a = S[i], col = ACT_COL[a.ai % ACT_COL.length];
  const bumps = S.slice(0, i + 1).filter(s => s.x_).length;
  document.getElementById('hud').innerHTML =
    'Step <b>' + i + '</b> / ' + (n - 1) + ' &nbsp;|&nbsp; heading <b>' + D.hnames[a.h] + '</b>' +
    ' &nbsp;|&nbsp; state <b>#' + a.s + '</b> (value ' + a.v.toFixed(2) + ')' +
    ' &nbsp;|&nbsp; collisions <b>' + bumps + '</b><br>' +
    'Policy action: <span style="background:' + col + ';color:#fff;padding:1px 8px;border-radius:10px">' + esc(a.a) + '</span>' +
    ' &nbsp; front ' + a.m[0].toFixed(2) + ' m, left ' + a.m[1].toFixed(2) + ' m, right ' +
    a.m[2].toFixed(2) + ' m, back ' + a.m[3].toFixed(2) + ' m';
}

function tick(now) {
  const dt = now - last; last = now;
  if (playing && i < n - 1) {
    p += dt / stepMs();
    while (p >= 1 && i < n - 1) { p -= 1; i++; }
    if (i >= n - 1) { p = 0; playing = false; document.getElementById('play').textContent = 'Play'; }
  }
  draw(i, p); hud(i); scrub.value = i;
  requestAnimationFrame(tick);
}

document.getElementById('play').onclick = function () {
  if (i >= n - 1) { i = 0; p = 0; }
  playing = !playing; this.textContent = playing ? 'Pause' : 'Play';
};
document.getElementById('step').onclick = function () {
  playing = false; document.getElementById('play').textContent = 'Play';
  i = Math.min(i + 1, n - 1); p = 0;
};
document.getElementById('reset').onclick = function () {
  i = 0; p = 0; playing = true; document.getElementById('play').textContent = 'Pause';
};
scrub.oninput = function () {
  playing = false; document.getElementById('play').textContent = 'Play'; i = Number(scrub.value); p = 0;
};
requestAnimationFrame(tick);
</script>
"""


def build_html(walls, steps, action_names, cell_m, max_px=760):
    """Return (html, height_px) for st.components.v1.html."""
    h, w = walls.shape
    cp = max(14, min(48, max_px // w))
    order = ["front", "left", "right", "back"]
    data = {
        "w": w, "h": h, "cp": cp, "cellM": cell_m,
        "walls": walls.astype(int).tolist(),
        "offsets": SENSOR_OFFSETS,
        "hnames": ["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
        "steps": [
            {"x": s["x"], "y": s["y"], "h": s["heading"],
             "c": [round(s["cells"][k], 3) for k in order],
             "m": [round(s["meters"][k], 3) for k in order],
             "s": s["state"], "v": s["value"], "a": s["action"],
             "ai": action_names.index(s["action"]), "x_": s["collided"]}
            for s in steps
        ],
    }
    payload = json.dumps(data).replace("<", "\\u003c")   # keep labels from closing the script tag
    return TEMPLATE.replace("__DATA__", payload), h * cp + 210
