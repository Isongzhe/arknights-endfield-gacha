// Browser port of the first-UP waiting-time recursion (src/gacha/models/endfield.py).
// build.py checks it against the Python engine on every build; change rules there first.
// o: {t, n, free, vac: [..], soft: bool, guar: 120} -> pmf[j] = P(first UP with exactly j own pulls)
function firstUp(o) {
  const H = 80, q = 0.5, base = 0.008, guar = o.guar || 120;
  const p = (t) => (t >= H - 1 ? 1 : o.soft && t >= 65 ? Math.min(1, base + 0.05 * (t - 64)) : base);
  const pv = 1 - Math.pow(1 - base * q, 10);
  let m = new Float64Array(H);
  m[o.t] = 1;
  const pmf = [0];
  let paid = 0;
  for (let n = o.n; n < guar; n++) {
    if (n >= o.free) { paid++; pmf[paid] = 0; }
    const n1 = n + 1;
    if (n1 === guar) { let alive = 0; for (let t = 0; t < H; t++) alive += m[t]; pmf[paid] += alive; break; }
    const nm = new Float64Array(H);
    let s = 0;
    for (let t = 0; t < H; t++) {
      const mt = m[t];
      if (!mt) continue;
      const pt = p(t);
      s += mt * pt * q;
      nm[0] += mt * pt * (1 - q);
      if (pt < 1) nm[t + 1] += mt * (1 - pt);
    }
    if (o.vac.includes(n1)) {
      let rest = 0;
      for (let t = 0; t < H; t++) { rest += nm[t]; nm[t] *= 1 - pv; }
      s += rest * pv;
    }
    pmf[paid] += s;
    m = nm;
  }
  return pmf;
}
const cumsum = (a) => { let s = 0; return a.map((v) => (s += v)); };
// Two independent banners sharing one stock B; pull banner A first with at most x own pulls.
function decide(fa, fb, B) {
  const Fa = cumsum(fa), Fb = cumsum(fb);
  const cb = (b) => (b < 0 ? 0 : Fb[Math.min(b, Fb.length - 1)]);
  const rows = [];
  for (let x = 0; x <= B; x++) {
    let both = 0;
    for (let j = 0; j <= Math.min(x, fa.length - 1); j++) both += fa[j] * cb(B - j);
    const pa = Fa[Math.min(x, Fa.length - 1)];
    rows.push({ x, a: pa, b: both + (1 - pa) * cb(B - x), both });
  }
  return rows;
}
function convolve(a, b) {
  const out = new Array(a.length + b.length - 1).fill(0);
  for (let i = 0; i < a.length; i++) for (let j = 0; j < b.length; j++) out[i + j] += a[i] * b[j];
  return out;
}
if (typeof module !== "undefined") module.exports = { firstUp, decide, convolve, cumsum };
