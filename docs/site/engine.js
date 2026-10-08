// Browser port of the first-UP waiting-time recursion (src/gacharisk/models/endfield.py).
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
// Several copies on one limited banner (target_copies in src/gacharisk/models/endfield.py).
// o: {t, n, free, vac, soft, guar, copies} -> pmf[j] = P(the copies-th copy arrives with exactly j own pulls).
// The 120 guarantee fires once; every 240th counted pull grants a token that counts as a copy.
function copiesUp(o) {
  const H = 80, q = 0.5, base = 0.008, guar = o.guar || 120, K = o.copies, every = 240;
  const p = (t) => (t >= H - 1 ? 1 : o.soft && t >= 65 ? Math.min(1, base + 0.05 * (t - 64)) : base);
  const bin = []; { const r = base * q; let c = 1; for (let i = 0; i <= 10; i++) { bin.push(c * Math.pow(r, i) * Math.pow(1 - r, 10 - i)); c = (c * (10 - i)) / (i + 1); } }
  const idx = (t, c, u) => (t * K + c) * 2 + u, size = H * K * 2;
  let m = new Float64Array(size);
  m[idx(o.t, 0, 0)] = 1;
  const pmf = [0];
  let paid = 0, alive = 1;
  for (let n = o.n; alive > 1e-15 && n < 4000; n++) {
    if (n >= o.free) { paid++; pmf[paid] = 0; }
    const n1 = n + 1, token = n1 % every === 0 ? 1 : 0, vac = o.vac.includes(n1);
    const nm = new Float64Array(size);
    let done = 0;
    const put = (mass, t, c, u) => {
      c += token;
      if (!vac) { if (c >= K) done += mass; else nm[idx(t, c, u)] += mass; return; }
      let rest = mass;
      for (let i = 0; i <= 10 && c + i < K; i++) { nm[idx(t, c + i, u)] += mass * bin[i]; rest -= mass * bin[i]; }
      done += rest;
    };
    for (let t = 0; t < H; t++) for (let c = 0; c < K; c++) for (let u = 0; u < 2; u++) {
      const mt = m[idx(t, c, u)];
      if (!mt) continue;
      if (u === 0 && n1 === guar) { put(mt, 0, c + 1, 1); continue; }
      const pt = p(t);
      put(mt * pt * q, 0, c + 1, 1);
      put(mt * pt * (1 - q), 0, c, u);
      if (pt < 1) put(mt * (1 - pt), t + 1, c, u);
    }
    pmf[paid] += done;
    alive -= done;
    m = nm;
  }
  return pmf;
}
if (typeof module !== "undefined") module.exports = { firstUp, decide, convolve, cumsum, copiesUp };
// Weapon banner (src/gacharisk/models/weapon.py): pmf[k] = P(rate-up weapon arrives in the k-th
// issue from now). o: {issuesDone} ; 4% 6*, 25% rate-up, 6* forced on the 40th pull since the
// last one, rate-up forced on the 80th pull of the banner.
function weaponUp(o) {
  const P6 = 0.04, UP = 0.25, PITY = 40, GUAR = 80, n0 = o.issuesDone * 10;
  let m = new Float64Array(PITY);
  m[o.sinceSix == null ? n0 % PITY : o.sinceSix] = 1;
  const pmf = [0];
  for (let n = n0; n < GUAR; n++) {
    const k = Math.floor((n - n0) / 10) + 1;
    if (pmf.length <= k) pmf[k] = 0;
    if (n + 1 === GUAR) { let alive = 0; for (let s = 0; s < PITY; s++) alive += m[s]; pmf[k] += alive; break; }
    const nm = new Float64Array(PITY);
    let six = 0;
    for (let s = 0; s < PITY; s++) {
      if (!m[s]) continue;
      const p = s === PITY - 1 ? 1 : P6;
      six += m[s] * p;
      if (p < 1) nm[s + 1] += m[s] * (1 - p);
    }
    pmf[k] += six * UP;
    nm[0] += six * (1 - UP);
    m = nm;
  }
  return pmf;
}
if (typeof module !== "undefined") module.exports.weaponUp = weaponUp;
