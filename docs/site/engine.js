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
// Weapon banner (src/gacha/models/weapon.py): pmf[k] = P(rate-up weapon arrives in the k-th
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
// Consecutive limited banners with carried pity and the 60-pull dossier (src/gacha/models/plan.py).
// o: {t0, free, want: [bool...], useFree} -> {stages: [pmf of own pulls after each wanted banner]}
// Policy: chase every wanted banner to its rate-up (at most 120 counted pulls); on a skipped
// banner use the free pulls when useFree is true (dossier pulls are always used).
function limitedPlan(o) {
  const H = 80, q = 0.5, base = 0.008, GUAR = 120, pv = 1 - Math.pow(1 - base * q, 10);
  const p = (t) => (t >= H - 1 ? 1 : t >= 65 ? Math.min(1, base + 0.05 * (t - 64)) : base);
  const evolve = (m, k) => {  // pity after k pulls, 6* resets it
    for (let i = 0; i < k; i++) {
      const nm = new Float64Array(H);
      for (let t = 0; t < H; t++) { if (!m[t]) continue; const pt = p(t); nm[0] += m[t] * pt; if (pt < 1) nm[t + 1] += m[t] * (1 - pt); }
      m = nm;
    }
    return m;
  };
  const kernels = new Map();
  function kernel(t0, d, wanted) {
    const key = t0 * 4 + d * 2 + (wanted ? 1 : 0);
    if (kernels.has(key)) return kernels.get(key);
    const out = new Map();  // (j, t, d) -> prob
    const add = (j, t, dd, pr) => { if (pr <= 0) return; const k = (j * H + t) * 2 + dd; out.set(k, (out.get(k) || 0) + pr); };
    const start = new Float64Array(H); start[t0] = 1;
    if (!wanted) {
      const f = (o.skipFree != null ? o.skipFree : o.useFree ? o.free : 0) + (d ? 10 : 0), m = evolve(start, f);
      for (let t = 0; t < H; t++) add(0, t, 0, m[t]);
    } else {
      const f = o.free + (d ? 10 : 0);
      let m = start;
      for (let n = 0; n < GUAR; n++) {
        const n1 = n + 1, own = Math.max(0, n1 - f);
        if (n1 === GUAR) { let alive = 0; for (let t = 0; t < H; t++) alive += m[t]; add(own, 0, 1, alive); break; }
        const nm = new Float64Array(H);
        let up = 0;
        for (let t = 0; t < H; t++) {
          if (!m[t]) continue;
          const pt = p(t);
          up += m[t] * pt * q; nm[0] += m[t] * pt * (1 - q);
          if (pt < 1) nm[t + 1] += m[t] * (1 - pt);
        }
        if (up > 0) {
          if (n1 < f) {  // the remaining free pulls are still used and move the pity on
            const z = new Float64Array(H); z[0] = 1; const e = evolve(z, f - n1);
            for (let t = 0; t < H; t++) add(0, t, f >= 60 ? 1 : 0, up * e[t]);
          } else add(own, 0, n1 >= 60 ? 1 : 0, up);
        }
        if (n1 === 30) for (let t = 0; t < H; t++) { if (!nm[t]) continue; add(own, t, 0, nm[t] * pv); nm[t] *= 1 - pv; }
        m = nm;
      }
    }
    const list = [...out].map(([k, pr]) => ({ j: Math.floor(k / (2 * H)), t: Math.floor(k / 2) % H, d: k % 2, pr }));
    kernels.set(key, list);
    return list;
  }
  const L = o.want.filter(Boolean).length * GUAR + 1;
  let dist = new Map([[o.t0 * 2, Float64Array.of(1)]]);  // key t*2+d -> pmf over own pulls
  const stages = [];
  o.want.forEach((wanted) => {
    const next = new Map();
    for (const [key, arr] of dist) {
      for (const e of kernel(Math.floor(key / 2), key % 2, wanted)) {
        const k2 = e.t * 2 + e.d;
        let tgt = next.get(k2);
        if (!tgt) { tgt = new Float64Array(L); next.set(k2, tgt); }
        for (let j = 0; j < arr.length; j++) if (arr[j]) tgt[j + e.j] += arr[j] * e.pr;
      }
    }
    dist = next;
    if (wanted) {
      const tot = new Float64Array(L);
      for (const arr of dist.values()) for (let j = 0; j < L; j++) tot[j] += arr[j];
      let last = L - 1; while (last > 0 && tot[last] < 1e-15) last--;
      stages.push(Array.from(tot.subarray(0, last + 1)));
    }
  });
  return { stages };
}
if (typeof module !== "undefined") module.exports.limitedPlan = limitedPlan;
