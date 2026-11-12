/* CrashLab occupant + injury-risk model (runs in the browser so belt / helmet toggles update live).

   Occupant kinematics: a lumped torso/head mass in the vehicle frame (x forward, y left) driven by
   the vehicle's crash pulse (half-sine acceleration with the impact's ΔV and a typical duration).
   Restraints are springs + dampers per unit mass: belt (with slack), seat back, frontal / side
   airbag cushions once inflated, and hard interior surfaces (wheel/dash, door, front seat back).
   Door intrusion from the crush model reduces the near-side space.

   Injury risk (indicative, NOT a medical prediction):
   - Car occupants, MAIS3+ vs ΔV. Belted frontal curve anchored to ~10% at 70 km/h for modern
     belted, airbag-equipped cars (Lübbe et al. 2024, AAP 202:107586); near-side anchored so the
     equivalent striking speed matches the same study. Adjustments as odds ratios:
       unbelted ×2.2 (NHTSA: belts cut moderate-critical injury ~50%),
       frontal airbag deployed ×0.71 (NHTSA: −29% driver deaths in frontal crashes),
       side head airbag deployed ×0.63 (IIHS: −37% driver deaths in driver-side crashes),
       seat-zone intrusion >0.25 m ×2, tip-over ×2 belted / ×6 unbelted (ejection).
   - Riders / cyclists: fatality vs impact speed, Rosén & Sander 2009 (pedestrian curve used as a
     proxy), helmet ×0.58 (Liu et al. 2008 Cochrane review, motorcycle helmets). */
"use strict";
const SAFE = (() => {
  const KMH = 3.6;
  const logistic = z => 1 / (1 + Math.exp(-z));
  const oddsMul = (p, k) => { p = Math.min(Math.max(p, 1e-6), 1 - 1e-6); const o = p / (1 - p) * k; return o / (1 + o); };

  // seat geometry per style: [x from CG (m), y (m, +left), interior distance ahead (m), door half-gap]
  function seatPos(spec, seat){
    const L = spec.length, off = spec.rect_off, st = spec.style;
    const front = off + L / 2;
    let xf, xr, yo;
    if(st === "auto"){ xf = front - 0.95; xr = front - 1.75; yo = 0.33; }
    else if(st === "truck"){ xf = front - 1.05; xr = xf - 0.8; yo = 0.55; }
    else if(st === "minitruck"){ xf = front - 0.75; xr = xf - 0.7; yo = 0.36; }
    else if(st === "bus"){ xf = front - 1.0; xr = xf - 2.5; yo = 0.75; }
    else { const cab0 = front - L * (st === "suv" || st === "csuv" ? 0.36 : 0.40); xf = cab0 - 0.35; xr = xf - 0.88; yo = spec.width * 0.205; }
    const m = {"driver": [xf, -yo], "front passenger": [xf, yo], "rear left": [xr, yo], "rear right": [xr, -yo],
               "rear centre": [xr, 0], "rider": [-0.05, 0], "passengers": [xr, 0]};
    const p = m[seat] || [xf, 0];
    if(st === "auto" && seat === "driver") p[1] = 0;
    return p;
  }
  const isRider = spec => spec.kind === "two-wheeler" || spec.kind === "cycle";

  /* simulate one occupant through all impacts of its vehicle */
  function simulate(veh, occ, opts = {}){
    const sp = veh.spec, sf = veh.safety, belt = !!occ.belt && !isRider(sp);
    const seat = occ.seat, [sx, sy] = seatPos(sp, seat);
    const front = seat === "driver" || seat === "front passenger";
    const auto = sp.style === "auto", bus = sp.style === "bus";
    const xInt = auto ? (seat === "driver" ? 0.38 : 0.55) : front ? (seat === "driver" ? 0.42 : 0.55) : 0.62;
    const res = {series: [], maxX: 0, maxY: 0, contact: null, ejected: false, peakG: 0, oiv: null};
    const bags = sf.airbags_deployed || [];
    const dt = 0.001;
    for(const pu of sf.pulses){
      if(Math.hypot(pu.dvl, pu.dvt) < 1.0) continue;
      const Tp = pu.Tp, amp = Math.PI / (2 * Tp);
      // near-side door space (intrusion when the struck face is the seat's side)
      const seatSide = sy > 0.05 ? "left" : sy < -0.05 ? "right" : "";
      let door = auto ? 0.65 : bus ? 0.5 : 0.30;
      let intr = 0;
      if((pu.face === "left" || pu.face === "right") && pu.face === seatSide){
        const xs = (sx - sp.rect_off) / (sp.length / 2);
        if(Math.abs(xs - pu.u) < 0.55) intr = pu.depth * Math.max(0, 1 - Math.abs(xs - pu.u) / 0.55);
      }
      const fBag = bags.find(b => b.type === "front" && Math.abs(b.t - pu.t) < 0.2 && front);
      const sBag = bags.find(b => (b.type === "side" || b.type === "curtain") && b.side === seatSide && Math.abs(b.t - pu.t) < 0.2 && front);
      let x = 0, y = 0, vx = 0, vy = 0, s = [], t = 0, cg = 0;
      for(let k = 0; k <= 350; k++){
        t = k * dt;
        const pa = t <= Tp ? amp * Math.sin(Math.PI * t / Tp) : 0;
        // relative to the cabin the occupant is pushed opposite the vehicle's acceleration
        const fx = -(pu.dvl / KMH) * pa, fy = -(pu.dvt / KMH) * pa;
        let rx = 0, ry = 0;
        if(belt){
          if(x > 0.05) rx += -1250 * (x - 0.05) - 28 * vx;
          ry += -260 * y - 6 * vy;
        }
        if(x < 0) rx += -9000 * x - 90 * vx;                     // seat back
        if(fBag && t >= (fBag.t - pu.t) + 0.03 && x > 0.27) rx += -1500 * (x - 0.27) - 55 * vx;
        if(x > xInt){ rx += -45000 * (x - xInt) - 400 * vx; if(!res.contact) res.contact = {what: front ? (seat === "driver" ? "steering wheel / dashboard" : "dashboard") : "front seat back", v: Math.abs(vx) * KMH, t: pu.t + t}; }
        // doors / intrusion (door moves inward with the crush during the pulse)
        const grow = Math.min(1, t / Tp);
        const dside = seatSide === "left" ? 1 : seatSide === "right" ? -1 : 0;
        if(dside){
          const gap = door - intr * grow;
          if(sBag && t >= (sBag.t - pu.t) + 0.02 && dside * y > gap - 0.14) ry += -dside * 1800 * (dside * y - (gap - 0.14)) - 50 * vy;
          if(dside * y > gap){
            if(auto){ if(!res.ejected && dside * y > gap + 0.25) res.ejected = true; }
            else { ry += -dside * 40000 * (dside * y - gap) - 350 * vy; if(!res.contact || res.contact.what.indexOf("door") < 0) res.contact = {what: intr > 0.05 ? "intruding door" : "door", v: Math.abs(vy) * KMH, t: pu.t + t}; }
          }
          const far = -dside;  // other side: console / other occupant
          if(far * y > 0.55){ ry += -far * 9000 * (far * y - 0.55) - 120 * vy; }
        } else if(Math.abs(y) > (auto ? 0.6 : 0.45)){   // centre seat: open sides (auto) or the cabin walls
          if(auto){ res.ejected = true; } else { ry += -Math.sign(y) * 20000 * (Math.abs(y) - 0.45) - 200 * vy; }
        }
        cg = Math.max(cg, Math.hypot(rx, ry) / 9.81);
        vx += (fx + rx) * dt; vy += (fy + ry) * dt; x += vx * dt; y += vy * dt;
        if(res.ejected && Math.abs(y) > 0.9){ y = Math.sign(y) * 0.9; vy = 0; }   // out of the cabin: shown at the opening
        if(k % 5 === 0) s.push([pu.t + t, x, y]);
      }
      s.push([pu.t + 0.36, x * 0.5, y * 0.5], [pu.t + 0.6, 0, 0]);
      res.series.push(...s);
      res.peakG = Math.max(res.peakG, cg);
      for(const p of s){ res.maxX = Math.max(res.maxX, p[1]); res.maxY = Math.max(res.maxY, Math.abs(p[2])); }
      res.intrusion = Math.max(res.intrusion || 0, intr);
    }
    res.series.sort((a, b) => a[0] - b[0]);
    return res;
  }

  /* interpolate an occupant's excursion at time t */
  function excursion(series, t){
    if(!series || !series.length || t < series[0][0]) return [0, 0];
    let lo = 0, hi = series.length - 1;
    if(t >= series[hi][0]) return [series[hi][1], series[hi][2]];
    while(hi - lo > 1){ const m = (lo + hi) >> 1; if(series[m][0] <= t) lo = m; else hi = m; }
    const a = series[lo], b = series[hi], f = (t - a[0]) / Math.max(1e-6, b[0] - a[0]);
    return [a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
  }

  /* MASH flail-space occupant impact velocity for an unrestrained occupant (longitudinal 0.6 m,
     lateral 0.3 m), from the half-sine pulse. Used only as a severity indicator. */
  function oiv(pu){
    const comp = (dv, lim) => {
      dv = Math.abs(dv) / KMH; if(dv < 1e-3) return null;
      const Tp = pu.Tp, amp = Math.PI / (2 * Tp); let v = 0, x = 0;
      for(let t = 0; t < 1.5; t += 0.0005){ const a = t <= Tp ? dv * amp * Math.sin(Math.PI * t / Tp) : 0; v += a * 0.0005; x += v * 0.0005; if(x >= lim) return v * KMH; }
      return null; };
    return {x: comp(pu.dvl, 0.6), y: comp(pu.dvt, 0.3)};
  }

  function direction(pu, seatSide){
    const c = pu.clock;
    if([10, 11, 12, 1, 2].includes(c)) return "frontal";
    if([5, 6, 7].includes(c)) return "rear";
    const face = pu.face;
    if(face === seatSide) return "near-side";
    return "far-side";
  }

  function risk(veh, occ, sim){
    const sp = veh.spec, sf = veh.safety;
    if(isRider(sp)){
      const pu = sf.pulses.reduce((m, p) => (!m || Math.abs(p.closing) > Math.abs(m.closing)) ? p : m, null);
      if(!pu) return {p: 0, kind: "fatality", basis: "no impact", factors: []};
      const v = Math.abs(pu.closing);
      let p = logistic(-6.9 + 0.090 * v);
      const f = [`impact speed ${v.toFixed(0)} km/h (Rosén & Sander 2009 curve)`];
      if(occ.helmet){ p *= 0.58; f.push("helmet ×0.58"); } else f.push("no helmet");
      if(pu.other === "truck" || pu.other === "bus"){ p = oddsMul(p, 1.8); f.push("heavy vehicle ×1.8 odds"); }
      return {p, kind: "fatality", basis: "rider", factors: f};
    }
    let best = {p: 0, factors: ["no significant impact"], dir: "–"};
    const [sx, sy] = seatPos(sp, occ.seat);
    const seatSide = sy > 0.05 ? "left" : sy < -0.05 ? "right" : "";
    const front = occ.seat === "driver" || occ.seat === "front passenger";
    for(const pu of sf.pulses){
      const dv = Math.hypot(pu.dvl, pu.dvt); if(dv < 3) continue;
      const dir = direction(pu, seatSide);
      let p, f = [];
      if(dir === "frontal"){ p = logistic(-7.32 + 0.0732 * dv); f.push(`frontal ΔV ${dv.toFixed(0)} km/h`); }
      else if(dir === "rear"){ p = logistic(-7.32 + 0.0732 * dv * 0.5); f.push(`rear ΔV ${dv.toFixed(0)} km/h`); }
      else { p = logistic(-5.99 + 0.0999 * dv); f.push(`${dir} ΔV ${dv.toFixed(0)} km/h`); if(dir === "far-side"){ p = oddsMul(p, 0.5); f.push("far side ×0.5"); } }
      const bags = (sf.airbags_deployed || []).filter(b => Math.abs(b.t - pu.t) < 0.2);
      if(!occ.belt){ p = oddsMul(p, 2.2); f.push("unbelted ×2.2"); }
      if(dir === "frontal" && front && bags.some(b => b.type === "front")){ p = oddsMul(p, 0.71); f.push("front airbag ×0.71"); }
      if(dir === "near-side" && front && bags.some(b => (b.type === "side" || b.type === "curtain") && b.side === seatSide)){ p = oddsMul(p, 0.63); f.push("side airbag ×0.63"); }
      if(sim && sim.intrusion > 0.25 && dir === "near-side"){ p = oddsMul(p, 2); f.push(`intrusion ${(sim.intrusion * 100).toFixed(0)} cm ×2`); }
      if(sp.style === "auto"){ p = oddsMul(p, 1.5); f.push("light open body ×1.5"); }
      if(sim && sim.contact && sim.contact.v > 15 && dir !== "rear"){ const k = sim.contact.v > 30 ? 3 : 1.8; p = oddsMul(p, k); f.push(`hits ${sim.contact.what} at ${sim.contact.v.toFixed(0)} km/h ×${k}`); }
      if(p > best.p) best = {p, factors: f, dir};
    }
    if(sf.toppled){ const k = occ.belt ? 2 : 6; best.p = oddsMul(Math.max(best.p, 0.02), k); best.factors.push(`tip-over ×${k}${occ.belt ? "" : " (ejection)"}`); }
    if(sim && sim.ejected){ best.p = oddsMul(best.p, 3); best.factors.push("thrown out of open side ×3"); }
    return {p: best.p, kind: "serious injury (MAIS3+)", basis: best.dir, factors: best.factors};
  }

  function assess(veh, occ){
    const sim = isRider(veh.spec) ? null : simulate(veh, occ);
    const r = risk(veh, occ, sim);
    const pu = veh.safety.pulses.find(p => Math.hypot(p.dvl, p.dvt) > 3);
    return {sim, risk: r, oiv: pu ? oiv(pu) : null};
  }
  return {assess, simulate, excursion, seatPos, isRider};
})();
