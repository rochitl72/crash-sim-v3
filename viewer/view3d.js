/* CrashLab 3D view (Three.js r149). World (x east, y north) -> three (x, 0, -y), y up.
   Every vehicle is procedural low-poly geometry built from its spec (length, width, height,
   wheel layout, style). Damage deforms the body mesh from the crush model; roll comes from the
   simulated tip-over angle; occupants move with the occupant model's excursion. */
"use strict";
const W3D = (() => {
  let R, scene, cam, host, lblHost, world, sc = null, V = [], marks = [], flashes = [], trails = [];
  let sims = {}, opts = {damage: true, occ: true, trail: true, skid: true, labels: true};
  let camMode = "orbit", visible = false, ready = false;
  const orb = {target: new THREE.Vector3(), r: 40, th: -2.3, ph: 0.95, smooth: new THREE.Vector3()};
  const W2T = (x, y, z = 0) => new THREE.Vector3(x, z, -y);
  const C = c => new THREE.Color(c);

  /* ------------------------------------------------------------ setup */
  function init(container, labels){
    host = container; lblHost = labels;
    R = new THREE.WebGLRenderer({antialias: true, powerPreference: "high-performance"});
    R.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
    R.shadowMap.enabled = true; R.shadowMap.type = THREE.PCFSoftShadowMap;
    R.outputEncoding = THREE.sRGBEncoding; R.toneMapping = THREE.ACESFilmicToneMapping; R.toneMappingExposure = 1.05;
    host.appendChild(R.domElement);
    cam = new THREE.PerspectiveCamera(42, 1, 0.05, 2000);
    bindControls(R.domElement);
    ready = true;
  }
  function resize(){
    if(!ready) return;
    const r = host.getBoundingClientRect(); const w = Math.max(50, r.width), h = Math.max(50, r.height);
    R.setSize(w, h, false); R.domElement.style.width = w + "px"; R.domElement.style.height = h + "px";
    cam.aspect = w / h; cam.updateProjectionMatrix();
  }
  function setVisible(v){ visible = v; if(v) resize(); }

  /* ------------------------------------------------------------ materials */
  const MAT = {};
  function mats(){
    MAT.glass = new THREE.MeshStandardMaterial({color: 0x0b1620, roughness: .08, metalness: .2, transparent: true, opacity: .38, depthWrite: false});
    MAT.tyre = new THREE.MeshStandardMaterial({color: 0x111214, roughness: .9});
    MAT.rim = new THREE.MeshStandardMaterial({color: 0xb9c0c8, roughness: .35, metalness: .8});
    MAT.black = new THREE.MeshStandardMaterial({color: 0x15181c, roughness: .7});
    MAT.chrome = new THREE.MeshStandardMaterial({color: 0xcfd6de, roughness: .25, metalness: .9});
    MAT.head = new THREE.MeshStandardMaterial({color: 0xfff6d0, emissive: 0xfff1b8, emissiveIntensity: .9});
    MAT.dummy = new THREE.MeshStandardMaterial({color: 0xf2c14e, roughness: .55});
    MAT.dummyD = new THREE.MeshStandardMaterial({color: 0x2b2f36, roughness: .6});
    MAT.skin = new THREE.MeshStandardMaterial({color: 0xc68b59, roughness: .7});
    MAT.belt = new THREE.MeshStandardMaterial({color: 0xff7a00, emissive: 0xff5a00, emissiveIntensity: .35, roughness: .6});
    MAT.bag = new THREE.MeshStandardMaterial({color: 0xf4f6f8, roughness: .85, emissive: 0x9aa4ae, emissiveIntensity: .25, transparent: true, opacity: .92});
    MAT.helmet = new THREE.MeshStandardMaterial({color: 0x23272e, roughness: .3, metalness: .2});
    MAT.cargo = new THREE.MeshStandardMaterial({color: 0x8f8a74, roughness: .9, vertexColors: true});
  }

  /* ------------------------------------------------------------ geometry helpers */
  // deformable box: positions are in BODY coordinates (x fwd, z up, y left) mapped to three (x, z, -y)
  function dbox(len, ht, wid, cx, cz, cy, segs, mat, taper){
    const g = new THREE.BoxGeometry(len, ht, wid, segs[0], segs[1], segs[2]);
    const p = g.attributes.position;
    if(taper){ for(let i = 0; i < p.count; i++){ const y = p.getY(i);
      if(y > 0){ let x = p.getX(i); x = x > 0 ? x - taper.f : x + taper.r; p.setX(i, x); p.setZ(i, p.getZ(i) * (taper.w || .9)); } } }
    g.translate(cx, cz, -cy);
    const col = new Float32Array(p.count * 3); g.setAttribute("color", new THREE.BufferAttribute(col, 3));
    g.computeVertexNormals();
    const m = new THREE.Mesh(g, mat); m.castShadow = true; m.receiveShadow = true;
    m.userData.orig = Float32Array.from(p.array); m.userData.deform = true;
    return m;
  }
  function box(len, ht, wid, cx, cz, cy, mat){
    const m = new THREE.Mesh(new THREE.BoxGeometry(len, ht, wid), mat); m.position.set(cx, cz, -cy); m.castShadow = true; return m;
  }
  function cyl(r, h, mat, seg = 18){ const m = new THREE.Mesh(new THREE.CylinderGeometry(r, r, h, seg), mat); m.castShadow = true; return m; }
  function wheel(r, w, thin){
    const g = new THREE.Group();
    if(thin){ const t = new THREE.Mesh(new THREE.TorusGeometry(r - .03, thin, 8, 28), MAT.tyre); g.add(t);
      for(let k = 0; k < 6; k++){ const s = new THREE.Mesh(new THREE.CylinderGeometry(.006, .006, r * 2 - .08, 4), MAT.chrome); s.rotation.z = k * Math.PI / 6; g.add(s); }
    } else {
      const t = cyl(r, w, MAT.tyre, 22); t.rotation.x = Math.PI / 2; g.add(t);
      const h = cyl(r * .58, w + .01, MAT.rim, 14); h.rotation.x = Math.PI / 2; g.add(h);
    }
    g.castShadow = true;
    const spin = new THREE.Group(); spin.add(g); return spin;
  }
  function vertexColor(mesh, hex){
    const c = C(hex); const col = mesh.geometry.attributes.color;
    for(let i = 0; i < col.count; i++) col.setXYZ(i, c.r, c.g, c.b);
    col.needsUpdate = true; mesh.userData.base = c;
  }

  /* ------------------------------------------------------------ dummies */
  function dummy(kind, belted, helmet){
    const g = new THREE.Group();               // pivot at hip
    const torso = new THREE.Group(); g.add(torso);
    const mat = MAT.dummy;
    const t = box(.24, .5, .36, 0, .25, 0, mat); torso.add(t);
    const hd = new THREE.Mesh(new THREE.SphereGeometry(.105, 16, 12), helmet ? MAT.helmet : (kind === "rider" ? MAT.skin : mat));
    hd.position.set(.02, .63, 0); hd.castShadow = true; torso.add(hd);
    if(helmet){ const v = box(.02, .07, .16, .1, .62, 0, MAT.glass); torso.add(v); }
    // thighs
    const th = box(.42, .14, .34, .2, -.02, 0, mat); g.add(th);
    let beltM = null;
    if(belted){ beltM = new THREE.Group();
      const d = box(.03, .62, .07, .135, .28, 0, MAT.belt); d.rotation.x = .62; beltM.add(d);
      const l = box(.03, .07, .4, .135, .02, 0, MAT.belt); beltM.add(l); torso.add(beltM); }
    g.userData = {torso, head: hd, belt: beltM};
    return g;
  }

  /* ------------------------------------------------------------ vehicles */
  function buildVehicle(v, vi){
    const s = v.spec, L = s.length, Wd = s.width, H = s.height, off = s.rect_off, st = s.style;
    const root = new THREE.Group(); const pivot = new THREE.Group(); const body = new THREE.Group();
    root.add(pivot); pivot.add(body);
    const paint = new THREE.MeshStandardMaterial({color: 0xffffff, vertexColors: true, roughness: .38, metalness: .35});
    const front = off + L / 2, rear = off - L / 2;
    const def = [], wheels = [], seats = [], bags = {}, lights = {brake: []};
    const add = (m, d) => { body.add(m); if(d) def.push(m); return m; };
    const wr = {hatch: .30, sedan: .31, csuv: .34, suv: .37, minitruck: .30, truck: .52, bus: .52, auto: .21, moto: .30, scooter: .23, cycle: .34}[st] || .3;
    let cabin = null;
    if(["hatch", "sedan", "csuv", "suv"].includes(st)){
      const P = {hatch: [.17, .50, .36, .97, .55, .18], sedan: [.16, .47, .36, .80, .62, .45], csuv: [.2, .58, .33, .96, .5, .12], suv: [.22, .62, .31, .97, .42, .08]}[st];
      const [clr, lowH, c0, c1, rf, rr_] = P;
      const lo = add(dbox(L, lowH, Wd, off, clr + lowH / 2, 0, [22, 3, 10], paint, {f: .06, r: .04, w: .97}), true);
      vertexColor(lo, s.color);
      const cx0 = front - L * c0, cx1 = front - L * c1, cl = cx0 - cx1, ch = H - clr - lowH;
      cabin = add(dbox(cl, ch, Wd * .96, (cx0 + cx1) / 2, clr + lowH + ch / 2, 0, [10, 3, 8], MAT.glass, {f: rf, r: rr_, w: .86}), true);
      const roof = add(dbox(cl - rf - rr_ - .02, .05, Wd * .82, (cx0 + cx1) / 2 + (rr_ - rf) / 2, H - .02, 0, [6, 1, 4], paint), true);
      vertexColor(roof, s.color);
      // pillars (edges of the greenhouse)
      const eg = new THREE.LineSegments(new THREE.EdgesGeometry(cabin.geometry, 30), new THREE.LineBasicMaterial({color: C(s.color).multiplyScalar(.75)}));
      body.add(eg); cabin.userData.edges = eg;
      // lights
      const hl = [box(.05, .1, .3, front - .02, clr + lowH * .75, Wd / 2 - .25, MAT.head), box(.05, .1, .3, front - .02, clr + lowH * .75, -Wd / 2 + .25, MAT.head)];
      hl.forEach(m => add(m));
      for(const sg of [1, -1]){ const tm = new THREE.MeshStandardMaterial({color: 0x5a0b0b, emissive: 0xff1a1a, emissiveIntensity: .25});
        lights.brake.push(tm); add(box(.04, .12, .32, rear + .02, clr + lowH * .78, sg * (Wd / 2 - .24), tm)); }
      // wheels
      s.wheels.forEach(w => { const wh = wheel(wr, .22); wh.position.set(w[0], wr, -(Math.sign(w[1]) * (Wd / 2 - .14))); body.add(wh); wheels.push({m: wh, steer: w[2], r: wr}); });
      // seats / dummies
      bags.cab = {x0: cx1, x1: cx0, z0: clr + lowH, z1: H, clr, lowH};
    } else if(st === "minitruck" || st === "truck"){
      const clr = st === "truck" ? .45 : .28, cabL = st === "truck" ? 2.3 : 1.35, cabH = H - clr - (st === "truck" ? .3 : 0);
      const cab = add(dbox(cabL, cabH, Wd, front - cabL / 2, clr + cabH / 2, 0, [8, 6, 8], paint, {f: .12, r: 0, w: .97}), true);
      vertexColor(cab, s.color);
      add(box(.04, cabH * .35, Wd * .86, front + .005 - .1, clr + cabH * .72, 0, MAT.glass));
      add(box(L - cabL - .2, .18, Wd * .3, rear + (L - cabL) / 2, clr, 0, MAT.black));
      if(st === "truck"){
        const cg = add(dbox(L - cabL - .25, H - clr - .55, Wd, rear + (L - cabL - .25) / 2, clr + .55 + (H - clr - .55) / 2, 0, [12, 5, 6], MAT.cargo), true);
        vertexColor(cg, "#8f8a74");
      } else {
        const bed = add(dbox(L - cabL - .1, .45, Wd, rear + (L - cabL - .1) / 2, .75 + .22, 0, [10, 2, 6], MAT.cargo), true);
        vertexColor(bed, "#7d7a6a");
      }
      for(const sg of [1, -1]){ add(box(.05, .14, .25, front, clr + .4, sg * (Wd / 2 - .2), MAT.head));
        const tm = new THREE.MeshStandardMaterial({color: 0x5a0b0b, emissive: 0xff1a1a, emissiveIntensity: .25}); lights.brake.push(tm);
        add(box(.05, .14, .25, rear, clr + .45, sg * (Wd / 2 - .2), tm)); }
      s.wheels.forEach(w => { const wh = wheel(wr, .3); wh.position.set(w[0], wr, -(Math.sign(w[1]) * (Wd / 2 - .2))); body.add(wh); wheels.push({m: wh, steer: w[2], r: wr}); });
      bags.cab = {x0: front - cabL, x1: front, z0: clr, z1: H, clr, lowH: .6};
    } else if(st === "bus"){
      const clr = .32, bh = H - clr;
      const b = add(dbox(L, bh, Wd, off, clr + bh / 2, 0, [24, 6, 6], paint, {f: .05, r: .02, w: .99}), true);
      vertexColor(b, s.color);
      for(const sg of [1, -1]) add(box(L - 1.2, .85, .02, off - .3, clr + 1.55, sg * (Wd / 2 + .005), MAT.glass));
      add(box(.03, 1.1, Wd * .9, front + .01, clr + 1.45, 0, MAT.glass));
      for(const sg of [1, -1]){ add(box(.05, .15, .3, front + .01, clr + .45, sg * (Wd / 2 - .25), MAT.head));
        const tm = new THREE.MeshStandardMaterial({color: 0x5a0b0b, emissive: 0xff1a1a, emissiveIntensity: .25}); lights.brake.push(tm);
        add(box(.05, .2, .25, rear - .01, clr + .5, sg * (Wd / 2 - .2), tm)); }
      s.wheels.forEach(w => { const wh = wheel(wr, .3); wh.position.set(w[0], wr, -(Math.sign(w[1]) * (Wd / 2 - .22))); body.add(wh); wheels.push({m: wh, steer: w[2], r: wr}); });
      bags.cab = {x0: rear, x1: front, z0: clr, z1: H, clr, lowH: .9};
    } else if(st === "auto"){
      const clr = .2;
      const tub = add(dbox(L - .25, .62, Wd, off - .12, clr + .31, 0, [12, 3, 8], paint, {f: .1, r: 0, w: .98}), true);
      vertexColor(tub, s.color);
      const nose = add(dbox(.45, .9, Wd * .5, front - .25, clr + .45, 0, [3, 4, 4], paint, {f: .12, r: 0, w: .9}), true);
      vertexColor(nose, s.color);
      const canopy = add(dbox(L - .55, .07, Wd * .98, off - .2, H - .04, 0, [8, 1, 6], MAT.black), true);
      vertexColor(canopy, "#16191d");
      for(const [px, py] of [[front - .55, 1], [front - .55, -1], [rear + .1, 1], [rear + .1, -1]]) add(box(.04, H - clr - .62, .04, px, clr + .62 + (H - clr - .62) / 2, py * (Wd / 2 - .04), MAT.black));
      add(box(.03, .5, Wd * .85, front - .5, clr + 1.12, 0, MAT.glass));
      add(box(.05, .1, .14, front, clr + .65, 0, MAT.head));
      const tm = new THREE.MeshStandardMaterial({color: 0x5a0b0b, emissive: 0xff1a1a, emissiveIntensity: .25}); lights.brake.push(tm);
      add(box(.04, .08, .2, rear, clr + .45, .4, tm)); add(box(.04, .08, .2, rear, clr + .45, -.4, tm));
      s.wheels.forEach(w => { const wh = wheel(wr, .14); wh.position.set(w[0], wr, -w[1]); body.add(wh); wheels.push({m: wh, steer: w[2], r: wr}); });
      bags.cab = {x0: rear, x1: front, z0: clr, z1: H, clr, lowH: .62};
    } else { // two-wheelers and cycles
      const thin = st === "cycle" ? .022 : (st === "scooter" ? .055 : .065);
      s.wheels.forEach(w => { const wh = wheel(wr, .1, thin); wh.position.set(w[0], wr, 0); body.add(wh); wheels.push({m: wh, steer: w[2], r: wr}); });
      const ax = s.a, bx = -s.b;
      if(st === "cycle"){
        const fm = new THREE.MeshStandardMaterial({color: C(s.color), roughness: .4, metalness: .5});
        const tube = (x0, z0, x1, z1) => { const l = Math.hypot(x1 - x0, z1 - z0); const m = cyl(.022, l, fm, 6);
          m.position.set((x0 + x1) / 2, (z0 + z1) / 2, 0); m.rotation.z = Math.atan2(x0 - x1, z1 - z0) ; add(m, false); };
        tube(bx, wr, -.05, .55); tube(-.05, .55, ax - .12, .8); tube(-.1, .9, ax - .12, .8); tube(bx, wr, -.1, .9); tube(-.05, .55, -.1, .9); tube(ax, wr, ax - .12, .8);
        add(box(.24, .04, .1, -.12, .95, 0, MAT.black)); add(box(.04, .04, .5, ax - .12, .98, 0, MAT.chrome));
      } else {
        const bodyM = add(dbox(st === "scooter" ? 1.1 : .95, st === "scooter" ? .42 : .32, .3, st === "scooter" ? -.05 : .0, st === "scooter" ? .45 : .62, 0, [6, 3, 3], paint, {f: .05, r: .05, w: .85}), true);
        vertexColor(bodyM, s.color);
        add(box(.55, .08, .26, -.25, st === "scooter" ? .72 : .82, 0, MAT.black));
        add(box(.04, .04, .62, ax - .12, st === "scooter" ? 1.0 : 1.02, 0, MAT.chrome));
        const fk = cyl(.025, .75, MAT.chrome, 6); fk.position.set(ax - .06, .62, 0); fk.rotation.z = .3; add(fk);
        add(box(.05, .1, .12, ax + .02, .85, 0, MAT.head));
        const tm = new THREE.MeshStandardMaterial({color: 0x5a0b0b, emissive: 0xff1a1a, emissiveIntensity: .25}); lights.brake.push(tm);
        add(box(.04, .06, .12, bx - .05, .72, 0, tm));
      }
      bags.cab = null;
    }
    // occupants
    const occ = v.occupants || [];
    occ.forEach((o, k) => {
      const [sx, sy] = SAFE.seatPos(s, o.seat);
      const rider = SAFE.isRider(s);
      const d = dummy(rider ? "rider" : "occ", !!o.belt && !rider, !!o.helmet);
      let hz;
      if(rider){ hz = st === "cycle" ? .95 : st === "scooter" ? .72 : .8; d.position.set(sx - .05, hz, 0); d.userData.torso.rotation.z = st === "cycle" ? -.45 : -.3; }
      else { const cb = bags.cab; hz = cb ? cb.clr + Math.min(cb.lowH * .55, .55) + .05 : .6; if(st === "bus" || st === "truck") hz = cb.clr + .9;
        d.position.set(sx, hz, -sy); }
      d.userData.seat = o.seat; d.userData.sx = sx; d.userData.sy = sy; d.userData.idx = k; d.userData.hz = hz;
      body.add(d); seats.push(d);
    });
    // airbags (only meshes for fitted types)
    const ab = [];
    if(bags.cab && s.airbags && s.airbags.length){
      const cb = bags.cab;
      const front2 = SAFE.seatPos(s, "driver")[0];
      const mk = (type, side, geo, pos) => { const m = new THREE.Mesh(geo, MAT.bag); m.position.copy(pos); m.visible = false; m.castShadow = true; body.add(m); ab.push({type, side, m}); };
      const zc = (cb.z0 + cb.z1) / 2;
      if(s.airbags.includes("front")){
        mk("front", "right", new THREE.SphereGeometry(.3, 18, 12), new THREE.Vector3(front2 + .42, zc + .02, -SAFE.seatPos(s, "driver")[1]));
        mk("front", "left", new THREE.SphereGeometry(.34, 18, 12), new THREE.Vector3(front2 + .55, zc + .02, -SAFE.seatPos(s, "front passenger")[1]));
      }
      if(s.airbags.includes("knee")) mk("knee", "right", new THREE.BoxGeometry(.12, .16, .4), new THREE.Vector3(front2 + .38, cb.clr + .45, -SAFE.seatPos(s, "driver")[1]));
      for(const sd of ["left", "right"]){
        const sg = sd === "left" ? 1 : -1;
        if(s.airbags.includes("side")) mk("side", sd, new THREE.SphereGeometry(.2, 14, 10).scale(1.4, 1, .45), new THREE.Vector3(front2 - .02, zc - .05, -sg * (Wd / 2 - .2)));
        if(s.airbags.includes("curtain")) mk("curtain", sd, new THREE.BoxGeometry(cb.x1 - cb.x0 - .5, .38, .05), new THREE.Vector3((cb.x0 + cb.x1) / 2 - .1, cb.z1 - .3, -sg * (Wd / 2 - .12)));
      }
    }
    root.traverse(o => { if(o.isMesh) o.castShadow = true; });
    return {root, pivot, body, def, wheels, seats, ab, lights, spec: s, v, vi, roll: 0, dmgKey: ""};
  }

  /* ------------------------------------------------------------ damage deformation */
  function deform(Vh, t){
    const s = Vh.spec, dmg = (Vh.v.safety && Vh.v.safety.damage) || [];
    const act = dmg.map(d => Math.max(0, Math.min(1, (t - d.t) / 0.09)));
    const key = act.map(a => a.toFixed(2)).join(",") + (opts.damage ? "h" : "");
    if(key === Vh.dmgKey) return; Vh.dmgKey = key;
    const L = s.length, Wd = s.width, off = s.rect_off;
    const hot = C("#ff3b1f"), scr = C("#3a3f46");
    for(const m of Vh.def){
      const pos = m.geometry.attributes.position, o = m.userData.orig, col = m.geometry.attributes.color, base = m.userData.base;
      for(let i = 0; i < pos.count; i++){
        let bx = o[3 * i], bz = o[3 * i + 1], by = -o[3 * i + 2];
        let heat = 0;
        dmg.forEach((d, k) => {
          const pr = act[k]; if(pr <= 0 || d.depth < .004) return;
          const depth = d.depth * pr;
          const top = Math.max(s.bumper || .5, d.other_bumper || .5) + .55;
          const fz = bz <= top ? 1 : Math.max(.25, 1 - (bz - top) / .6);
          let dist, along, hw, nx = 0, ny = 0;
          if(d.face === "front"){ dist = (off + L / 2) - bx; along = by - d.u * Wd / 2; hw = Math.min(d.width, Wd) / 2; nx = -1; }
          else if(d.face === "rear"){ dist = bx - (off - L / 2); along = by - d.u * Wd / 2; hw = Math.min(d.width, Wd) / 2; nx = 1; }
          else if(d.face === "left"){ dist = Wd / 2 - by; along = bx - (off + d.u * L / 2); hw = Math.min(d.width, L) / 2; ny = -1; }
          else { dist = by + Wd / 2; along = bx - (off + d.u * L / 2); hw = Math.min(d.width, L) / 2; ny = 1; }
          const fa = Math.exp(-Math.pow(Math.abs(along) / Math.max(hw, .15), 4));
          const fd = Math.max(0, 1 - dist / (depth * 1.6 + .35));
          const dd = depth * fa * fd * fz;
          if(dd <= 0) return;
          // crumple: inward push plus a small wrinkle
          const wr = .12 * dd * Math.sin(along * 9 + bz * 7);
          bx += nx * dd + (ny ? wr : 0); by += ny * dd + (nx ? wr : 0); bz -= .15 * dd * fa;
          heat = Math.max(heat, dd / Math.max(d.depth, .06));
        });
        pos.setXYZ(i, bx, bz, -by);
        if(col && base){ const c = base.clone(); if(heat > 0) c.lerp(opts.damage ? hot : scr, Math.min(1, heat * (opts.damage ? 1.1 : .6))); col.setXYZ(i, c.r, c.g, c.b); }
      }
      pos.needsUpdate = true; if(col) col.needsUpdate = true; m.geometry.computeVertexNormals();
      if(m.userData.edges){ m.userData.edges.geometry.dispose(); m.userData.edges.geometry = new THREE.EdgesGeometry(m.geometry, 30); }
    }
  }

  /* ------------------------------------------------------------ world */
  function clear(){
    if(world){ scene.remove(world); world.traverse(o => { if(o.geometry) o.geometry.dispose(); }); }
    lblHost.innerHTML = ""; V = []; marks = []; flashes = []; trails = [];
  }
  function rng(seed){ let s = seed; return () => (s = (s * 16807) % 2147483647) / 2147483647; }
  function inRoad(x, y, m){ return sc.road.polys.some(p => { const xs = p.map(q => q[0]), ys = p.map(q => q[1]);
    return x > Math.min(...xs) - m && x < Math.max(...xs) + m && y > Math.min(...ys) - m && y < Math.max(...ys) + m; }); }

  function load(scn){
    if(!ready) return;
    if(!MAT.glass) mats();
    sc = scn; clear();
    scene = new THREE.Scene(); world = new THREE.Group(); scene.add(world);
    const fog = sc.id === "fog_pileup";
    const sky = fog ? 0x8f989f : 0x1a2430;
    scene.background = C(sky);
    scene.fog = fog ? new THREE.Fog(sky, 10, 58) : new THREE.Fog(sky, 90, 260);
    const hemi = new THREE.HemisphereLight(fog ? 0xd8dde2 : 0xbfd3ea, 0x1d2420, fog ? 1.0 : .85); world.add(hemi);
    const sun = new THREE.DirectionalLight(0xfff1dc, fog ? .45 : 1.35);
    const vw = sc.view || [-30, -20, 30, 20], cx = (vw[0] + vw[2]) / 2, cy = (vw[1] + vw[3]) / 2, span = Math.max(vw[2] - vw[0], vw[3] - vw[1]);
    sun.position.copy(W2T(cx - 30, cy - 45, 60)); sun.target.position.copy(W2T(cx, cy, 0));
    sun.castShadow = true; sun.shadow.mapSize.set(2048, 2048);
    const sh = sun.shadow.camera; sh.left = -span * .8; sh.right = span * .8; sh.top = span * .8; sh.bottom = -span * .8; sh.near = 1; sh.far = 250;
    sun.shadow.bias = -0.0004; world.add(sun, sun.target);
    // ground
    const gnd = new THREE.Mesh(new THREE.PlaneGeometry(900, 900), new THREE.MeshStandardMaterial({color: fog ? 0x56604f : 0x26342a, roughness: 1}));
    gnd.rotation.x = -Math.PI / 2; gnd.position.y = -0.02; gnd.receiveShadow = true; world.add(gnd);
    const asph = new THREE.MeshStandardMaterial({color: 0x30353d, roughness: .95});
    sc.road.polys.forEach(p => { const shp = new THREE.Shape(p.map(q => new THREE.Vector2(q[0], q[1])));
      const m = new THREE.Mesh(new THREE.ShapeGeometry(shp), asph); m.rotation.x = -Math.PI / 2; m.position.y = .0; m.receiveShadow = true; world.add(m);
      // kerbs
      for(let i = 0; i < p.length; i++){ const a = p[i], b = p[(i + 1) % p.length]; const l = Math.hypot(b[0] - a[0], b[1] - a[1]);
        const k = new THREE.Mesh(new THREE.BoxGeometry(l, .12, .25), new THREE.MeshStandardMaterial({color: 0x8a8f96, roughness: .9}));
        k.position.copy(W2T((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, .06)); k.rotation.y = Math.atan2(b[1] - a[1], b[0] - a[0]); k.receiveShadow = true;
        // skip kerb segments that cross another road polygon (junction mouths)
        const mx = (a[0] + b[0]) / 2, my = (a[1] + b[1]) / 2;
        const inside = sc.road.polys.some(pp => pp !== p && (() => { const xs = pp.map(q => q[0]), ys = pp.map(q => q[1]); return mx > Math.min(...xs) + .1 && mx < Math.max(...xs) - .1 && my > Math.min(...ys) + .1 && my < Math.max(...ys) - .1; })());
        if(!inside && l < 400) world.add(k); }
    });
    // markings
    const byCol = {};
    sc.road.lines.forEach(l => { for(let i = 0; i + 1 < l.pts.length; i++){ const a = l.pts[i], b = l.pts[i + 1];
      const len = Math.hypot(b[0] - a[0], b[1] - a[1]), ux = (b[0] - a[0]) / len, uy = (b[1] - a[1]) / len;
      const segs = l.dash ? [] : [[0, len]];
      if(l.dash){ for(let s = 0; s < len; s += l.dash[0] + l.dash[1]) segs.push([s, Math.min(len, s + l.dash[0])]); }
      const arr = byCol[l.color] = byCol[l.color] || [];
      segs.forEach(([s0, s1]) => { const w = l.w / 2, nx = -uy * w, ny = ux * w;
        const p0 = [a[0] + ux * s0, a[1] + uy * s0], p1 = [a[0] + ux * s1, a[1] + uy * s1];
        const q = [[p0[0] + nx, p0[1] + ny], [p1[0] + nx, p1[1] + ny], [p1[0] - nx, p1[1] - ny], [p0[0] - nx, p0[1] - ny]];
        arr.push(...[q[0], q[1], q[2], q[0], q[2], q[3]].flatMap(pt => [pt[0], .012, -pt[1]])); }); } });
    Object.entries(byCol).forEach(([c, arr]) => { const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.Float32BufferAttribute(arr, 3));
      world.add(new THREE.Mesh(g, new THREE.MeshBasicMaterial({color: C(c), side: THREE.DoubleSide, transparent: true, opacity: .8}))); });
    // roadside: trees + low buildings (deterministic)
    const r = rng(7 + sc.id.length * 13), tg = new THREE.ConeGeometry(1.1, 3.2, 7), tk = new THREE.CylinderGeometry(.14, .18, 1.1, 6);
    const tm = new THREE.MeshStandardMaterial({color: fog ? 0x4e5e4c : 0x2f5a3a, roughness: .9, flatShading: true}), km = new THREE.MeshStandardMaterial({color: 0x4b3a2b});
    const bm = [0x3b4652, 0x46505c, 0x524a44, 0x3e4a44].map(c => new THREE.MeshStandardMaterial({color: c, roughness: .9}));
    for(let k = 0; k < 140; k++){
      const x = cx + (r() - .5) * span * 2.6, y = cy + (r() - .5) * span * 2.6;
      if(inRoad(x, y, 3.5)) continue;
      if(r() < .78){ const tr = new THREE.Group(); const c = new THREE.Mesh(tg, tm); c.position.y = 2.6; c.castShadow = true; const t = new THREE.Mesh(tk, km); t.position.y = .55;
        tr.add(c, t); const sz = .7 + r() * .8; tr.scale.setScalar(sz); tr.position.copy(W2T(x, y)); world.add(tr); }
      else if(!inRoad(x, y, 8)){ const h = 3 + r() * 9, w = 5 + r() * 8, d = 5 + r() * 8; const b = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), bm[k % 4]);
        b.position.copy(W2T(x, y, h / 2)); b.rotation.y = r() * .2; b.castShadow = b.receiveShadow = true; world.add(b); }
    }
    // marks (ribbons revealed over time)
    sc.marks.forEach(m => {
      const scr = m.w < 0, w = scr ? .32 : .2, n = m.pts.length; if(n < 2) return;
      const arr = [];
      for(let i = 0; i + 1 < n; i++){ const a = m.pts[i], b = m.pts[i + 1]; const l = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1e-3;
        const nx = -(b[1] - a[1]) / l * w / 2, ny = (b[0] - a[0]) / l * w / 2, h = scr ? .017 : .015;
        const q = [[a[0] + nx, a[1] + ny], [b[0] + nx, b[1] + ny], [b[0] - nx, b[1] - ny], [a[0] - nx, a[1] - ny]];
        arr.push(...[q[0], q[1], q[2], q[0], q[2], q[3]].flatMap(pt => [pt[0], h, -pt[1]])); }
      const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.Float32BufferAttribute(arr, 3));
      const mesh = new THREE.Mesh(g, new THREE.MeshBasicMaterial({color: scr ? 0x9aa0a8 : 0x050607, transparent: true, opacity: scr ? .55 : .88, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: -2}));
      world.add(mesh); marks.push({mesh, ts: m.pts.map(p => p[2])});
    });
    // trajectories
    sc.veh.forEach((v, vi) => { const pts = sc.frames.map(f => W2T(f[vi][0], f[vi][1], .05));
      const g = new THREE.BufferGeometry().setFromPoints(pts); const ln = new THREE.Line(g, new THREE.LineDashedMaterial({color: C(v.spec.color), dashSize: .6, gapSize: .8, transparent: true, opacity: .7}));
      ln.computeLineDistances(); world.add(ln); trails.push(ln); });
    // vehicles
    sc.veh.forEach((v, vi) => { const Vh = buildVehicle(v, vi); world.add(Vh.root); V.push(Vh);
      // cumulative rolled distance for wheel spin (locked wheels do not turn)
      let dsum = 0; Vh.roll = sc.frames.map((f, i) => { if(i){ const p = sc.frames[i - 1][vi], q = f[vi];
        const locked = q[6] < -12 && !v.abs; if(!locked && !q[7]) dsum += Math.hypot(q[0] - p[0], q[1] - p[1]); } return dsum; });
      const lb = document.createElement("div"); lb.className = "l3 veh"; lb.innerHTML = `<b style="background:${v.spec.color}">${v.label}</b>`; lblHost.appendChild(lb); Vh.lbl = lb;
      const dl = document.createElement("div"); dl.className = "l3 dmg"; lblHost.appendChild(dl); Vh.dlbl = dl; });
    // impact flashes
    sc.events.forEach(e => { const m = new THREE.Mesh(new THREE.SphereGeometry(1, 20, 14), new THREE.MeshBasicMaterial({color: 0xffb020, transparent: true, opacity: 0, depthWrite: false, blending: THREE.AdditiveBlending}));
      m.position.copy(W2T(e.P[0], e.P[1], .7)); world.add(m); const pl = new THREE.PointLight(0xffa040, 0, 18); pl.position.copy(m.position); world.add(pl); flashes.push({e, m, pl}); });
    // camera
    resetCam(); orb.smooth.copy(orb.target);
    if(!["orbit", "top"].includes(camMode.split(":")[0]) || (camMode.includes(":") && +camMode.split(":")[1] >= sc.veh.length)) camMode = "orbit";
  }
  function camR(span, fog){ return fog ? 17 : Math.max(10, Math.min(26, span * .3)); }
  function resetCam(){
    if(!sc) return; const vw = sc.view || [-30, -20, 30, 20], span = Math.max(vw[2] - vw[0], vw[3] - vw[1]); const e0 = sc.events[0];
    if(!e0){ orb.target.copy(W2T((vw[0] + vw[2]) / 2, (vw[1] + vw[3]) / 2, .6)); orb.r = camR(span, false); orb.th = -2.25; orb.ph = .98; return; }
    // frame the first impact and where the vehicles involved come to rest
    const pts = [[e0.P[0], e0.P[1]], ...[e0.ia, e0.ib].map(i => [sc.final[i].x, sc.final[i].y])];
    const c = pts.reduce((a, p) => [a[0] + p[0] / pts.length, a[1] + p[1] / pts.length], [0, 0]);
    const spread = Math.max(...pts.map(p => Math.hypot(p[0] - c[0], p[1] - c[1]))) + Math.max(...[e0.ia, e0.ib].map(i => sc.veh[i].spec.length)) * .6;
    const fog = sc.id === "fog_pileup";
    orb.target.copy(W2T(c[0], c[1], .6)); orb.r = fog ? 17 : Math.max(10, Math.min(34, spread * 2.1)); orb.th = -2.25; orb.ph = .98; }

  /* ------------------------------------------------------------ per-frame update */
  function bagScale(t, tf){ if(t < tf) return 0; const a = t - tf; if(a < .03) return a / .03; if(a < .15) return 1; return Math.max(.55, 1 - (a - .15) * 1.0); }
  function update(t, P){
    if(!ready || !sc || !visible) return;
    const F = sc.frames;
    V.forEach((Vh, vi) => {
      const p = P(vi), s = Vh.spec;
      Vh.root.position.copy(W2T(p.x, p.y, 0)); Vh.root.rotation.y = p.psi;
      // roll: simulated tip-over angle, else small cosmetic body roll / pitch from accelerations
      const fi = Math.max(0, Math.min(F.length - 1, Math.round(t / sc.dt)));
      const roll = F[fi][vi][8] || 0;
      const bt = (s.kind === "two-wheeler" || s.kind === "cycle") ? 0 : (s.ssf ? s.ssf * s.cg_h : s.width / 2);
      const sg = roll >= 0 ? 1 : -1;
      let cosR = 0, cosP = 0;
      if(Math.abs(roll) < .01){
        const h = .08, a1 = P(vi, t + h), a0 = P(vi, t - h);
        const along = (a1.v - a0.v) / (2 * h), lat = p.v * p.r;
        cosP = Math.max(-.05, Math.min(.05, along * .0055)); cosR = Math.max(-.06, Math.min(.06, -lat * .0085));
        if(s.kind === "two-wheeler" || s.kind === "cycle") cosR = Math.max(-.5, Math.min(.5, Math.atan(lat / 9.81)));
      }
      Vh.pivot.position.set(0, 0, -sg * bt); Vh.body.position.set(0, 0, sg * bt);
      Vh.pivot.rotation.set(-(roll || cosR), 0, cosP);
      // wheels
      const dist = Vh.roll[fi] || 0;
      Vh.wheels.forEach(w => { w.m.rotation.set(0, w.steer ? p.delta : 0, 0); w.m.children[0].rotation.z = -dist / w.r; });
      // brake lights
      Vh.lights.brake.forEach(m => m.emissiveIntensity = p.acc < -2.5 ? 2.2 : .25);
      deform(Vh, t);
      // occupants
      const sim = sims[vi] || [];
      Vh.seats.forEach(d => { const k = d.userData.idx, o = Vh.v.occupants[k], ex = sim[k] ? SAFE.excursion(sim[k].sim && sim[k].sim.series, t) : [0, 0];
        d.visible = opts.occ;
        const tor = d.userData.torso; if(!SAFE.isRider(s)){
          tor.rotation.z = -Math.atan2(Math.max(-.2, ex[0]), .48); tor.rotation.x = -Math.atan2(ex[1], .55);
          d.position.x = d.userData.sx + Math.max(0, ex[0]) * .35; d.position.z = -(d.userData.sy + ex[1] * .4); }
        if(d.userData.belt) d.userData.belt.visible = !!o.belt; });
      // airbags
      const dep = (Vh.v.safety && Vh.v.safety.airbags_deployed) || [];
      Vh.ab.forEach(b => { const f = dep.find(d => d.type === b.type && (b.type === "front" || b.type === "knee" || d.side === b.side));
        const occupied = b.type !== "front" || b.side === "right" || Vh.v.occupants.some(o => o.seat === "front passenger");
        const sc_ = f && occupied ? bagScale(t, f.t) : 0; b.m.visible = sc_ > 0.01; if(b.m.visible) b.m.scale.setScalar(Math.max(.05, sc_)); });
      // labels
      const tp = W2T(p.x, p.y, s.height + .55); const sp = project(tp); const lb = Vh.lbl;
      lb.style.display = opts.labels && sp ? "" : "none"; if(sp){ lb.style.transform = `translate(${sp[0]}px,${sp[1]}px) translate(-50%,-100%)`; }
      const top = Vh.v.safety && Vh.v.safety.toppled;
      const isTop = !!(top && t >= top.t);
      if(lb.dataset.top !== String(isTop)){ lb.dataset.top = isTop; lb.innerHTML = `<b style="background:${Vh.v.spec.color}">${Vh.v.label}</b>${isTop ? '<i class="tp">TIPPED OVER</i>' : ""}`; }
      const dm = ((Vh.v.safety && Vh.v.safety.damage) || []).filter(d => t >= d.t && d.depth > .01);
      const dl = Vh.dlbl;
      if(opts.damage && opts.labels && dm.length){ const d = dm.reduce((a, b) => b.depth > a.depth ? b : a);
        const bx = d.face === "front" ? s.rect_off + s.length / 2 : d.face === "rear" ? s.rect_off - s.length / 2 : s.rect_off + d.u * s.length / 2;
        const by = d.face === "left" ? s.width / 2 : d.face === "right" ? -s.width / 2 : d.u * s.width / 2;
        const c = Math.cos(p.psi), sn = Math.sin(p.psi); const wp = W2T(p.x + c * bx - sn * by, p.y + sn * bx + c * by, .9);
        const q = project(wp);
        if(q){ dl.style.display = ""; dl.style.transform = `translate(${q[0]}px,${q[1]}px) translate(-50%,-140%)`;
          dl.innerHTML = `${Vh.v.label} · ${d.face} · <b>${d.severity}</b> ${(d.depth * 100).toFixed(0)} cm`; dl.dataset.sev = d.severity; }
        else dl.style.display = "none";
      } else dl.style.display = "none";
    });
    // marks
    marks.forEach(m => { let n = 0; while(n + 1 < m.ts.length && m.ts[n + 1] <= t) n++; m.mesh.geometry.setDrawRange(0, n * 6); m.mesh.visible = opts.skid; });
    const fi = Math.max(0, Math.min(F.length - 1, Math.floor(t / sc.dt)));
    trails.forEach(l => { l.geometry.setDrawRange(0, fi + 1); l.visible = opts.trail; });
    flashes.forEach(f => { const d = t - f.e.t; const on = d >= 0 && d < .8; const a = on ? 1 - d / .8 : 0;
      f.m.material.opacity = .55 * a; f.m.scale.setScalar(.4 + d * 5); f.m.visible = on; f.pl.intensity = on ? 6 * a : 0; });
    camUpdate(P);
    R.render(scene, cam);
  }
  function project(v){ const p = v.clone().project(cam); if(p.z > 1 || p.z < -1) return null;
    const r = R.domElement; return [(p.x + 1) / 2 * r.clientWidth, (1 - p.y) / 2 * r.clientHeight]; }

  /* ------------------------------------------------------------ camera */
  function camUpdate(P){
    const [m, idx] = camMode.split(":"); const i = +idx;
    if(m === "chase" || m === "driver"){
      const p = P(i), Vh = V[i], s = Vh.spec, hd = new THREE.Vector3(Math.cos(p.psi), 0, -Math.sin(p.psi));
      if(m === "chase"){ const back = Math.max(7, s.length * 1.9);
        const want = W2T(p.x, p.y, 0).addScaledVector(hd, -back).add(new THREE.Vector3(0, Math.max(2.8, s.height * 1.6), 0));
        cam.position.lerp(want, .18); cam.lookAt(W2T(p.x, p.y, s.height * .5).addScaledVector(hd, 4)); }
      else { const d = Vh.seats[0]; const sx = d ? d.userData.sx : 0, sy = d ? d.userData.sy : 0, hz = d ? d.userData.hz + .62 : s.height * .8;
        const c = Math.cos(p.psi), sn = Math.sin(p.psi);
        const sim = sims[i] && sims[i][0] && sims[i][0].sim; const ex = sim ? SAFE.excursion(sim.series, S_t()) : [0, 0];
        const wx = p.x + c * (sx - .05 + ex[0] * .5) - sn * (sy + ex[1] * .4), wy = p.y + sn * (sx - .05 + ex[0] * .5) + c * (sy + ex[1] * .4);
        const roll = Vh.pivot.rotation.x;
        cam.position.copy(W2T(wx, wy, hz)); cam.up.set(Math.sin(roll) * Math.sin(p.psi), Math.cos(roll), Math.sin(roll) * Math.cos(p.psi));
        cam.lookAt(W2T(wx + c * 10, wy + sn * 10, hz - .6)); cam.up.set(0, 1, 0); }
      return;
    }
    if(m === "follow"){ const p = P(i); orb.target.lerp(W2T(p.x, p.y, .6), .15); }
    if(m === "top"){ orb.ph = .02; }
    const sp = new THREE.Vector3(Math.sin(orb.ph) * Math.cos(orb.th), Math.cos(orb.ph), Math.sin(orb.ph) * Math.sin(orb.th)).multiplyScalar(orb.r);
    cam.position.copy(orb.target).add(sp); cam.lookAt(orb.target);
  }
  let S_t = () => 0;
  function setTimeGetter(f){ S_t = f; }
  function setCam(m){ camMode = m; if(m === "orbit" || m === "top"){ if(m === "orbit") orb.ph = Math.max(orb.ph, .55); } if(m === "orbit") resetCam(); }
  function bindControls(el){
    let drag = null, pts = new Map();
    el.addEventListener("contextmenu", e => e.preventDefault());
    el.addEventListener("pointerdown", e => { el.setPointerCapture(e.pointerId); pts.set(e.pointerId, [e.clientX, e.clientY]);
      drag = {x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey, th: orb.th, ph: orb.ph, t: orb.target.clone(), d: null};
      if(pts.size === 2){ const a = [...pts.values()]; drag.d = Math.hypot(a[0][0] - a[1][0], a[0][1] - a[1][1]); drag.r = orb.r; } });
    el.addEventListener("pointermove", e => { if(!drag) return; pts.set(e.pointerId, [e.clientX, e.clientY]);
      if(camMode.startsWith("chase") || camMode.startsWith("driver")) return;
      if(pts.size === 2 && drag.d){ const a = [...pts.values()]; const d = Math.hypot(a[0][0] - a[1][0], a[0][1] - a[1][1]); orb.r = Math.max(3, Math.min(250, drag.r * drag.d / d)); return; }
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      if(drag.pan){ const k = orb.r / 600; const fwd = new THREE.Vector3(-Math.cos(orb.th), 0, -Math.sin(orb.th)), rt = new THREE.Vector3(-Math.sin(orb.th), 0, Math.cos(orb.th));
        orb.target.copy(drag.t).addScaledVector(rt, -dx * k).addScaledVector(fwd, dy * k); if(camMode.startsWith("follow")) camMode = "orbit"; }
      else { orb.th = drag.th + dx * .006; orb.ph = Math.max(.05, Math.min(1.45, drag.ph - dy * .005)); if(camMode === "top") camMode = "orbit"; } });
    const up = e => { pts.delete(e.pointerId); if(!pts.size) drag = null; };
    el.addEventListener("pointerup", up); el.addEventListener("pointercancel", up);
    el.addEventListener("wheel", e => { e.preventDefault(); orb.r = Math.max(3, Math.min(250, orb.r * Math.exp(e.deltaY * .0012))); }, {passive: false});
  }
  return {init, load, update, resize, setVisible, setCam, resetCam, setOpts: o => Object.assign(opts, o), setSims: s => { sims = s; },
          setTimeGetter, get camMode(){ return camMode; }};
})();
