"""Multi-vehicle scene simulator: drivers -> tire dynamics -> collisions -> logging."""
import numpy as np
from .vehicles import G
from .dynamics import new_state, vehicle_step, forward_speed
from .collision import corners, sat_contact, apply_impulse, kinetic_energy
from .drivers import wrap


class VehicleCfg:
    def __init__(self, label, spec, x, y, psi, speed, steer_fn, accel_fn, abs_on=False,
                 post_impact="brake", note="", occupants=None):
        self.label, self.spec, self.x, self.y, self.psi, self.speed = label, spec, x, y, psi, speed
        self.occupants = occupants if occupants is not None else default_occupants(spec)
        self.steer_fn, self.accel_fn, self.abs_on = steer_fn, accel_fn, abs_on
        self.post_impact, self.note = post_impact, note


def default_occupants(spec):
    """Seat list. India is right-hand drive: the driver sits on the right (body -y)."""
    k = spec.kind
    if k in ("two-wheeler",):
        return [dict(seat="rider", helmet=True)]
    if k == "cycle":
        return [dict(seat="rider", helmet=False)]
    if k == "three-wheeler":
        return [dict(seat="driver", belt=False), dict(seat="rear left", belt=False),
                dict(seat="rear right", belt=False)]
    return [dict(seat="driver", belt=True)]


class Scene:
    def __init__(self, sid, title, subtitle, story, vehicles, road, mu=0.75, e=0.25, mu_c=0.15,
                 t_end=12.0, t_min=2.0, view=None, tags=()):
        self.id, self.title, self.subtitle, self.story = sid, title, subtitle, story
        self.vehicles, self.road, self.mu, self.e, self.mu_c = vehicles, road, mu, e, mu_c
        self.t_end, self.t_min, self.view, self.tags = t_end, t_min, view, tags


class _Veh:
    def __init__(self, cfg):
        self.cfg, self.spec = cfg, cfg.spec
        self.st = new_state(1, cfg.x, cfg.y, cfg.psi, cfg.speed)
        self.hit = False
        self.marks = [[] for _ in range(len(cfg.spec.wheels))]
        self.cur = [None] * len(cfg.spec.wheels)
        self.path = []
        self.phi, self.om = 0.0, 0.0          # roll angle (+ = tipping toward body +y) and rate
        self.toppled_t, self.topple_side = None, 0
        self.scrape, self.scur = [], None
        self.max_tilt = 0.0


def _snap(v):
    s = v.st
    return dict(vx=float(s["vx"][0]), vy=float(s["vy"][0]), r=float(s["r"][0]), delta=float(s["delta"][0]),
                x=float(s["x"][0]), y=float(s["y"][0]), psi=float(s["psi"][0]),
                ke=float(kinetic_energy(v.spec, s)[0]))


def _zone(angle):
    a = np.degrees(wrap(angle))
    if abs(a) <= 35: return "front"
    if abs(a) >= 145: return "rear"
    return "left side" if a > 0 else "right side"


def _roll_impulse(v, Px, Py, hc):
    """Collision impulse applied at height hc changes the roll rate (about the tipping axis)."""
    sp = v.spec
    psi = float(v.st["psi"][0])
    Pyb = -np.sin(psi) * Px + np.cos(psi) * Py            # body-lateral impulse
    if sp.slide_on_hit:                                   # two-wheelers / cycles: remember fall side
        if abs(Pyb) > 1e-6 and v.topple_side == 0:
            v.topple_side = int(np.sign(Pyb))
        return
    if sp.b_tip <= 0:
        return
    r2 = sp.b_tip ** 2 + sp.h ** 2
    Ip = sp.ix + sp.mass * r2
    v.om += Pyb * (hc - sp.h) / Ip


def _roll_step(v, dt, t):
    """1-DOF roll / tip-over about the outer wheel line (tripped by the lateral ground force).
    psi'' = m (D z - g d) / Ip with CG height z and inboard offset d measured from the pivot.
    Past the tipping angle beta = atan(b/h) gravity tips it the rest of the way."""
    sp, st = v.spec, v.st
    if sp.slide_on_hit:                                   # inverted-pendulum fall after impact
        if st["slide"][0]:
            if v.topple_side == 0:
                v.topple_side = 1
            s = v.topple_side
            if abs(v.phi) < 1e-9:
                v.phi = 0.12 * s
            if abs(v.phi) < np.pi / 2:
                v.om += (G / max(sp.h, 0.3)) * np.sin(v.phi) * dt
                v.phi += v.om * dt
                if abs(v.phi) >= np.pi / 2:
                    v.phi, v.om = (np.pi / 2) * s, 0.0
                    if v.toppled_t is None:
                        v.toppled_t = t
        v.max_tilt = max(v.max_tilt, abs(v.phi))
        return
    if sp.b_tip <= 0:
        return
    m, h, b = sp.mass, sp.h, sp.b_tip
    r = np.hypot(b, h); beta = np.arctan2(b, h); Ip = sp.ix + m * r * r
    if v.toppled_t is not None:                           # on its side, sliding
        return
    D = -float(st["ay"][0])                               # specific force toward +y tips toward +y
    if abs(v.phi) < 1e-9 and abs(v.om) < 1e-9:
        if abs(D) * h <= G * b:                           # wheels stay on the ground
            return
        s = np.sign(D)
    else:
        s = np.sign(v.phi) if abs(v.phi) > 1e-9 else np.sign(v.om)
    psi_ = abs(v.phi); om_ = v.om * s
    lifted = psi_ < beta
    Dd = s * D if lifted else 0.0                         # once past balance, no tyre force left
    acc = m * (Dd * r * np.cos(beta - psi_) - G * r * np.sin(beta - psi_)) / Ip
    om_ += acc * dt
    psi_ += om_ * dt
    if psi_ <= 0:
        psi_, om_ = 0.0, 0.0
    if psi_ >= np.pi / 2:
        psi_, om_ = np.pi / 2, 0.0
        v.toppled_t, v.topple_side = t, int(s)
        st["slide"][:] = True
    v.phi, v.om = s * psi_, s * om_
    v.max_tilt = max(v.max_tilt, psi_)


def run_scene(scene, dt=0.002, rec_dt=0.04, mark_every=4, verbose=False):
    veh = [_Veh(c) for c in scene.vehicles]
    nveh = len(veh)
    frames, events, open_by_pair = [], [], {}
    t, step, quiet = 0.0, 0, 0.0
    rec_every = int(round(rec_dt / dt))
    t_stop = None

    while t < scene.t_end + 1e-9:
        for v in veh:
            c, st = v.cfg, v.st
            if v.hit:
                steer = float(st["delta"][0])
                a = (-1.5 * G if c.post_impact == "brake" else
                     c.accel_fn(t, st, veh) if c.post_impact == "script" else 0.0)
            else:
                steer = c.steer_fn(t, st, veh)
                a = c.accel_fn(t, st, veh)
            rec = {} if step % mark_every == 0 else None
            vehicle_step(v.spec, st, steer, a, scene.mu, dt, c.abs_on, rec)
            v._last_a = a
            _roll_step(v, dt, t)
            if rec is not None:
                spd = float(np.hypot(st["vx"][0], st["vy"][0]))
                for k in range(len(v.cur)):
                    if rec["marking"][0, k] and spd > 0.8 and not st["slide"][0]:
                        if v.cur[k] is None:
                            v.cur[k] = []
                            v.marks[k].append(v.cur[k])
                        v.cur[k].append([round(float(rec["wx"][0, k]), 2),
                                         round(float(rec["wy"][0, k]), 2), round(t, 2)])
                    else:
                        v.cur[k] = None

        # ---- collisions
        for i in range(nveh):
            for j in range(i + 1, nveh):
                A, B = veh[i], veh[j]
                d = np.hypot(A.st["x"][0] - B.st["x"][0], A.st["y"][0] - B.st["y"][0])
                if d > A.spec.radius + B.spec.radius + 0.5:
                    continue
                res = sat_contact(corners(A.spec, A.st), corners(B.spec, B.st))
                if res is None:
                    continue
                n, depth, P = res
                key = (i, j)
                ev = open_by_pair.get(key)
                if ev is None or t - ev["t_last"] > 0.5:
                    pre = (_snap(A), _snap(B))
                    ev = dict(i=i, j=j, t0=t, t_last=t, pre=pre, J=0.0, n=n.copy(), P=P.copy(),
                              depth=depth, post=pre)
                    ev["vc0"] = float(((pre[0]["vx"] - pre[0]["r"] * (P[1] - pre[0]["y"])) -
                                       (pre[1]["vx"] - pre[1]["r"] * (P[1] - pre[1]["y"]))) * n[0] +
                                      ((pre[0]["vy"] + pre[0]["r"] * (P[0] - pre[0]["x"])) -
                                       (pre[1]["vy"] + pre[1]["r"] * (P[0] - pre[1]["x"]))) * n[1])
                    ev["psi0"] = (pre[0]["psi"], pre[1]["psi"])
                    events.append(ev)
                    open_by_pair[key] = ev
                    ev["contacts"] = []
                vA0 = (A.st["vx"][0], A.st["vy"][0]); vB0 = (B.st["vx"][0], B.st["vy"][0])
                Jn = apply_impulse(A.spec, A.st, B.spec, B.st, P, n,
                                   scene.e if ev["vc0"] > 0.8 else 0.0, scene.mu_c)
                hc = min(A.spec.bumper, B.spec.bumper) if min(A.spec.bumper, B.spec.bumper) > 0 else 0.5
                for V_, v0_ in ((A, vA0), (B, vB0)):
                    _roll_impulse(V_, V_.spec.mass * (V_.st["vx"][0] - v0_[0]),
                                  V_.spec.mass * (V_.st["vy"][0] - v0_[1]), hc)
                # keep the contact in a resting state: no restitution once nearly stuck
                ev["J"] += float(Jn[0])
                # positional correction
                wA = B.spec.mass / (A.spec.mass + B.spec.mass)
                A.st["x"][0] -= n[0] * depth * 0.9 * wA
                A.st["y"][0] -= n[1] * depth * 0.9 * wA
                B.st["x"][0] += n[0] * depth * 0.9 * (1 - wA)
                B.st["y"][0] += n[1] * depth * 0.9 * (1 - wA)
                ev["t_last"] = t
                ev["post"] = (_snap(A), _snap(B))
                if len(ev["contacts"]) < 1:
                    ev["contacts"].append([float(P[0]), float(P[1])])
                for V in (A, B):
                    if not V.hit:
                        V.hit = True
                        if V.spec.slide_on_hit:
                            V.st["slide"][:] = True
                        V.hit_t = t
        # ---- record
        if step % rec_every == 0:
            row = []
            for v in veh:
                s = v.st
                row.append([round(float(s["x"][0]), 3), round(float(s["y"][0]), 3),
                            round(float(s["psi"][0]), 4),
                            round(float(np.hypot(s["vx"][0], s["vy"][0])), 3),
                            round(float(s["delta"][0]), 3),
                            round(float(s["r"][0]), 3),
                            round(float(v._last_a), 2),
                            1 if v.st["slide"][0] else 0,
                            round(float(v.phi), 3)])
                spd = float(np.hypot(v.st["vx"][0], v.st["vy"][0]))
                if v.st["slide"][0] and spd > 0.4:
                    if v.scur is None:
                        v.scur = []; v.scrape.append(v.scur)
                    v.scur.append([round(float(v.st["x"][0]), 2), round(float(v.st["y"][0]), 2), round(t, 2)])
                else:
                    v.scur = None
            frames.append(row)
        # ---- stop criterion
        moving = max(np.hypot(v.st["vx"][0], v.st["vy"][0]) + abs(v.st["r"][0]) for v in veh)
        if t > scene.t_min and moving < 0.08:
            quiet += dt
        else:
            quiet = 0.0
        if quiet > 0.8 and t_stop is None:
            t_stop = t + 1.2
        if t_stop is not None and t >= t_stop:
            break
        t += dt
        step += 1

    return _package(scene, veh, frames, events, rec_dt)


def _package(scene, veh, frames, events, rec_dt):
    out_events = []
    events = sorted(events, key=lambda e: e["t0"])
    for k, ev in enumerate(events):
        A, B = veh[ev["i"]], veh[ev["j"]]
        pa, pb = ev["pre"]
        qa, qb = ev["post"]
        n = ev["n"]
        dva = np.hypot(qa["vx"] - pa["vx"], qa["vy"] - pa["vy"])
        dvb = np.hypot(qb["vx"] - pb["vx"], qb["vy"] - pb["vy"])
        spdA, spdB = np.hypot(pa["vx"], pa["vy"]), np.hypot(pb["vx"], pb["vy"])
        angA = np.arctan2(n[1], n[0]) - pa["psi"]
        angB = np.arctan2(-n[1], -n[0]) - pb["psi"]
        out_events.append(dict(
            n=k + 1, t=round(ev["t0"], 3), a=A.cfg.label, b=B.cfg.label,
            P=[round(float(ev["P"][0]), 2), round(float(ev["P"][1]), 2)],
            normal=[round(float(n[0]), 3), round(float(n[1]), 3)],
            closing_kmh=round(ev["vc0"] * 3.6, 1),
            speed_a_kmh=round(spdA * 3.6, 1), speed_b_kmh=round(spdB * 3.6, 1),
            dv_a_kmh=round(dva * 3.6, 1), dv_b_kmh=round(dvb * 3.6, 1),
            energy_kj=round((pa["ke"] + pb["ke"] - qa["ke"] - qb["ke"]) / 1000, 1),
            zone_a=_zone(angA), zone_b=_zone(angB),
            angle_a_deg=round(float(np.degrees(wrap(angA))), 1),
            angle_b_deg=round(float(np.degrees(wrap(angB))), 1),
            J=round(ev["J"], 0),
            e=scene.e,
            pre=dict(a=dict(x=pa["x"], y=pa["y"], psi=pa["psi"], vx=pa["vx"], vy=pa["vy"], r=pa["r"]),
                     b=dict(x=pb["x"], y=pb["y"], psi=pb["psi"], vx=pb["vx"], vy=pb["vy"], r=pb["r"])),
            ia=ev["i"], ib=ev["j"], t0=ev["t0"],
            Pf=[float(ev["P"][0]), float(ev["P"][1])], nf=[float(n[0]), float(n[1])],
        ))
    marks = []
    for vi, v in enumerate(veh):
        for k, pl in enumerate(v.marks):
            if len(pl) and any(len(p) > 1 for p in pl):
                for p in pl:
                    if len(p) > 1:
                        marks.append(dict(v=vi, w=k, pts=p))
        for p in v.scrape:
            if len(p) > 1:
                marks.append(dict(v=vi, w=-1, pts=p))
    final = [dict(x=float(v.st["x"][0]), y=float(v.st["y"][0]), psi=float(v.st["psi"][0]))
             for v in veh]
    from .safety import analyse
    safety = analyse(veh, events, out_events)
    return dict(scene=scene, frames=frames, events=out_events, marks=marks, final=final,
                dt=rec_dt, veh=veh, safety=safety)
