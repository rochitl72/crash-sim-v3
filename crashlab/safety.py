"""Damage, crush and restraint-system logic, evaluated per impact from the impulse results.

Crush depth: Campbell linear crush model, F/w = A + B*C. The energy absorbed by one vehicle over
a crush width w is E = w*(A + B*C)^2 / (2B) (incl. the elastic A^2/2B term), so
C = (sqrt(2*B*E/w) - A) / B. The dissipated energy of an impact is split between the two
vehicles in proportion to compliance 1/B (equal contact force, springs in series).

Collision Deformation Classification: simplified 7-character code after SAE J224
(clock direction of force, deformed area, horizontal location, vertical location, type, extent).

Airbag logic: a simple sensor model on the vehicle's ΔV components (thresholds typical of
published deployment ranges: frontal ~18-25 km/h barrier-equivalent, side ~12-15 km/h).
All of this is indicative engineering, not a certified injury or damage assessment."""
import numpy as np
from .drivers import wrap

KMH = 3.6
FRONT_THR, SIDE_THR = 18.0, 12.0     # km/h airbag thresholds (indicative)
SEV = [(0.03, "scuff"), (0.10, "dent"), (0.25, "moderate"), (0.45, "severe"), (9e9, "crushed")]


def sev(d):
    for lim, name in SEV:
        if d < lim:
            return name
    return "crushed"


def _body(vx, vy, psi):
    c, s = np.cos(psi), np.sin(psi)
    return c * vx + s * vy, -s * vx + c * vy


def _face(spec, pb):
    """Nearest face of the body rectangle to a body-frame point, and position along that face."""
    off, L, W = spec.rect_off, spec.length, spec.width
    x, y = pb[0] - off, pb[1]
    d = {"front": abs(L / 2 - x), "rear": abs(L / 2 + x), "left": abs(W / 2 - y), "right": abs(W / 2 + y)}
    # corners: prefer the face the normal points through (handled by caller via n_body tie-break)
    return d, x / (L / 2), y / (W / 2)


def _parts(spec, face, u, depth):
    """Named parts by vehicle style, face and position along the face (u in -1..1)."""
    st = spec.style
    out = []
    add = lambda name, k=1.0: out.append(dict(name=name, sev=sev(depth * k), depth=round(depth * k, 3)))
    if st in ("hatch", "sedan", "csuv", "suv"):
        if face == "front":
            add("Front bumper")
            if u > 0.25: add("Left headlamp", .8); add("Left front fender", .6)
            elif u < -0.25: add("Right headlamp", .8); add("Right front fender", .6)
            else: add("Grille", .9)
            if depth > .12: add("Bonnet", .7); add("Radiator & condenser", .6)
            if depth > .35: add("Engine bay / crash rails", .5)
            if depth > .6: add("Windscreen & A-pillar", .3); add("Footwell intrusion", .25)
        elif face == "rear":
            add("Rear bumper")
            if u > 0.25: add("Left tail lamp", .8); add("Left rear quarter", .6)
            elif u < -0.25: add("Right tail lamp", .8); add("Right rear quarter", .6)
            add("Tailgate / boot lid", .7 if st != "sedan" else .6)
            if depth > .3: add("Rear floor & spare-wheel well", .5)
            if depth > .6: add("Rear seat compartment intrusion", .3)
        else:
            sd = "Left" if face == "left" else "Right"
            # u = x/(L/2) along the body: +1 front, -1 rear
            if u > 0.45: add(f"{sd} front fender"); add(f"{sd} front wheel & suspension", .7)
            elif u > 0.0: add(f"{sd} front door"); add(f"{sd} mirror", .5)
            elif u > -0.4: add(f"{sd} rear door" if st != "hatch" or u > -0.2 else f"{sd} rear quarter")
            else: add(f"{sd} rear quarter panel"); add(f"{sd} rear wheel", .7)
            if abs(u) < 0.5 and depth > .12: add("B-pillar", .7)
            if depth > .18 and abs(u) < 0.6: add(f"{sd} cabin intrusion", .8)
            if depth > .3: add("Side sill / floor", .5)
    elif st in ("truck", "minitruck", "bus"):
        if face == "front":
            add("Front bumper / under-run guard"); add("Grille & cab front", .8)
            if u > 0.3: add("Left headlamp", .8)
            elif u < -0.3: add("Right headlamp", .8)
            if depth > .2: add("Windscreen", .5); add("Radiator", .6)
            if depth > .4 and st == "minitruck": add("Cab footwell intrusion (cab-over)", .6)
        elif face == "rear":
            add("Rear under-run guard"); add("Tail lamps", .7)
            add("Cargo body tailgate" if st != "bus" else "Rear panel & engine cover", .7)
        else:
            sd = "Left" if face == "left" else "Right"
            if st == "bus":
                add(f"{sd} side skirt panel"); add(f"{sd} window glass", .4)
                if u > 0.6: add("Front door / driver area", .7)
            else:
                add(f"{sd} side guard" if st == "truck" else f"{sd} load body side")
                if u > 0.4: add(f"{sd} cab door", .8)
                if st == "truck" and -0.2 < u < 0.4: add(f"{sd} fuel tank", .6)
    elif st == "auto":
        if face == "front":
            add("Front cowl / fairing"); add("Headlamp", .8); add("Front fork & suspension", .7)
            if depth > .15: add("Windscreen", .6); add("Driver foot area intrusion", .6)
        elif face == "rear":
            add("Rear panel / engine cover"); add("Tail lamps", .7)
            if depth > .2: add("Engine & transmission", .5)
        else:
            sd = "Left" if face == "left" else "Right"
            add(f"{sd} side panel / canvas"); add(f"{sd} passenger opening", .8)
            if u < -0.2: add(f"{sd} rear wheel", .8)
            add("Roof frame", .4)
    elif st in ("moto", "scooter"):
        if face == "front":
            add("Front wheel"); add("Front fork", .9); add("Headlamp & handlebar", .7)
        elif face == "rear":
            add("Rear wheel"); add("Tail lamp", .7); add("Rear mudguard", .7)
        else:
            sd = "Left" if face == "left" else "Right"
            add(f"{sd} crash guard / leg shield"); add("Fuel tank" if st == "moto" else "Floorboard", .7)
            add(f"{sd} footrest & engine cover", .7)
    else:  # cycle
        if face == "front":
            add("Front wheel (buckled)"); add("Fork", .8); add("Handlebar", .6)
        elif face == "rear":
            add("Rear wheel"); add("Carrier", .7)
        else:
            add("Frame (top / down tube)"); add("Crank & pedals", .7); add("Wheels", .5)
    # dedupe keep max
    seen = {}
    for p in out:
        if p["name"] not in seen or p["depth"] > seen[p["name"]]["depth"]:
            seen[p["name"]] = p
    return list(seen.values())


def _cdc(clock, face, u, width, depth, sideswipe, spec):
    area = {"front": "F", "rear": "B", "left": "L", "right": "R"}[face]
    if face in ("front", "rear"):
        span = width / max(spec.width, 0.3)
        if span > 0.8: hz = "D"
        elif u > 0.33: hz = "L" if span < 0.45 else "Y"
        elif u < -0.33: hz = "R" if span < 0.45 else "Z"
        else: hz = "C"
        if face == "rear":   # viewed from behind, left/right stay vehicle-based
            pass
    else:
        span = width / max(spec.length, 0.5)
        if span > 0.8: hz = "D"
        elif u > 0.45: hz = "F" if span < 0.35 else "Y"
        elif u < -0.4: hz = "B" if span < 0.35 else "Z"
        else: hz = "P"
    typ = "S" if sideswipe else ("W" if width > 0.41 else "N")
    ext = int(min(9, max(1, np.ceil(depth / 0.15))))
    return f"{clock:02d}{area}{hz}E{typ}{ext}"


def analyse(veh, events_raw, out_events):
    """Attach damage / airbag info to each output event and build per-vehicle summaries."""
    per = [dict(damage=[], airbags_deployed=[], pulses=[]) for _ in veh]
    for ev, oe in zip(events_raw, out_events):
        ia, ib = ev["i"], ev["j"]
        P = np.asarray(ev["P"], float); n = np.asarray(ev["n"], float)
        E = max(oe["energy_kj"] * 1000.0, 0.0)
        info = []
        for k, (vi, nn) in enumerate(((ia, n), (ib, -n))):
            V = veh[vi]; sp = V.spec
            pre, post = ev["pre"][k], ev["post"][k]
            psi = pre["psi"]
            pb = _body(P[0] - pre["x"], P[1] - pre["y"], psi)
            nb = _body(nn[0], nn[1], psi)
            d, ux, uy = _face(sp, pb)
            # face: nearest edge, tie-broken by the normal direction
            ang = np.degrees(np.arctan2(nb[1], nb[0]))
            by_n = "front" if abs(ang) <= 45 else "rear" if abs(ang) >= 135 else ("left" if ang > 0 else "right")
            near = min(d, key=d.get)
            face = by_n if d[by_n] < d[near] + 0.35 else near
            u = uy if face in ("front", "rear") else ux
            other = veh[ib if k == 0 else ia].spec
            # direction the force comes from, as a clock (12 = from ahead)
            clock = int(round((12 - ang / 30.0))) % 12 or 12
            dvx, dvy = post["vx"] - pre["vx"], post["vy"] - pre["vy"]
            dvl, dvt = _body(dvx, dvy, psi)
            info.append(dict(vi=vi, sp=sp, face=face, u=float(np.clip(u, -1, 1)), pb=pb, clock=clock,
                             dvl=dvl * KMH, dvt=dvt * KMH, other=other, nb=nb))
        # crush widths and energy split
        def face_w(sp, face):
            return sp.width * 0.85 if face in ("front", "rear") else sp.length * 0.6
        w = min(face_w(info[0]["sp"], info[0]["face"]), face_w(info[1]["sp"], info[1]["face"]))
        w = max(w, 0.25)
        AB = [x["sp"].crush["side" if x["face"] in ("left", "right") else x["face"]] for x in info]
        comp = np.array([1.0 / AB[0][1], 1.0 / AB[1][1]])
        Es = E * comp / comp.sum()
        # sideswipe if the tangential relative velocity dominates the closing velocity
        vt = abs(oe["closing_kmh"])
        oe["damage"] = []
        for k, x in enumerate(info):
            A_, B_ = AB[k]
            C = max((np.sqrt(2 * B_ * Es[k] / w) - A_) / B_, 0.0)
            mx = x["sp"].max_crush[0 if x["face"] == "front" else 2 if x["face"] == "rear" else 1]
            C = float(min(C, mx))
            ss = vt < 8 and abs(x["dvt"]) + abs(x["dvl"]) > 0 and x["face"] in ("left", "right") and abs(x["dvl"]) > 2 * abs(x["dvt"])
            parts = _parts(x["sp"], x["face"], x["u"], C) if C > 0.005 else []
            dmg = dict(event=oe["n"], t=oe["t"], face=x["face"], u=round(x["u"], 3),
                       pb=[round(float(x["pb"][0]), 3), round(float(x["pb"][1]), 3)],
                       nb=[round(float(x["nb"][0]), 3), round(float(x["nb"][1]), 3)],
                       depth=round(C, 3), width=round(float(w), 2), energy_kj=round(Es[k] / 1000, 1),
                       severity=sev(C), clock=x["clock"],
                       cdc=_cdc(x["clock"], x["face"], x["u"], w, C, ss, x["sp"]),
                       dv_long=round(float(x["dvl"]), 1), dv_lat=round(float(x["dvt"]), 1), parts=parts,
                       other_bumper=x["other"].bumper, other_style=x["other"].style)
            oe["damage"].append(dmg)
            vi = x["vi"]
            per[vi]["damage"].append(dmg)
            # pulse duration: stiffer/heavier partner -> shorter pulse (typical 60-120 ms)
            Tp = 0.11 if x["face"] in ("front", "rear") else 0.07
            if x["other"].kind in ("truck", "bus"): Tp *= 0.8
            per[vi]["pulses"].append(dict(t=oe["t"], dvl=round(float(x["dvl"]), 2), dvt=round(float(x["dvt"]), 2),
                                          Tp=Tp, face=x["face"], u=round(x["u"], 3), depth=round(C, 3),
                                          clock=x["clock"], closing=oe["closing_kmh"],
                                          other=x["other"].kind, other_mass=x["other"].mass))
            # ---- airbag sensor logic (once per bag type)
            fitted = x["sp"].airbags
            got = {a["type"] + a.get("side", "") for a in per[vi]["airbags_deployed"]}
            fire = []
            if "front" in fitted and -x["dvl"] >= FRONT_THR and x["clock"] in (10, 11, 12, 1, 2):
                fire.append(dict(type="front", side=""))
                if "knee" in fitted: fire.append(dict(type="knee", side=""))
            if x["face"] in ("left", "right") and abs(x["dvt"]) >= SIDE_THR:
                for typ in ("side", "curtain"):
                    if typ in fitted: fire.append(dict(type=typ, side=x["face"]))
            for f in fire:
                if f["type"] + f["side"] not in got:
                    f["t"] = round(oe["t"] + (0.015 if f["type"] != "side" else 0.008), 3)
                    f["event"] = oe["n"]
                    per[vi]["airbags_deployed"].append(f)
    for vi, V in enumerate(veh):
        p = per[vi]
        p["toppled"] = None if V.toppled_t is None else dict(t=round(V.toppled_t, 2), side=V.topple_side)
        p["max_tilt_deg"] = round(float(np.degrees(V.max_tilt)), 1)
        # rollover: curtains fire when the vehicle tips over
        if V.toppled_t is not None and "curtain" in V.spec.airbags:
            for sd in ("left", "right"):
                if not any(a["type"] == "curtain" and a["side"] == sd for a in p["airbags_deployed"]):
                    p["airbags_deployed"].append(dict(type="curtain", side=sd, t=round(V.toppled_t - 0.3, 2),
                                                      event=0, rollover=True))
        p["max_depth"] = max([d["depth"] for d in p["damage"]], default=0.0)
        p["severity"] = sev(p["max_depth"]) if p["damage"] else "none"
    return per
