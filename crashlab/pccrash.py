"""Read a PC-Crash "values output" text export and turn it into a CrashLab scene.

The parser is deterministic: every number in the simulation comes from the file, never from the AI.
The AI layer (crashlab/ai.py) only classifies vehicles, reads the story and fills clearly-flagged gaps.

Sequence interpretation (stated in the UI): each PC-Crash sequence holds one BRAKE and one STEERING
instruction. Sequences run one after another; a sequence lasts its "maximum stopping time"; the
brake deceleration applies for the whole sequence; the steering angle ramps linearly to the new
angle over the "steering time" and is then held. Inputs keep running after the impact.
"""
import re
import numpy as np
from . import vehicles as V
from .engine import Scene, VehicleCfg

STYLE_FACTORY = {"hatch": V.hatchback, "sedan": V.sedan, "csuv": V.compact_suv, "suv": V.suv,
                 "minitruck": V.mini_truck, "truck": V.truck, "bus": V.bus, "auto": V.auto_rickshaw,
                 "moto": V.motorcycle, "scooter": V.scooter, "cycle": V.bicycle}
KIND = {"hatch": "car", "sedan": "car", "csuv": "car", "suv": "car", "minitruck": "truck", "truck": "truck",
        "bus": "bus", "auto": "three-wheeler", "moto": "two-wheeler", "scooter": "two-wheeler",
        "cycle": "cycle", "barricade": "object", "object": "object"}
COLORS = ["#2ec4b6", "#e76f51", "#ffb703", "#8ecae6", "#c77dff", "#80ed99", "#f15bb5"]


def _key(k):
    return re.sub(r"[^a-z0-9]+", "_", k.lower()).strip("_")


def _val(v):
    v = v.strip()
    if v == "":
        return None
    if v.lower() == "no":
        return False
    if v.lower() == "yes":
        return True
    try:
        return float(v)
    except ValueError:
        return v


def parse(text):
    """Return dict(vehicles=[...], warnings=[...]). Each vehicle: name, database, record, start, end,
    input (dicts keyed by normalised label) and sequence (list of steps)."""
    veh, warn = [], []
    lines = text.replace("\r", "").split("\n")
    mode, cols, seq_v, step, brake_hdr = None, [], None, None, False

    def find(name, record):
        for i, v in enumerate(veh):
            if v["name"] == name and (record is None or v["record"] in (None, record)):
                return i
        return None

    i = 0
    while i < len(lines):
        raw = lines[i]; line = raw.strip(); i += 1
        if not line:
            continue
        up = line.upper()
        if up in ("START VALUES", "END VALUES", "INPUT VALUES", "SEQUENCES"):
            if mode == "seq" and up == "START VALUES":
                step = dict(type="start"); veh[seq_v]["sequence"].append(step); continue
            if up == "INPUT VALUES":
                mode = "input_hdr"
            elif up == "SEQUENCES":
                mode = "seq"
            else:
                mode = "start" if up.startswith("START") else "end"
            continue
        if mode == "seq":
            m = re.match(r"^(\d+)\s+(.+?):\s*$", line)
            if m:
                seq_v = int(m.group(1)) - 1
                if seq_v >= len(veh):
                    warn.append(f"Sequence {m.group(1)} has no matching vehicle"); seq_v = None
                step = None; continue
            if seq_v is None:
                continue
            if up in ("BRAKE", "STEERING"):
                step = dict(type=up.lower()); veh[seq_v]["sequence"].append(step); brake_hdr = False; continue
            if line.lower().startswith("brake force"):
                brake_hdr = True; step.setdefault("brake_force", {}); continue
            if ":" in line and step is not None:
                k, rest = line.split(":", 1)
                vals = [x for x in rest.split("\t") if x.strip() != ""]
                v0 = _val(vals[0]) if vals else None
                kk = _key(k)
                if brake_hdr and kk.startswith("axle"):
                    step["brake_force"][kk] = v0
                elif kk.startswith("new_steering_angle"):
                    step["steer_axle1_deg"] = v0
                elif kk == "axle_2":
                    step["steer_axle2_deg"] = v0
                else:
                    step[kk] = v0
            continue
        if ":" not in line:
            continue
        k, rest = raw.split(":", 1)
        vals = rest.split("\t")[1:] if rest.startswith("\t") or rest.startswith(" \t") else rest.split("\t")
        vals = [x for x in (rest.strip("\n").split("\t"))]
        vals = [x for x in vals if x.strip() != "" or False]
        kk = _key(k)
        if kk == "vehicle":
            names = [x.strip() for x in rest.split("\t") if x.strip()]
            recs = None
            # look ahead for Database / RecordID lines to disambiguate
            for j in range(i, min(i + 3, len(lines))):
                if lines[j].strip().lower().startswith("recordid"):
                    recs = [x.strip() for x in lines[j].split(":", 1)[1].split("\t") if x.strip()]
            cols = []
            for n, name in enumerate(names):
                rec = recs[n] if recs and n < len(recs) else None
                idx = find(name, rec)
                if idx is None:
                    if mode == "input_hdr":
                        warn.append(f"Input values for unknown vehicle '{name}'")
                    veh.append(dict(name=name, database=None, record=rec, start={}, end={}, input={}, sequence=[]))
                    idx = len(veh) - 1
                cols.append(idx)
            if mode == "input_hdr":
                mode = "input"
            continue
        if kk in ("database", "recordid"):
            vs = [x.strip() for x in rest.split("\t") if x.strip()]
            for n, c in enumerate(cols):
                if n < len(vs):
                    veh[c]["database" if kk == "database" else "record"] = vs[n]
            continue
        if kk == "characters":
            continue
        vs = [x.strip() for x in rest.split("\t")]
        vs = [x for x in vs if x != ""]
        target = {"start": "start", "end": "end", "input": "input"}.get(mode)
        if not target:
            continue
        for n, c in enumerate(cols):
            if n < len(vs):
                veh[c][target][kk] = _val(vs[n])
        # skip "(1.32)" alternative-value lines (no colon) automatically
    for v in veh:
        if not v["input"]:
            warn.append(f"No INPUT VALUES for {v['name']}")
        if not v["start"]:
            warn.append(f"No START VALUES for {v['name']}")
    return dict(vehicles=veh, warnings=warn)


def g(d, *keys, default=None):
    """First present key whose normalised name starts with any of the given prefixes."""
    for p in keys:
        for k, v in d.items():
            if k.startswith(p) and v is not None and not isinstance(v, str):
                return v
    return default


# ------------------------------------------------------------------ vehicle classification fallback
def guess_style(v):
    n = v["name"].lower(); I = v["input"]
    L = g(I, "length_m", default=4.2); H = g(I, "height_m", default=1.5); m = g(I, "total_weight", "mass_empty", default=1200)
    if any(w in n for w in ("barricade", "barrier", "cone", "pole", "bollard", "drum")):
        return "barricade"
    if any(w in n for w in ("cycle", "bicycle")) and m < 60:
        return "cycle"
    if any(w in n for w in ("scooter", "activa", "jupiter", "access")):
        return "scooter"
    if m < 350 or any(w in n for w in ("bike", "motorcycle", "splendor", "pulsar", "bullet", "enfield")):
        return "moto"
    if any(w in n for w in ("auto", "rickshaw", "ape", "re compact")):
        return "auto"
    if any(w in n for w in ("bus", "volvo b", "marcopolo")) or (L > 9 and H > 2.8):
        return "bus"
    if any(w in n for w in ("eicher", "tata lpt", "ashok", "bharatbenz", "truck", "tipper", "lorry")) or L > 6.5:
        return "truck"
    if any(w in n for w in ("ace", "super carry", "dost", "pickup", "bolero pik")) or (L < 4.2 and H > 1.75):
        return "minitruck"
    if H >= 1.7 or any(w in n for w in ("xuv", "scorpio", "fortuner", "innova", "safari", "harrier")):
        return "suv"
    if H >= 1.58 or any(w in n for w in ("creta", "nexon", "brezza", "venue", "seltos", "sonet")):
        return "csuv"
    if L < 4.0:
        return "hatch"
    return "sedan"


# ------------------------------------------------------------------ spec from measured inputs
def make_spec(v, style, color, airbags=None):
    I = v["input"]
    base = STYLE_FACTORY.get(style, V.sedan)() if style != "barricade" else None
    L = g(I, "length_m", default=base.length if base else 1.8)
    W = g(I, "width_m", default=base.width if base else 0.6)
    H = g(I, "height_m", default=base.height if base else 1.2)
    wb = g(I, "wheelbase_m", default=base.wheelbase if base else 1.0)
    a = g(I, "distance_c_g_front_axle", default=wb * 0.45)
    b = max(wb - a, 0.05)
    fo = g(I, "front_overhang", default=max(0.1, (L - wb) * 0.5))
    tf = g(I, "front_track", default=W * 0.85); tr = g(I, "rear_track", default=W * 0.85)
    m = g(I, "total_weight", "mass_empty", default=base.mass if base else 100.0)
    h = g(I, "c_g_height", default=base.h if base else 0.5)
    izz = g(I, "yaw_moment_of_inertia", default=m * (L * L + W * W) / 12.0)
    nax = int(g(I, "number_of_axles", default=2))
    kind = KIND.get(style, "car")
    if style == "barricade":
        tf = tr = min(tf, W * 0.9)
        sp = V._make(v["name"], "object", m, izz, L, W, a, b, h, 2, 2, tf, tr, "rear", 0.5, 4.0, 0.9, color,
                     slide_on_hit=True, slide_mu=0.6, height=H, style="barricade", example="Road barricade",
                     crush=dict(front=(3e3, 50e3), side=(3e3, 50e3), rear=(3e3, 50e3)), bumper=0.6,
                     max_crush=(0.4, 0.3, 0.4))
    else:
        two = kind in ("two-wheeler", "cycle")
        nf, nr = (1, 1) if two else ((1, 2) if style == "auto" else (2, 2))
        sp = V._make(v["name"], kind, m, izz, L, W, a, b, h, nf, nr, tf if nf == 2 else 0.0, tr if nr == 2 else 0.0,
                     "front" if kind == "car" else "rear", 0.65, base.c_alpha if hasattr(base, "c_alpha") else 8.5,
                     0.7 if kind == "car" else 3.0, color, slide_on_hit=base.slide_on_hit, max_steer=base.max_steer,
                     slide_mu=base.slide_mu, height=H, style=style, example=v["name"],
                     crush=base.crush, airbags=tuple(airbags if airbags is not None else base.airbags),
                     bumper=base.bumper, max_crush=base.max_crush)
    # geometric centre of the body rectangle relative to the CG (front bumper = a + front overhang)
    sp.rect_off = (a + fo) - L / 2.0
    sp.radius = 0.5 * np.hypot(L, W) + abs(sp.rect_off)
    ix = g(I, "roll_moment_of_inertia")
    if ix:
        sp.ix = float(ix)
    sp.name = short_name(v["name"])
    sp.example = v["name"]
    return sp


def short_name(n):
    n = re.sub(r"\(\d{4}\)", "", n).strip()
    w = n.split()
    return " ".join(w[:2]) if w else "Vehicle"


# ------------------------------------------------------------------ sequence drivers
def timeline(v):
    """Turn the PC-Crash sequence list into [(t0, t1, decel m/s2, steer_from_deg, steer_to_deg, steer_time)]."""
    seq = v["sequence"]; out = []; t = 0.0; cur_steer = 0.0
    brakes = [s for s in seq if s["type"] == "brake"]; steers = [s for s in seq if s["type"] == "steering"]
    n = max(len(brakes), len(steers))
    for k in range(n):
        br = brakes[k] if k < len(brakes) else None
        st = steers[k] if k < len(steers) else None
        dur = (br or {}).get("maximum_stopping_time_s") or (st or {}).get("steering_time_s") or 1.0
        dec = abs((br or {}).get("specified_brake_acceleration_m_s") or 0.0)
        if br and dec == 0 and br.get("brake_force"):
            dec = 0.0
        to = st.get("steer_axle1_deg", cur_steer) if st else cur_steer
        stt = st.get("steering_time_s", dur) if st else 0.0
        out.append(dict(t0=round(t, 3), t1=round(t + dur, 3), decel=dec, steer_from=cur_steer, steer_to=to,
                        steer_time=stt, brake_force=(br or {}).get("brake_force")))
        t += dur; cur_steer = to
    return out


def make_drivers(tl):
    def steer(t, st, veh):
        if not tl:
            return 0.0
        for ph in tl:
            if t < ph["t1"] or ph is tl[-1]:
                if t < ph["t0"]:
                    continue
                f = 1.0 if ph["steer_time"] <= 0 else min(1.0, (t - ph["t0"]) / ph["steer_time"])
                return np.radians(ph["steer_from"] + (ph["steer_to"] - ph["steer_from"]) * f)
        return np.radians(tl[-1]["steer_to"])

    def accel(t, st, veh):
        if not tl:
            return 0.0
        for ph in tl:
            if ph["t0"] <= t < ph["t1"]:
                return -ph["decel"]
        return -tl[-1]["decel"]          # the last instruction stays in force
    return steer, accel


def brake_front_share(tl):
    for ph in tl:
        bf = ph.get("brake_force") or {}
        f = sum(v or 0 for k, v in bf.items() if k.startswith("axle_1")); r = sum(v or 0 for k, v in bf.items() if k.startswith("axle_2"))
        if f + r > 0:
            return f / (f + r)
    return None


# ------------------------------------------------------------------ road
def road_along(x0, y0, psi, back, ahead, left, right, lane_w=3.5):
    c, s = np.cos(psi), np.sin(psi)
    P = lambda u, w: (x0 + c * u - s * w, y0 + s * u + c * w)
    polys = [[P(-back, -right), P(ahead, -right), P(ahead, left), P(-back, left)]]
    lines = [dict(pts=[P(-back, -right + .35), P(ahead, -right + .35)], dash=None, color="#e8e8e8", w=0.16),
             dict(pts=[P(-back, left - .35), P(ahead, left - .35)], dash=None, color="#e8e8e8", w=0.16)]
    w = -right + lane_w
    while w < left - 1.0:
        lines.append(dict(pts=[P(-back, w), P(ahead, w)], dash=[3, 6], color="#e8e8e8", w=0.14)); w += lane_w
    return dict(polys=polys, lines=lines)


# ------------------------------------------------------------------ scene
def build_scene(parsed, ai=None, notes=""):
    ai = ai or {}
    vs = parsed["vehicles"]
    aiv = {int(x.get("index", -1)): x for x in ai.get("vehicles", []) if isinstance(x, dict)}
    cfgs, specs, info = [], [], []
    mu = None
    for i, v in enumerate(vs):
        a = aiv.get(i, {})
        style = a.get("style") if a.get("style") in list(STYLE_FACTORY) + ["barricade"] else guess_style(v)
        color = a.get("color") if isinstance(a.get("color"), str) and re.match(r"^#[0-9a-fA-F]{6}$", a.get("color", "")) else COLORS[i % len(COLORS)]
        if style == "barricade":
            color = "#ff8c1a"
        ab = a.get("airbags")
        ab = [x for x in ab if x in ("front", "side", "curtain", "knee")] if isinstance(ab, list) else None
        sp = make_spec(v, style, color, ab)
        tl = timeline(v)
        bfs = brake_front_share(tl)
        if bfs is not None:
            for w in sp.wheels:
                pass
            sp.wheels[:, 4] = np.where(sp.wheels[:, 0] > 0, bfs / max(1, (sp.wheels[:, 0] > 0).sum()),
                                       (1 - bfs) / max(1, (sp.wheels[:, 0] <= 0).sum()))
        S0 = v["start"]
        x, y = g(S0, "center_of_gravity_x", default=0.0), g(S0, "center_of_gravity_y", default=0.0)
        psi = np.radians(g(S0, "heading_angle", default=0.0))
        spd = g(S0, "velocity_magnitude", default=0.0) / 3.6
        for s in v["sequence"]:
            if s["type"] == "start" and s.get("friction_coefficient"):
                mu = mu or s["friction_coefficient"]
        steer_fn, accel_fn = make_drivers(tl)
        absv = g(v["input"], "abs")
        I = v["input"]
        nfront = int(round((g(I, "mass_of_front_occupants", default=0) or 0) / 80))
        nrear = int(round((g(I, "mass_of_rear_occupants", default=0) or 0) / 80))
        occ = a.get("occupants") if isinstance(a.get("occupants"), list) else None
        if occ is None:
            seats = []
            if sp.kind in ("car", "truck", "bus"):
                seats = ["driver", "front passenger"][:max(1, nfront)] + ["rear left", "rear right", "rear centre"][:nrear]
                occ = [dict(seat=s_, belt=True, assumed=True) for s_ in seats]
            elif sp.kind in ("two-wheeler", "cycle"):
                occ = [dict(seat="rider", helmet=sp.kind == "two-wheeler", assumed=True)]
            elif sp.kind == "three-wheeler":
                occ = [dict(seat="driver", belt=False)]
            else:
                occ = []
        occ = [dict(seat=str(o.get("seat", "driver")), belt=bool(o.get("belt", False)), helmet=bool(o.get("helmet", False)),
                    assumed=bool(o.get("assumed", False))) for o in occ if isinstance(o, dict)]
        label = (a.get("label") or ("V%d" % (i + 1)))[:4]
        note = a.get("note") or f"{v['name']} · start {g(S0, 'velocity_magnitude', default=0):.0f} km/h"
        cfg = VehicleCfg(label, sp, x, y, psi, spd, steer_fn, accel_fn, abs_on=bool(absv), post_impact="script",
                         note=note, occupants=occ)
        if style == "barricade":
            cfg.post_impact = "none"
        cfgs.append(cfg); specs.append(sp)
        info.append(dict(index=i, name=v["name"], style=style, record=v.get("record"), timeline=tl,
                         start=dict(x=x, y=y, heading_deg=float(np.degrees(psi)), speed_kmh=spd * 3.6)))
    sc_ai = ai.get("scene", {}) if isinstance(ai.get("scene"), dict) else {}
    ov = ai.get("overrides", {}) if isinstance(ai.get("overrides"), dict) else {}
    if isinstance(ov.get("mu"), (int, float)) and 0.1 <= ov["mu"] <= 1.2:
        mu = float(ov["mu"])
    mu = mu or 0.8
    # road along the fastest vehicle's heading, wide enough for every object
    lead = max(range(len(cfgs)), key=lambda k: cfgs[k].speed) if cfgs else 0
    L0 = cfgs[lead]
    c, s = np.cos(L0.psi), np.sin(L0.psi)
    us = [(cf.x - L0.x) * c + (cf.y - L0.y) * s for cf in cfgs]
    ws = [-(cf.x - L0.x) * s + (cf.y - L0.y) * c for cf in cfgs]
    lanes = int(sc_ai.get("lanes") or 0) or 0
    left = max(max(ws) + 3.0, 5.25); right = max(-min(ws) + 3.0, 5.25)
    if lanes >= 2:
        left = max(left, 1.75 + 3.5 * (lanes // 2)); right = max(right, 1.75 + 3.5 * (lanes - lanes // 2) - 3.5 + 1.0)
    road = road_along(L0.x, L0.y, L0.psi, max(40.0, -min(us) + 25), max(140.0, max(us) + 70), left, right)
    xs = [cf.x for cf in cfgs]; ys = [cf.y for cf in cfgs]
    pad = 14
    view = [min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad]
    title = (sc_ai.get("title") or "Reconstruction from PC-Crash export")[:90]
    sub = (sc_ai.get("subtitle") or " · ".join(v["name"] for v in vs))[:140]
    story = (sc_ai.get("story") or "Scene recreated from the officer's PC-Crash values export.")[:900]
    tags = [str(t_)[:24] for t_ in (sc_ai.get("tags") or ["PC-Crash import"])][:5]
    scene = Scene("case", title, sub, story, cfgs, road, mu=mu, e=0.12, t_end=float(sc_ai.get("t_end") or 9.0),
                  t_min=3.0, view=view, tags=tags)
    return scene, info
