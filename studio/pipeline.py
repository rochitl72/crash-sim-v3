"""PC-Crash export (+ officer notes) -> AI-described, simulated CrashLab case viewer."""
import json, os, sys, time, uuid
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from crashlab import pccrash as P, ai                     # noqa: E402
from crashlab.engine import run_scene                      # noqa: E402
import build                                              # noqa: E402


def _r(x, n=2):
    return None if x is None else round(float(x), n)


def data_checks(parsed, res):
    """Deterministic checks (no AI)."""
    out = []
    vs = parsed["vehicles"]
    same = all(v["start"] and v["start"] == v["end"] for v in vs)
    if same:
        out.append("The export's END VALUES are identical to its START VALUES, so the file holds no PC-Crash "
                   "simulation result to compare against. The rest positions shown here come only from CrashLab.")
    else:
        for k, v in enumerate(vs):
            E = v["end"]
            if not E:
                continue
            ex, ey = P.g(E, "center_of_gravity_x"), P.g(E, "center_of_gravity_y")
            if ex is None:
                continue
            f = res["final"][k]
            d = float(np.hypot(f["x"] - ex, f["y"] - ey))
            out.append(f"{v['name']}: CrashLab rest position is {d:.1f} m from the PC-Crash end position.")
    for v in vs:
        I = v["input"]
        m0, mt = P.g(I, "mass_empty"), P.g(I, "total_weight")
        occ = (P.g(I, "mass_of_front_occupants", default=0) or 0) + (P.g(I, "mass_of_rear_occupants", default=0) or 0) \
            + (P.g(I, "mass_of_cargo", default=0) or 0) + (P.g(I, "mass_of_roof", default=0) or 0)
        if m0 and mt and abs(m0 + occ - mt) > 5:
            out.append(f"{v['name']}: total weight {mt:.0f} kg does not equal empty mass + occupants/cargo "
                       f"({m0 + occ:.0f} kg). The total weight was used.")
    out.append("Sequence reading: each PC-Crash sequence = one brake + one steering instruction, run in order; a sequence "
               "lasts its 'maximum stopping time', steering ramps to the new angle over the 'steering time', and the last "
               "instruction stays in force (also after impact).")
    out.append("Road geometry is not in the export: a straight carriageway along the moving vehicle's heading is drawn for context only.")
    return out


def input_table(parsed, scene):
    rows = []
    for k, v in enumerate(parsed["vehicles"]):
        I, S0 = v["input"], v["start"]
        c = scene.vehicles[k]
        rows.append(dict(label=c.label, name=v["name"], record=v.get("record"), cls=c.spec.style,
                         speed=_r(P.g(S0, "velocity_magnitude"), 1), heading=_r(P.g(S0, "heading_angle"), 2),
                         x=_r(P.g(S0, "center_of_gravity_x")), y=_r(P.g(S0, "center_of_gravity_y")),
                         mass=_r(c.spec.mass, 0), L=_r(c.spec.length), W=_r(c.spec.width), H=_r(c.spec.height),
                         a=_r(c.spec.a), h=_r(c.spec.h), abs=bool(c.abs_on),
                         seq=[dict(t0=p["t0"], t1=p["t1"], decel=p["decel"], steer=p["steer_to"]) for p in P.timeline(v)]))
    return rows


def report_payload(scene, res, interp, checks):
    ev = []
    for e in res["events"]:
        ev.append(dict(t=e["t"], vehicles=[e["a"], e["b"]], closing_kmh=e["closing_kmh"],
                       speed_kmh={e["a"]: e["speed_a_kmh"], e["b"]: e["speed_b_kmh"]},
                       delta_v_kmh={e["a"]: e["dv_a_kmh"], e["b"]: e["dv_b_kmh"]}, energy_kj=e["energy_kj"],
                       damage=[dict(vehicle=(e["a"] if k == 0 else e["b"]), face=d["face"], crush_cm=round(d["depth"] * 100),
                                    severity=d["severity"], cdc=d["cdc"], parts=[p["name"] for p in d["parts"]][:8])
                               for k, d in enumerate(e.get("damage", []))]))
    vs = []
    for k, v in enumerate(res["veh"]):
        sf = res["safety"][k]; f = res["final"][k]
        moved = float(np.hypot(f["x"] - v.cfg.x, f["y"] - v.cfg.y))
        dirs = ["+x (east on the diagram)", "north-east", "+y (north on the diagram)", "north-west", "-x (west on the diagram)",
                "south-west", "-y (south on the diagram)", "south-east"]
        hd = dirs[int(round((np.degrees(v.cfg.psi) % 360) / 45)) % 8]
        vs.append(dict(label=v.cfg.label, name=v.spec.name, cls=v.spec.style, start_speed_kmh=round(v.cfg.speed * 3.6, 1),
                       start_heading=f"{np.degrees(v.cfg.psi):.1f} deg, i.e. towards {hd}",
                       moved_to_rest_m=round(moved, 1), rest_heading_change_deg=round(float(np.degrees(
                           (f["psi"] - v.cfg.psi + np.pi) % (2 * np.pi) - np.pi)), 1),
                       airbags_fitted_assumed=list(v.spec.airbags),
                       airbags_deployed=[a["type"] + (" " + a["side"] if a.get("side") else "") for a in sf["airbags_deployed"]],
                       tipped_over=bool(sf["toppled"]), occupants=v.cfg.occupants))
    hit = {x for e in res["events"] for x in (e["a"], e["b"])}
    not_struck = [f"{v.cfg.label} ({v.spec.example or v.spec.name})" for v in res["veh"] if v.cfg.label not in hit]
    return dict(title=scene.title, input_story=scene.story, road_friction=scene.mu, not_struck=not_struck,
                input_timeline=interp.get("timeline", []), impacts=ev, vehicles=vs, data_checks=checks,
                note="No impact occurred in the simulated time." if not ev else "")


def run(text, notes="", use_ai=True, log=print):
    t0 = time.time()
    parsed = P.parse(text)
    if not parsed["vehicles"]:
        raise ValueError("No vehicles found. Paste the full PC-Crash values export (it starts with 'Vehicle:').")
    log("parsed", len(parsed["vehicles"]), "vehicles")
    model, err, interp = None, None, None
    if use_ai:
        try:
            interp, model = ai.interpret(parsed, notes)
        except Exception as e:  # noqa
            err = str(e)
    if not interp:
        interp = ai.fallback_interpret(parsed)
    scene, info = P.build_scene(parsed, interp, notes)
    log("simulating")
    res = run_scene(scene)
    checks = data_checks(parsed, res)
    rep = None
    if use_ai and not err:
        try:
            rep, _ = ai.report(report_payload(scene, res, interp, checks))
        except Exception as e:  # noqa
            err = str(e)
    pay = build.scene_payload(res)
    if rep and isinstance(rep.get("title"), str) and rep["title"].strip():
        pay["title"] = rep["title"].strip()[:90]
    pay["ai"] = dict(model=model, error=err, interpret=interp, report=rep, checks=checks,
                     inputs=input_table(parsed, scene), notes=notes, warnings=parsed["warnings"],
                     seconds=round(time.time() - t0, 1))
    base = json.load(open(os.path.join(ROOT, "dist", "data.json"), encoding="utf-8"))
    base["scenes"] = [pay]
    return base, pay


def save_case(base, cases_dir):
    os.makedirs(cases_dir, exist_ok=True)
    cid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    js = json.dumps(base, separators=(",", ":"))
    html = build.bundle_html(js)
    path = os.path.join(cases_dir, cid + ".html")
    open(path, "w", encoding="utf-8").write(html)
    return cid, path


if __name__ == "__main__":
    txt = open(sys.argv[1], encoding="latin-1").read()
    notes = sys.argv[2] if len(sys.argv) > 2 else ""
    base, pay = run(txt, notes)
    cid, path = save_case(base, os.path.join(ROOT, "cases"))
    print("case", cid, path)
    print(json.dumps(pay["ai"]["report"], indent=1)[:2500] if pay["ai"]["report"] else pay["ai"]["error"])
