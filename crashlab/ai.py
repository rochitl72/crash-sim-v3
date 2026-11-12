"""AI layer (Groq, OpenAI-compatible API). Reads GROQ_API_KEY from the environment or a local .env.

Two calls:
  interpret(): classify vehicles, read the PC-Crash sequences, write the scene description, list gaps.
  report():    write the reconstruction report from the *simulated* numbers.
The model never supplies physics inputs: speeds, positions, masses and brake/steer inputs come from the
parser. AI suggestions (vehicle class, airbags fitted, occupants/belts) are validated and flagged.
If the API is unreachable, deterministic fallbacks keep the pipeline working."""
import json, os, re, ssl, time, urllib.request, urllib.error


def _ssl_ctx():
    """Python from python.org on macOS ships without CA certificates; fall back to certifi or the system bundle."""
    ctx = ssl.create_default_context()
    if os.environ.get("SSL_CERT_FILE") or ctx.cert_store_stats().get("x509_ca", 0) > 0:
        return ctx
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        pass
    for f in ("/etc/ssl/cert.pem", "/etc/ssl/certs/ca-certificates.crt", "/opt/homebrew/etc/openssl@3/cert.pem"):
        if os.path.exists(f):
            try:
                ctx = ssl.create_default_context(); ctx.load_verify_locations(cafile=f); return ctx
            except Exception:
                pass
    return ssl.create_default_context()
import numpy as np

API = "https://api.groq.com/openai/v1/chat/completions"
MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
STYLES = ["hatch", "sedan", "csuv", "suv", "minitruck", "truck", "bus", "auto", "moto", "scooter", "cycle", "barricade"]


def _key():
    k = os.environ.get("GROQ_API_KEY")
    if k:
        return k.strip()
    for p in (os.path.join(os.path.dirname(__file__), "..", ".env"), ".env"):
        try:
            for line in open(p, encoding="utf-8"):
                if line.strip().startswith("GROQ_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        except OSError:
            pass
    return None


def _call(system, user, max_tokens=2500, temperature=0.2):
    key = _key()
    if not key:
        raise RuntimeError("GROQ_API_KEY not set (put it in .env)")
    last = None
    for model in MODELS:
        body = dict(model=model, temperature=temperature, max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        req = urllib.request.Request(API, data=json.dumps(body).encode(), method="POST",
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                                              "User-Agent": "crashlab-studio/1.0"})
        for attempt in range(2):
            try:
                with urllib.request.urlopen(req, timeout=60, context=_ssl_ctx()) as r:
                    data = json.loads(r.read())
                txt = data["choices"][0]["message"]["content"]
                m = re.search(r"\{.*\}", txt, re.S)
                return json.loads(m.group(0) if m else txt), model
            except urllib.error.HTTPError as e:
                last = f"{model}: HTTP {e.code} {e.read()[:300]!r}"
                if e.code in (401, 403):
                    raise RuntimeError("Groq rejected the API key (401/403). Rotate it and update .env.")
                time.sleep(1.0)
            except Exception as e:  # noqa
                last = f"{model}: {e}"
                time.sleep(0.6)
    raise RuntimeError(last or "AI call failed")


def _geometry(parsed):
    """Pre-computed relative geometry so the model does not have to do trigonometry."""
    vs = parsed["vehicles"]; out = []
    from .pccrash import g
    P = [(g(v["start"], "center_of_gravity_x", default=0), g(v["start"], "center_of_gravity_y", default=0),
          g(v["start"], "heading_angle", default=0), g(v["start"], "velocity_magnitude", default=0)) for v in vs]
    for i, (x, y, h, s) in enumerate(P):
        for j, (x2, y2, h2, s2) in enumerate(P):
            if i == j or s <= 0:
                continue
            hr = np.radians(h); dx, dy = x2 - x, y2 - y
            ahead = dx * np.cos(hr) + dy * np.sin(hr); left = -dx * np.sin(hr) + dy * np.cos(hr)
            out.append(f"From {vs[i]['name']}: {vs[j]['name']} is {abs(ahead):.1f} m {'AHEAD of' if ahead > 0 else 'BEHIND'} it and "
                       f"{abs(left):.1f} m to the {'left' if left > 0 else 'right'} of its heading line "
                       f"(heading difference {((h2 - h + 180) % 360) - 180:.1f} deg).")
    return out


def _compact(parsed):
    from .pccrash import g, timeline
    vs = []
    for i, v in enumerate(parsed["vehicles"]):
        I, S0 = v["input"], v["start"]
        vs.append(dict(index=i, name=v["name"], database=v.get("database"), record=v.get("record"),
                       start_speed_kmh=g(S0, "velocity_magnitude"), heading_deg=g(S0, "heading_angle"),
                       heading_words="towards " + ["+x (east on the diagram)", "north-east", "+y (north on the diagram)", "north-west",
                                                   "-x (west on the diagram)", "south-west", "-y (south on the diagram)",
                                                   "south-east"][int(round(((g(S0, "heading_angle") or 0) % 360) / 45)) % 8],
                       x=g(S0, "center_of_gravity_x"), y=g(S0, "center_of_gravity_y"),
                       length=g(I, "length_m"), width=g(I, "width_m"), height=g(I, "height_m"),
                       mass_total_kg=g(I, "total_weight"), mass_empty_kg=g(I, "mass_empty"),
                       front_occupant_mass_kg=g(I, "mass_of_front_occupants"),
                       rear_occupant_mass_kg=g(I, "mass_of_rear_occupants"), abs=I.get("abs"),
                       sequence=[dict(t=f"{p['t0']:.2f}-{p['t1']:.2f} s", brake_decel_m_s2=p["decel"],
                                      steer_from_deg=p["steer_from"], steer_to_deg=p["steer_to"],
                                      steer_direction=("straight" if abs(p["steer_to"]) < 0.05 else
                                                       "to the RIGHT (clockwise)" if p["steer_to"] < 0 else "to the LEFT (anticlockwise)"),
                                      steer_ramp_s=p["steer_time"]) for p in timeline(v)]))
    return vs


SYS_INTERPRET = """You are a road-crash reconstruction assistant for Indian traffic police. You receive a
PC-Crash values export already parsed into JSON, pre-computed geometry and optional officer notes.
Return ONE JSON object only, with keys:
 "vehicles": [{"index": int, "label": "A"|"B"|... (short, 1-2 chars, objects may use "X"),
   "style": one of hatch|sedan|csuv|suv|minitruck|truck|bus|auto|moto|scooter|cycle|barricade,
   "airbags": list from ["front","side","curtain","knee"] that this make/model/year is LIKELY to have in India (empty if none),
   "airbag_basis": short reason, mark as an assumption,
   "occupants": [{"seat": "driver"|"front passenger"|"rear left"|"rear right"|"rear centre"|"rider", "belt": bool, "helmet": bool, "assumed": bool}]
     (use occupant masses at ~80 kg per person unless the notes say otherwise; if belt use is unknown set belt true and assumed true; objects have []),
   "role": short phrase (e.g. "striking car", "parked truck", "roadside barricade")}],
 "scene": {"title": short headline, "subtitle": one line, "story": 3-5 plain sentences describing what the input describes happening, in order, with the numbers given,
   "tags": up to 4 short tags, "lanes": integer guess or null, "road_type": short guess, "lighting": "day"|"night"|"dusk"|"unknown", "weather": short or "unknown"},
 "overrides": {"mu": number or null (only if the officer notes give a surface/friction that differs from the file)},
 "timeline": [{"t": "time range", "who": vehicle name, "text": plain-language reading of that input phase}],
 "conflicts": [any contradiction between the officer notes and the file data, e.g. a side of the road that does not match the geometry; empty list if none],
 "missing": [things an investigator should still collect, e.g. road geometry, rest positions, skid lengths, damage photos, belt use, lighting],
 "confidence": "low"|"medium"|"high" for how well the file describes the crash.
Rules: never invent speeds, positions or times that are not in the data. Use the pre-computed geometry and the
steer_direction and heading_words exactly; do not recompute them (diagram directions, not real compass). The story must describe ONLY the set-up and the input phases
(who is where, moving how, braking/steering when). Do NOT describe collisions, impacts or what happens afterwards:
a physics simulation will determine that. Objects BEHIND a moving vehicle are not in its path. Use the officer's notes for occupants/belts/helmets/weather when given. Keep text concise and factual."""

SYS_REPORT = """You are a road-crash reconstruction assistant writing for traffic police. You receive the simulated
reconstruction of a crash (numbers from a physics engine) plus the input summary. Return ONE JSON object only:
 "title": short headline of what the SIMULATION shows (max 70 chars),
 "summary": 2-4 sentences, plain language, the main finding with key numbers,
 "sequence": [{"t": "time", "text": "what happens"}] from start to rest (4-8 items),
 "impact": 2-4 sentences: impact speed(s), closing speed, delta-V, energy, impact configuration,
 "damage": 2-4 sentences naming the damaged areas and crush depths given,
 "occupants": 2-4 sentences on airbags deployed, belts as given, and what the delta-V means for occupants (qualitative),
 "consistency": 1-3 sentences comparing with any end values / evidence in the input (say clearly if the export has none),
 "assumptions": list of the key assumptions behind this reconstruction,
 "next_steps": list of 3-6 concrete evidence items or checks that would firm it up.
Rules: the "impacts" list is complete: only those collisions happened. Anything in "not_struck" was NOT hit and
must not be described as hit or contacted. Use ONLY numbers present in the input JSON; do not invent injuries or blame; write 'the simulation indicates',
not certainties; no legal conclusions."""


def interpret(parsed, notes=""):
    user = json.dumps(dict(vehicles=_compact(parsed), geometry=_geometry(parsed), parser_warnings=parsed["warnings"],
                           officer_notes=notes or "(none)"), default=str)
    out, model = _call(SYS_INTERPRET, user)
    return _clean_interpret(out, parsed), model


def _clean_interpret(o, parsed):
    n = len(parsed["vehicles"]); vs = []
    for x in o.get("vehicles", []) if isinstance(o.get("vehicles"), list) else []:
        if not isinstance(x, dict):
            continue
        try:
            i = int(x.get("index"))
        except (TypeError, ValueError):
            continue
        if not 0 <= i < n:
            continue
        x["style"] = x.get("style") if x.get("style") in STYLES else None
        x["label"] = re.sub(r"[^A-Za-z0-9]", "", str(x.get("label") or ""))[:3] or None
        vs.append(x)
    o["vehicles"] = vs
    return o


def report(payload):
    out, model = _call(SYS_REPORT, json.dumps(payload, default=str), max_tokens=2500)
    return out, model


def fallback_interpret(parsed):
    from .pccrash import guess_style
    vs = []
    for i, v in enumerate(parsed["vehicles"]):
        vs.append(dict(index=i, label=chr(65 + i), style=guess_style(v), role="", occupants=None))
    names = " vs ".join(v["name"] for v in parsed["vehicles"][:2])
    return dict(vehicles=vs, scene=dict(title=f"Reconstruction: {names}", subtitle="Imported from PC-Crash values export",
                                        story="AI description unavailable; scene built directly from the export.",
                                        tags=["PC-Crash import"]),
                timeline=[], missing=["AI service unavailable: description not generated"], confidence="unknown")
