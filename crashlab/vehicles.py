"""Vehicle catalogue for CrashLab.

All numbers are *typical class values* (rounded, from public spec sheets and
textbook ranges), NOT measured data for a specific make/model. The example
models in brackets ("XUV700-class") only indicate the size class. For a real
case replace them with the actual vehicle's mass, dimensions and CG location.

Crush stiffness (A, B) follows Campbell's linear crush model: force per unit
width F/w = A + B*C at crush depth C. Values are rounded literature-typical
magnitudes per class (front / side / rear), not certified test data.
"""
import numpy as np

G = 9.81


class Spec:
    """Rigid-body vehicle description. Body frame: x forward, y left, origin at CG."""

    def __init__(self, name, kind, mass, izz, length, width, a, b, h, wheels,
                 c_alpha=9.0, cda=0.7, color="#888888", slide_on_hit=False,
                 max_steer=0.55, drive_max_g=0.30, crr=0.015, slide_mu=0.5,
                 height=1.5, style="sedan", example="", crush=None, airbags=(), bumper=0.5,
                 max_crush=(0.9, 0.55, 0.8), seats=None):
        self.name, self.kind = name, kind
        self.height, self.style, self.example = float(height), style, example
        self.crush = crush or dict(front=(40e3, 450e3), side=(15e3, 150e3), rear=(30e3, 320e3))
        self.airbags = tuple(airbags)          # fitted: "front", "side", "curtain", "knee"
        self.bumper = float(bumper)            # height of the main load path above ground [m]
        self.max_crush = max_crush             # front, side, rear [m]
        self.seats = seats
        self.mass, self.izz = float(mass), float(izz)
        self.length, self.width = float(length), float(width)
        self.a, self.b, self.h = float(a), float(b), float(h)
        self.wheels = np.asarray(wheels, float)   # x, y, steer, drive, brake, fz0
        self.c_alpha, self.cda, self.color = c_alpha, cda, color
        self.slide_on_hit, self.max_steer = slide_on_hit, max_steer
        self.drive_max_g, self.crr, self.slide_mu = drive_max_g, crr, slide_mu
        w = self.wheels
        front = w[:, 0] > 0
        self.klong = np.where(front, -1.0 / max(front.sum(), 1), 1.0 / max((~front).sum(), 1))
        yy = w[:, 1]
        self.klat = np.where(np.abs(yy) > 1e-6,
                             -np.sign(yy) * 0.5 / (2 * np.maximum(np.abs(yy), 1e-6)), 0.0)
        # mild understeer bias (real cars are understeer-balanced; equal normalised
        # stiffness would be exactly neutral-steer). Not applied to bikes / 3-wheelers.
        us = 0.12 if (front.sum() == 2 and (~front).sum() == 2) else 0.0
        self.ca_mult = np.where(front, 1.0 - us, 1.0 + us)
        self.rect_off = (self.a - self.b) / 2.0          # geometric centre ahead of CG
        self.radius = 0.5 * np.hypot(self.length, self.width)
        self.wheelbase = self.a + self.b
        # --- roll / tip-over geometry. Tipping axis: line through the outer wheel contacts.
        # 4+ wheels: half-track. Delta tricycle (1 front, 2 rear): the axis runs from the front
        # wheel to a rear wheel, so the CG is much closer to it (why autos tip easily).
        fr = w[w[:, 0] > 0]; rr_ = w[w[:, 0] <= 0]
        tf = np.abs(fr[:, 1]).max() if len(fr) else 0.0
        tr = np.abs(rr_[:, 1]).max() if len(rr_) else 0.0
        if tf < 1e-6 and tr > 1e-6:
            self.b_tip = tr * self.a / np.hypot(self.wheelbase, tr)
        else:
            self.b_tip = min(tf, tr) if min(tf, tr) > 0 else 0.0
        self.ix = self.mass * (self.width ** 2 + self.height ** 2) / 12.0   # roll inertia about CG
        self.ssf = self.b_tip / self.h if self.b_tip > 0 else 0.0            # static stability factor

    def meta(self):
        return dict(name=self.name, kind=self.kind, mass=self.mass, length=self.length,
                    width=self.width, rect_off=self.rect_off, color=self.color,
                    height=self.height, style=self.style, example=self.example, cg_h=self.h,
                    a=self.a, b=self.b, airbags=list(self.airbags), bumper=self.bumper,
                    ssf=round(self.ssf, 2),
                    wheels=[[float(x), float(y), int(s)] for x, y, s, *_ in self.wheels])


def _make(name, kind, mass, izz, L, W, a, b, h, nf, nr, tf, tr, drive, brake_front,
          c_alpha, cda, color, **kw):
    ff, fr = b / (a + b), a / (a + b)
    wl = []
    for i in range(nf):
        y = 0.0 if nf == 1 else (tf / 2 if i == 0 else -tf / 2)
        wl.append([a, y, 1.0, (1.0 / nf if drive == "front" else 0.0), brake_front / nf, ff / nf])
    for i in range(nr):
        y = 0.0 if nr == 1 else (tr / 2 if i == 0 else -tr / 2)
        wl.append([-b, y, 0.0, (1.0 / nr if drive == "rear" else 0.0), (1 - brake_front) / nr, fr / nr])
    return Spec(name, kind, mass, izz, L, W, a, b, h, wl, c_alpha, cda, color, **kw)


CAR_SIDE = (15e3, 150e3)


def hatchback(color="#ffb703", airbags=("front",)):
    return _make("Hatchback", "car", 1000, 1350, 3.85, 1.70, 1.05, 1.40, 0.55, 2, 2, 1.45, 1.45,
                 "front", 0.70, 9.0, 0.65, color, height=1.52, style="hatch", example="Swift / i20 class",
                 crush=dict(front=(35e3, 400e3), side=(12e3, 140e3), rear=(28e3, 300e3)),
                 airbags=airbags, bumper=0.48, max_crush=(0.85, 0.5, 0.7))


def sedan(color="#2ec4b6", airbags=("front", "side", "curtain")):
    return _make("Sedan", "car", 1250, 1900, 4.50, 1.80, 1.25, 1.40, 0.55, 2, 2, 1.55, 1.55,
                 "front", 0.70, 9.5, 0.70, color, height=1.49, style="sedan", example="City / Verna class",
                 crush=dict(front=(40e3, 450e3), side=(15e3, 150e3), rear=(30e3, 320e3)),
                 airbags=airbags, bumper=0.48, max_crush=(0.95, 0.55, 0.85))


def compact_suv(color="#9b5de5", airbags=("front", "side", "curtain")):
    return _make("Compact SUV", "car", 1400, 2200, 4.30, 1.79, 1.15, 1.45, 0.62, 2, 2, 1.55, 1.55,
                 "front", 0.70, 9.0, 0.85, color, height=1.64, style="csuv", example="Creta / Nexon class",
                 crush=dict(front=(45e3, 550e3), side=(17e3, 170e3), rear=(34e3, 380e3)),
                 airbags=airbags, bumper=0.55, max_crush=(0.9, 0.55, 0.8))


def suv(color="#8ecae6", airbags=("front", "side", "curtain", "knee")):
    return _make("SUV", "car", 1900, 3300, 4.70, 1.85, 1.30, 1.45, 0.72, 2, 2, 1.55, 1.55,
                 "rear", 0.68, 8.5, 1.00, color, height=1.76, style="suv", example="XUV700 / Scorpio class",
                 crush=dict(front=(50e3, 650e3), side=(18e3, 190e3), rear=(38e3, 420e3)),
                 airbags=airbags, bumper=0.58, max_crush=(0.95, 0.55, 0.85))


xuv = suv


def mini_truck(color="#6c9a8b", airbags=()):
    return _make("Mini truck (laden)", "truck", 1500, 2600, 3.80, 1.50, 1.05, 1.05, 0.78, 2, 2,
                 1.30, 1.30, "rear", 0.65, 8.0, 1.3, color, height=1.84, style="minitruck",
                 example="Ace / Super Carry class",
                 crush=dict(front=(30e3, 380e3), side=(20e3, 250e3), rear=(60e3, 900e3)),
                 airbags=airbags, bumper=0.55, max_crush=(0.7, 0.45, 0.5))


def truck(color="#e76f51", airbags=()):
    return _make("Goods truck (laden)", "truck", 9000, 52000, 8.00, 2.50, 3.00, 1.80, 1.10, 2, 2,
                 2.00, 1.85, "rear", 0.55, 7.0, 5.00, color, height=3.2, style="truck",
                 example="16 t rigid truck class",
                 crush=dict(front=(120e3, 2.5e6), side=(100e3, 2.0e6), rear=(150e3, 3.0e6)),
                 airbags=airbags, bumper=0.75, max_crush=(0.5, 0.35, 0.4))


def bus(color="#f4a261", airbags=()):
    return _make("Bus", "bus", 11000, 130000, 11.00, 2.55, 3.30, 2.70, 1.15, 2, 2, 2.05, 1.85,
                 "rear", 0.55, 7.0, 5.40, color, height=3.1, style="bus", example="12 m city bus class",
                 crush=dict(front=(90e3, 1.8e6), side=(60e3, 1.2e6), rear=(100e3, 2.0e6)),
                 airbags=airbags, bumper=0.7, max_crush=(0.6, 0.4, 0.5))


def auto_rickshaw(color="#e9c46a", airbags=()):
    return _make("Auto-rickshaw", "three-wheeler", 520, 330, 2.65, 1.30, 1.15, 0.85, 0.60, 1, 2,
                 0.0, 1.00, "rear", 0.30, 7.0, 0.90, color, max_steer=0.6, height=1.70, style="auto",
                 example="3-seat passenger auto class",
                 crush=dict(front=(12e3, 120e3), side=(6e3, 60e3), rear=(15e3, 150e3)),
                 airbags=airbags, bumper=0.45, max_crush=(0.6, 0.45, 0.5), slide_mu=0.45)


def motorcycle(color="#b5179e"):
    return _make("Motorcycle + rider", "two-wheeler", 260, 55, 2.05, 0.75, 0.70, 0.65, 0.65, 1, 1,
                 0.0, 0.0, "rear", 0.60, 7.0, 0.50, color, slide_on_hit=True, max_steer=0.5,
                 slide_mu=0.45, height=1.25, style="moto", example="150-200 cc commuter class",
                 crush=dict(front=(8e3, 150e3), side=(5e3, 80e3), rear=(5e3, 80e3)), bumper=0.45,
                 max_crush=(0.4, 0.25, 0.3))


def scooter(color="#f15bb5"):
    return _make("Scooter + rider", "two-wheeler", 185, 40, 1.83, 0.70, 0.62, 0.62, 0.60, 1, 1,
                 0.0, 0.0, "rear", 0.60, 7.0, 0.50, color, slide_on_hit=True, max_steer=0.5,
                 slide_mu=0.45, height=1.20, style="scooter", example="110-125 cc scooter class",
                 crush=dict(front=(6e3, 110e3), side=(4e3, 70e3), rear=(4e3, 70e3)), bumper=0.42,
                 max_crush=(0.35, 0.25, 0.25))


def bicycle(color="#80ed99"):
    return _make("Bicycle + rider", "cycle", 88, 16, 1.75, 0.60, 0.55, 0.50, 0.95, 1, 1,
                 0.0, 0.0, "rear", 0.55, 6.0, 0.45, color, slide_on_hit=True, max_steer=0.5,
                 slide_mu=0.55, drive_max_g=0.08, height=1.70, style="cycle", example="standard roadster",
                 crush=dict(front=(2e3, 40e3), side=(1.5e3, 30e3), rear=(1.5e3, 30e3)), bumper=0.5,
                 max_crush=(0.3, 0.2, 0.2))


CATALOGUE = [hatchback, sedan, compact_suv, suv, mini_truck, truck, bus, auto_rickshaw, motorcycle,
             scooter, bicycle]
