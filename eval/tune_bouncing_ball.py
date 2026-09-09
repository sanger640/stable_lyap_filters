"""
Pick a usable Phase-2 regime. Two failure modes bound the parameter space, both hit empirically.

  * e too LOW -> INELASTIC COLLAPSE (Zeno). At e<=0.5 dissipation outruns the table's energy
    injection: the ball settles onto the table, impacts stop being discrete events, the impact
    detector reads ZERO, x_max goes negative (riding the oscillating table), and lambda_max
    comes back as ~190. `step()` now records this rather than silently truncating.

  * Gamma too HIGH -> bounce height scales as ~Gamma*A, so at Gamma=4, A=1 the ball reaches
    x~47 and impacts occupy 0.27% of samples. The guard -- the entire object of Phase 2 --
    would be almost never sampled.

The lever for impact DENSITY in the discrete map is `dt`, not `e`. At dt=0.02 a flight spans
~377 steps, but that is ~2.4 table periods, which is physically fine and merely finely
sampled. Coarsening dt gives the model frequent impacts without leaving the chaotic regime.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "systems"))
import bouncing_ball as bb  # noqa: E402


def probe(omega, e, dt, n=6000, lam_steps=10000):
    bb.step.collapsed = False
    tr, imp = bb.simulate(n, dt=dt, omega=omega, e=e, return_impacts=True)
    t = tr.numpy()
    ni = int(imp.sum())
    spi = n / max(ni, 1)
    lam = bb.lambda_max_two_particle(n_steps=lam_steps, dt=dt, omega=omega, e=e)
    collapsed = getattr(bb.step, "collapsed", False)
    usable = (not collapsed) and lam > 0.02 and 6 <= spi <= 60 and t[:, 0].max() < 15
    return dict(omega=omega, e=e, dt=dt, x_med=float(np.median(t[:, 0])),
                x_max=float(t[:, 0].max()), steps_per_impact=spi, lam=lam,
                collapsed=collapsed, usable=usable)


if __name__ == "__main__":
    print(f'{"omega":>6}{"Gam":>6}{"e":>5}{"dt":>6}{"x_med":>8}{"x_max":>8}'
          f'{"stp/imp":>9}{"lam_max":>9}  flags', flush=True)
    best = []
    for omega in (1.2, 1.4, 2.0):
        for dt in (0.05, 0.10, 0.20):
            r = probe(omega, 0.8, dt)
            flags = ("COLLAPSED " if r["collapsed"] else "") + ("USABLE" if r["usable"] else "")
            print(f'{omega:>6.1f}{omega**2:>6.1f}{0.8:>5.1f}{dt:>6.2f}'
                  f'{r["x_med"]:>8.2f}{r["x_max"]:>8.2f}{r["steps_per_impact"]:>9.0f}'
                  f'{r["lam"]:>9.4f}  {flags}', flush=True)
            if r["usable"]:
                best.append(r)
    print()
    if best:
        b = max(best, key=lambda r: r["lam"])
        print(f'CHOSEN: omega={b["omega"]} e={b["e"]} dt={b["dt"]}  '
              f'lambda_max={b["lam"]:.4f}  ~{b["steps_per_impact"]:.0f} steps/impact', flush=True)
    else:
        print('NO USABLE REGIME in this grid -- widen it before training anything.', flush=True)
