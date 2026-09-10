"""Does the perturbation DIRECTION matter? (It does, and more than eps does.)

The deviator agent perturbs actions with isotropic Gaussian noise. In a T-dimensional action
space that is a poor probe: a random direction is nearly ORTHOGONAL to whichever direction
actually causes failure, so most of the perturbation is spent on harmless dimensions. Measured
here: searching for the failure boundary along random directions finds it at a median radius of
0.500 (the search cap, i.e. never), while simply SCALING the action finds it at 0.391.

Jenga's action is 8x4 = 32-dimensional, so an isotropic probe puts only about 1/sqrt(32) ~ 18%
of its magnitude along any given critical direction.

Compares three probe families at equal budget (32 perturbations):
    isotropic   what the monitor does now: independent noise per timestep
    scaled      perturb the action's MAGNITUDE, a single shared factor
    smooth      low-frequency: a few sine components rather than white noise
"""
import sys; sys.path[:0]=['src/systems','eval','src/models','src/training']
import numpy as np, torch
import tipping_block as tb
from shplrnn import ShPLRNN
from run_phase3_block import ACT_DIM, N_STEPS, OBS_DIM, auc, make_data, metrics
from phase3_universal import latent_rollout
from phase3_margin import action_margin, spearman

thr = tb.topple_threshold(N_STEPS,10,60)
S,A,mu,sd,amu,asd = make_data(600, thr, seed=0)
model = ShPLRNN(d=4,H=128,action_dim=ACT_DIM,obs_dim=OBS_DIM).double()
model.load_state_dict(torch.load('results/phase3/universal_model.pt'))

rng=np.random.default_rng(1)
N,T,NP=400,100,32
acts=np.stack([tb.random_push(rng,N_STEPS,thr) for _ in range(N)])
margin=np.array([action_margin(a) for a in acts])
outcome=np.array([bool(tb.simulate(a)[1][-1]) for a in acts])
s0=((torch.zeros(1,OBS_DIM).double()-mu)/sd)

def scores(kind, eps):
    div=[]
    for a in acts:
        base=a[:T]
        if kind=='isotropic':                       # what the monitor does now
            pert=base[None]+thr*eps*rng.standard_normal((NP,T))
        elif kind=='scaled':                        # perturb the action's MAGNITUDE
            pert=base[None]*(1.0+eps*rng.standard_normal((NP,1)))
        elif kind=='smooth':                        # low-frequency: one bump, not white noise
            k=rng.standard_normal((NP,5))
            basis=np.stack([np.sin((j+1)*np.pi*np.arange(T)/T) for j in range(5)])
            pert=base[None]+thr*eps*(k@basis)/np.sqrt(5)
        allact=np.concatenate([base[None],pert])
        actn=((torch.from_numpy(allact[...,None]).double()-amu)/asd)
        Z=latent_rollout(model,actn,s0.expand(len(allact),OBS_DIM),T)
        div.append(float((Z[1:,-1]-Z[0,-1]).norm(dim=-1).mean()))
    return np.array(div)

for near in (0.10, 0.15):
    lab = margin < near
    print(f"\n=== proximity: margin < {near:.0%}   ({int(lab.sum())}/{N} near) ===")
    print(f"{'perturbation':<14}{'eps':>7}{'AUC prox':>10}{'AUC outcome':>13}{'rho vs margin':>15}")
    for kind in ('isotropic','scaled','smooth'):
        for eps in (0.05,0.10,0.20):
            d=scores(kind,eps)
            print(f"{kind:<14}{eps:>7.2f}{auc(d,lab):>10.3f}{auc(d,outcome):>13.3f}"
                  f"{spearman(d,margin):>15.3f}",flush=True)
print("\nDONE")
