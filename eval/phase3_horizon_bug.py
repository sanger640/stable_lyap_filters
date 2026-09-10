"""Same test with a horizon long enough for the fall to COMPLETE (commit ~133 + fall 164)."""
import sys; sys.path[:0]=['/home/sanger/wksp/stable_lyap_filters/src/systems',
                          '/home/sanger/wksp/stable_lyap_filters/eval']
import numpy as np, tipping_block as tb
from run_phase3_block import auc
from phase3_margin import spearman

NS=450
def margin_ns(act, s_max=0.6, coarse=0.02):
    base=bool(tb.simulate(act)[1][-1]); n=int(s_max/coarse)
    for i in range(1,n+1):
        for sign in (+1,-1):
            s=1.0+sign*i*coarse
            if s<=0: continue
            if bool(tb.simulate(act*s)[1][-1])!=base:
                lo,hi=1.0+sign*(i-1)*coarse,s
                for _ in range(20):
                    mid=0.5*(lo+hi)
                    if bool(tb.simulate(act*mid)[1][-1])!=base: hi=mid
                    else: lo=mid
                return abs(0.5*(lo+hi)-1.0)
    return s_max

def bc(x):
    x=np.asarray(x,float); n=len(x); s=x.std()
    if s<1e-12: return 0.0
    z=(x-x.mean())/s; g=float((z**3).mean()); k=float((z**4).mean()-3.0)
    den=k+3.0*(n-1)**2/((n-2)*(n-3))
    return float((g*g+1.0)/den) if den>1e-12 else 0.0

thr=tb.topple_threshold(NS,10,60); rng=np.random.default_rng(1)
N,NP=250,48
acts=np.stack([tb.random_push(rng,NS,thr) for _ in range(N)])
margin=np.array([margin_ns(a) for a in acts]); near=margin<0.10
print(f"horizon {NS} steps, {N} actions, {int(near.sum())} near\n")
# sanity: do toppled blocks now reach the absorbing state?
a=acts[int(np.argmin(margin))]
pert=a[None]*(1.0+0.10*rng.standard_normal((NP,1)))
d=np.array([np.linalg.norm(tb.simulate(p)[0][-1]) for p in pert])
f=np.array([bool(tb.simulate(p)[1][-1]) for p in pert])
print(f"NEAR action: flip {f.mean():.2f} | toppled |disp| {np.round(d[f][:4],3)} "
      f"| survivor |disp| {np.round(d[~f][:4],3)}")
print(f"  separation: toppled min {d[f].min():.3f} vs survivor max {d[~f].max():.3f}\n")
print(f"{'probe':<11}{'eps':>6}{'BC':>8}{'std':>8}{'spread':>9}{'IDEAL':>8}{'rho BC':>9}")
for kind in ('isotropic','scaled'):
    for eps in (0.05,0.10,0.20):
        B=[];S=[];V=[];F=[]
        for a in acts:
            if kind=='isotropic': pert=a[None]+thr*eps*rng.standard_normal((NP,NS))
            else:                 pert=a[None]*(1.0+eps*rng.standard_normal((NP,1)))
            ends=np.empty((NP,2)); fell=np.zeros(NP,bool)
            for j in range(NP):
                st,ff=tb.simulate(pert[j]); ends[j]=st[-1]; fell[j]=ff[-1]
            dd=np.linalg.norm(ends,axis=1)
            B.append(bc(dd)); S.append(float(dd.std()))
            V.append(float(np.linalg.norm(ends-ends.mean(0),axis=1).mean())); F.append(float(fell.mean()))
        B=np.array(B);S=np.array(S);V=np.array(V)
        ideal=1.0-np.abs(np.array(F)-0.5)*2
        print(f"{kind:<11}{eps:>6.2f}{auc(B,near):>8.3f}{auc(S,near):>8.3f}{auc(V,near):>9.3f}"
              f"{auc(ideal,near):>8.3f}{spearman(B,margin):>9.3f}",flush=True)
print("\nDONE")
