"""Three terminal states, not two: fallen-left, upright, fallen-right.

Every clustering attempt this session used k=2 and centroid distance. Both are wrong here: the
block topples LEFT or RIGHT, so 'toppled' is two modes at -pi/2 and +pi/2 whose centroid is
(0,0) -- exactly the upright state. Centroid separation is therefore ~0 by construction, and
2-means splits left-vs-right rather than toppled-vs-safe. Basin entropy needs the right number
of terminal states."""
import sys; sys.path[:0]=['/home/sanger/wksp/stable_lyap_filters/src/systems',
                          '/home/sanger/wksp/stable_lyap_filters/src/models',
                          '/home/sanger/wksp/stable_lyap_filters/eval']
import numpy as np, torch, tipping_block as tb
from shplrnn import ShPLRNN
from run_phase3_monitor import (ACT_DIM,OBS_DIM,T_ON,T_OFF,make_data,margin_of,probes,rollout)
from run_phase3_block import auc

n=450; thr=tb.topple_threshold(n,T_ON,T_OFF)
S_,A_,mu,sd,amu,asd=make_data(40,thr,n,seed=0)
model=ShPLRNN(d=4,H=128,action_dim=ACT_DIM,obs_dim=OBS_DIM).double()
model.load_state_dict(torch.load('results/phase3/monitor_model_T450.pt'))
s0=((torch.zeros(1,OBS_DIM).double()-mu)/sd); rng=np.random.default_rng(11)

def kmeans(X,k,iters=150,seed=0):
    r=np.random.default_rng(seed); C=X[r.choice(len(X),k,replace=False)].copy()
    for _ in range(iters):
        lab=np.argmin(((X[:,None]-C[None])**2).sum(-1),axis=1)
        for j in range(k):
            if (lab==j).any(): C[j]=X[lab==j].mean(0)
    return C,lab
def ent(labels,k):
    p=np.array([(labels==j).mean() for j in range(k)]); p=p[p>0]
    return float(-(p*np.log(p)).sum())
def latents(A):
    pn=((torch.from_numpy(A[:,:,None]).double()-amu)/asd)
    return rollout(model,pn,s0.expand(len(A),OBS_DIM),n)[:,-1].numpy()

M=800
cal=np.stack([tb.random_push(rng,n,thr) for _ in range(M)])
Et=np.stack([tb.simulate(a)[0][-1] for a in cal])          # TRUE terminal states
Em=latents(cal)                                            # MODEL terminal latents
th=Et[:,0]
truth3=np.where(th<-1.0,0,np.where(th>1.0,2,1))            # left / upright / right
print("true terminal-state counts (left/upright/right):",
      [int((truth3==j).sum()) for j in range(3)])
for nm,X in (("TRUE state",Et),("MODEL latent",Em)):
    C,lab=kmeans(X.copy(),3)
    # agreement of the discovered 3 clusters with the true 3 terminal states
    best=0
    import itertools
    for perm in itertools.permutations(range(3)):
        m=np.array([perm[l] for l in lab]); best=max(best,(m==truth3).mean())
    print(f"  {nm:<13} k=3 clusters match the true terminal states {100*best:>5.1f}% of the time")

C3,_=kmeans(Em.copy(),3)                                   # runtime classifier: model, k=3
N,NP,eps=250,48,0.20
acts=np.stack([tb.random_push(rng,n,thr) for _ in range(N)])
marg=np.array([margin_of(a) for a in acts]); near=marg<0.10
Smod=[];Sora=[]
for a in acts:
    P=probes("shared-action",a,thr,eps,NP,rng)
    lab=np.argmin(((latents(P)[:,None]-C3[None])**2).sum(-1),axis=1)
    Smod.append(ent(lab,3))
    t=np.stack([tb.simulate(p)[0][-1] for p in P])[:,0]
    Sora.append(ent(np.where(t<-1.0,0,np.where(t>1.0,2,1)),3))
Smod=np.array(Smod);Sora=np.array(Sora)
print(f"\n{N} actions, {int(near.sum())} near")
print(f"  MODEL basin entropy (k=3 latent clusters)  AUC {auc(Smod,near):.3f}")
print(f"  ORACLE basin entropy (3 true states)       AUC {auc(Sora,near):.3f}")
for nm,S in (("MODEL",Smod),("ORACLE",Sora)):
    pred=S>1e-9
    tp=int((pred&near).sum()); fp=int((pred&~near).sum()); fn=int((~pred&near).sum())
    pr=tp/(tp+fp) if tp+fp else 0.0; rc=tp/(tp+fn) if tp+fn else 0.0
    print(f"    {nm} S>0: flags {100*pred.mean():>3.0f}%  P {pr:.3f} R {rc:.3f} "
          f"F1 {2*pr*rc/(pr+rc) if pr+rc else 0:.3f}")
print("\nDONE")
