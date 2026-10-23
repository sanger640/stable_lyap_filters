import os,sys,json,numpy as np
os.environ.setdefault("MUJOCO_GL","egl"); os.environ["HARD_MASS"]="0"
sys.path[:0]=['eval','src']
import torch, bench2 as b2, cem_filter as cf, inspector_data as insp
from jenga_vjepa_encode import dino_tokens
C,R=b2.load('dev'); mon=cf.Monitor(); sim=insp._sim(); L=mon.latent
B=L.basis; print('basis orthonormal err', float((B.T@B-torch.eye(8,device=B.device)).abs().max()))
tot=[];cap=[];dtot=[];dcap=[]
for i in range(0,150,2):
    sim.restore((C['state'][i],C['ctrl'][i],C['target'][i],float(C['gripper'][i]))); f0=mon.frame(sim)
    for a in C['nominal'][i]: sim.execute(a)
    for _ in range(30): sim.execute(C['nominal'][i][-1])
    f1=mon.frame(sim)
    t=dino_tokens(L.encoder,np.stack([f0,f1]),'cuda').double(); Bd=B.double()
    x=t[0]-L.mean.double(); tot.append(float((x**2).sum())); cap.append(float(((x@Bd)**2).sum()))
    d=t[1]-t[0]; dtot.append(float((d**2).sum())); dcap.append(float(((d@Bd)**2).sum()))
    # change in patches that actually changed (top 5% patches by change norm)
    n=d.norm(dim=1); top=n>=n.quantile(0.95); dtt=float((d[top]**2).sum()); dct=float(((d[top]@Bd)**2).sum())
    tot.append(0) or None
    dtot[-1]=(dtot[-1],dtt); dcap[-1]=(dcap[-1],dct)
out={'frames':len(cap),'share_of_token_energy_in_8_pcs':sum(cap)/sum(t for t in tot if t),
     'share_of_change_energy_in_8_pcs':sum(c for c,_ in dcap)/sum(t for t,_ in dtot),
     'share_of_change_energy_in_8_pcs_top5pct_patches':sum(c for _,c in dcap)/sum(t for _,t in dtot)}
print(out); json.dump(out,open('results/jenga/deep_dive/pca_check.json','w'),indent=1)
