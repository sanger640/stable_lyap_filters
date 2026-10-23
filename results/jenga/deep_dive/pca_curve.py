import os,sys,json,numpy as np
os.environ.setdefault("MUJOCO_GL","egl"); os.environ["HARD_MASS"]="0"
sys.path[:0]=['eval','src']
import torch, bench2 as b2, cem_filter as cf, inspector_data as insp, cbf_barrier as cb
from jenga_vjepa_encode import dino_tokens
C,R=b2.load('dev'); mon=cf.Monitor(); sim=insp._sim(); L=mon.latent
D=[]
for i in range(150):
    sim.restore((C['state'][i],C['ctrl'][i],C['target'][i],float(C['gripper'][i]))); site=sim.data.site_xpos[sim.site].copy(); f0=mon.frame(sim)
    for a in C['nominal'][i]: sim.execute(a)
    for _ in range(30): sim.execute(C['nominal'][i][-1])
    f1=mon.frame(sim); t=dino_tokens(L.encoder,np.stack([f0,f1]),'cuda').double()
    keep=torch.as_tensor(~cb.hand_mask(np.vstack([site[None],C['nominal'][i][:,:3]])),device='cuda').reshape(-1)
    D.append((t[1]-t[0])[keep].cpu())
D=torch.cat(D); tr=D[: len(D)*2//3]; te=D[len(D)*2//3:]
U,S,V=torch.linalg.svd(tr - 0, full_matrices=False)
Bapp=L.basis.double().cpu()
out={'patches':len(D),'appearance_pca8_share':float(((te@Bapp)**2).sum()/(te**2).sum())}
for k in (8,16,32,64,128,256):
    P=V[:k].T; out[f'change_pca{k}_share']=float(((te@P)**2).sum()/(te**2).sum())
print(out); json.dump(out,open('results/jenga/deep_dive/pca_curve.json','w'),indent=1)
