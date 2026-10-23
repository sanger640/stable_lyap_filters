import json,sys,numpy as np
def conc(rows,key,sign):
    c=d=t=0
    for i in {r['i'] for r in rows}:
        rs=[r for r in rows if r['i']==i]
        for a in range(len(rs)):
            for b in range(a+1,len(rs)):
                dr=rs[a]['risk']-rs[b]['risk']
                if abs(dr)<2/32-1e-9: continue
                ds=sign*(rs[a][key]-rs[b][key])
                if ds*dr>0: c+=1
                elif ds*dr<0: d+=1
                else: t+=1
    return {'concordant':c,'discordant':d,'tied':t,'conc':round(c/max(c+d,1),3)}
if __name__=='__main__':
    R=[json.loads(l) for l in open(sys.argv[1])]
    print('chunks',len(R),'decisions',len({r['i'] for r in R}),'risky chunks',sum(r['risk']>=2/32 for r in R))
    for k,s in [(k,(-1 if k.startswith('h') else 1)) for k in R[0] if k[:2] in ('h_','o_')]: print(k,conc(R,k,s))
