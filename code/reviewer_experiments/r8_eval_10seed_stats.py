# Fast vectorized exhaustive B2T EER for DOMCS(10) & DANN(10) + paired t / Wilcoxon / d / CI.
import os, json, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.cluster import KMeans; from sklearn.metrics import roc_curve; from scipy import stats
ap=argparse.ArgumentParser()
ap.add_argument("--data",required=True); ap.add_argument("--domcs",required=True); ap.add_argument("--dann",required=True); ap.add_argument("--out",required=True)
a=ap.parse_args(); os.makedirs(a.out,exist_ok=True)
DEV=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); K=3
class CB(nn.Module):
    def __init__(s,ic,oc,k):
        super().__init__(); s.net=nn.Sequential(nn.Conv1d(ic,oc,k,padding=k//2,bias=False),nn.BatchNorm1d(oc),nn.ELU(inplace=True))
    def forward(s,x): return s.net(x)
def enc(): return nn.Sequential(CB(64,64,7),CB(64,128,5),CB(128,256,3),nn.AdaptiveAvgPool1d(1))
class DOMCS(nn.Module):
    def __init__(s):
        super().__init__(); s.enc=enc(); s.id_fc=nn.Linear(256,128,bias=False); s.id_n=nn.LayerNorm(128)
        s.st_fc=nn.Linear(256,128,bias=False); s.st_n=nn.LayerNorm(128)
    def zid(s,x): f=s.enc(x).squeeze(-1); return F.normalize(s.id_n(s.id_fc(f)),dim=-1)
class DANN(nn.Module):
    def __init__(s):
        super().__init__(); s.enc=enc(); s.id_fc=nn.Linear(256,128,bias=False); s.id_n=nn.LayerNorm(128)
        s.dom=nn.Sequential(nn.Linear(256,128),nn.ReLU(inplace=True),nn.Dropout(0.3),nn.Linear(128,2))
    def zid(s,x): f=s.enc(x).squeeze(-1); return F.normalize(s.id_n(s.id_fc(f)),dim=-1)
def load(mk,p): m=mk(); ck=torch.load(p,map_location="cpu"); m.load_state_dict(ck.get("model_state",ck),strict=False); return m.to(DEV).eval()
@torch.no_grad()
def zb(m,Xa,bs=8192):
    Z=[]
    for i in range(0,len(Xa),bs): Z.append(m.zid(torch.tensor(Xa[i:i+bs],dtype=torch.float32,device=DEV)))
    return torch.cat(Z,0)
@torch.no_grad()
def eer_b2t(m,X,y,sess):
    rest=np.isin(sess,[1,2]); task=sess>=3; Zr=zb(m,X[rest]).cpu().numpy(); yr=y[rest]; subs=np.unique(y); Pl=[]
    for u in subs:
        zu=Zr[yr==u]; c=KMeans(K,n_init=5,random_state=0).fit(zu).cluster_centers_ if len(zu)>=K else np.repeat(zu.mean(0,keepdims=True),K,0)
        Pl.append(c/(np.linalg.norm(c,axis=1,keepdims=True)+1e-9))
    P=torch.tensor(np.stack(Pl),dtype=torch.float32,device=DEV); S=P.shape[0]; Pf=P.reshape(S*K,128)
    s2i={u:i for i,u in enumerate(subs)}; Xt=X[task]; yi=np.array([s2i[v] for v in y[task]]); g,im=[],[]
    for i in range(0,len(Xt),4096):
        zt=m.zid(torch.tensor(Xt[i:i+4096],dtype=torch.float32,device=DEV)); sims=(zt@Pf.T).reshape(-1,S,K).amax(2)
        idx=torch.tensor(yi[i:i+4096],device=DEV); ar=torch.arange(len(idx),device=DEV)
        g.append(sims[ar,idx].cpu().numpy()); msk=torch.ones_like(sims,dtype=torch.bool); msk[ar,idx]=False; im.append(sims[msk].cpu().numpy())
    g=np.concatenate(g); im=np.concatenate(im); sc=np.concatenate([g,im]); lb=np.concatenate([np.ones_like(g),np.zeros_like(im)])
    fpr,tpr,_=roc_curve(lb,sc); fnr=1-tpr; return float(fpr[np.nanargmin(np.abs(fpr-fnr))]*100)
def ci(arr,n=10000,sd=0):
    r=np.random.default_rng(sd); v=[r.choice(arr,len(arr),replace=True).mean() for _ in range(n)]; return [float(np.percentile(v,2.5)),float(np.percentile(v,97.5))]
d=np.load(a.data,allow_pickle=True); X=d["X"].astype(np.float32); y=d["y"].astype(np.int64)
sess=np.array([int(str(s).lstrip("Rr")) for s in d["session"]],dtype=np.int64)
dom,dan=[],[]
for s in range(1,11):
    pe=os.path.join(a.domcs,f"seed_{s}","model_best.pt"); pn=os.path.join(a.dann,f"seed_{s}","model_best.pt")
    if os.path.exists(pe): dom.append(eer_b2t(load(DOMCS,pe),X,y,sess)); print(f"DOMCS s{s}: {dom[-1]:.3f}")
    if os.path.exists(pn): dan.append(eer_b2t(load(DANN,pn),X,y,sess)); print(f"DANN  s{s}: {dan[-1]:.3f}")
dom=np.array(dom); dan=np.array(dan); n=min(len(dom),len(dan))
res={"domcs_eers":dom.tolist(),"dann_eers":dan.tolist(),"domcs_mean_std":[float(dom.mean()),float(dom.std(ddof=1))],
     "dann_mean_std":[float(dan.mean()),float(dan.std(ddof=1))],"domcs_ci95":ci(dom)}
if n>=2:
    aa,bb=dom[:n],dan[:n]; t=stats.ttest_rel(aa,bb); w=stats.wilcoxon(aa,bb); df=aa-bb
    res["paired"]={"n":int(n),"t":float(t.statistic),"p_one":float(t.pvalue/2),"wilcoxon_p":float(w.pvalue),
                   "cohen_dz":float(df.mean()/(df.std(ddof=1)+1e-12)),"domcs_lower_in":int((aa<bb).sum())}
json.dump(res,open(os.path.join(a.out,"R8_10seed_stats.json"),"w"),indent=2)
print("\nDOMCS %.2f+/-%.2f CI%s | DANN %.2f+/-%.2f | paired:%s"%(dom.mean(),dom.std(ddof=1),res["domcs_ci95"],dan.mean(),dan.std(ddof=1),res.get("paired")))
print("Saved ->",a.out,"/R8_10seed_stats.json")
