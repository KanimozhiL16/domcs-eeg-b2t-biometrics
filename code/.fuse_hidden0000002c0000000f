#!/usr/bin/env python3
# =====================================================================================
#  R7 WORKER — BED benchmark: train ONE (model, seed) on ONE GPU, eval EER/AUC/CMC/CRR.
#  Called by the launcher. Models: DOMCS + 8 baselines (EEGMMIDB-matched), all Cin=14.
#  Protocol: enroll r01+r02, probe r03; KMeans K=3 gallery; exhaustive all-impostor EER.
#  Usage: python Cell_R7_BED_worker.py --model DOMCS --seed 1 --gpu 0 --bed <npz> --out <dir>
# =====================================================================================
import argparse, os, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ap=argparse.ArgumentParser()
for a in ["model","bed","out"]: ap.add_argument("--"+a,required=True)
ap.add_argument("--seed",type=int,required=True); ap.add_argument("--gpu",type=int,required=True)
A=ap.parse_args(); os.environ["CUDA_VISIBLE_DEVICES"]=str(A.gpu); DEV=torch.device("cuda:0")
torch.manual_seed(A.seed); np.random.seed(A.seed)
from sklearn.cluster import KMeans; from sklearn.metrics import roc_curve, roc_auc_score
NS=21; K=3; EP=60; BS=256; LR=3e-4; WD=1e-4; S_ARC,M_ARC,TAU=32.0,0.50,0.07
LS,LO,LSU=0.50,0.10,0.30

# ---------- shared pieces ----------
class CB(nn.Module):
    def __init__(s,ic,oc,k):
        super().__init__(); s.net=nn.Sequential(nn.Conv1d(ic,oc,k,padding=k//2,bias=False),nn.BatchNorm1d(oc),nn.ELU(inplace=True))
    def forward(s,x): return s.net(x)
class ArcHead(nn.Module):
    def __init__(s,n,d=128):
        super().__init__(); s.W=nn.Parameter(torch.FloatTensor(n,d)); nn.init.xavier_uniform_(s.W)
    def forward(s,z,y):
        W=F.normalize(s.W,1); c=F.linear(z,W).clamp(-1+1e-7,1-1e-7)
        oh=torch.zeros_like(c).scatter_(1,y.view(-1,1),1.); return (c*(1-oh)+torch.cos(torch.acos(c)+M_ARC)*oh)*S_ARC
def supcon(z,y,tau=TAU):
    n=z.size(0); z=F.normalize(z,1); sim=torch.mm(z,z.T)/tau; sim=sim-sim.detach().max(1,True)[0]
    eye=torch.eye(n,device=z.device); m=y.unsqueeze(0).eq(y.unsqueeze(1)).float()*(1-eye); p=m.sum(1).clamp(min=1)
    e=torch.exp(sim)*(1-eye); lp=sim-torch.log(e.sum(1,True)+1e-9); return (-(m*lp).sum(1)/p).mean()
def orth(zi,zs): return torch.abs(F.cosine_similarity(zi,zs,-1)).mean()
class GRL(torch.autograd.Function):
    @staticmethod
    def forward(c,x,l): c.l=l; return x.view_as(x)
    @staticmethod
    def backward(c,g): return -c.l*g, None

# ---------- 1D CNN backbone (DOMCS-style) -> 128 emb ----------
def enc1d(cin): return nn.Sequential(CB(cin,64,7),CB(64,128,5),CB(128,256,3),nn.AdaptiveAvgPool1d(1))
class CNN1D(nn.Module):                       # ArcFace-only / SupCon-only / DANN backbone
    def __init__(s,cin=14):
        super().__init__(); s.enc=enc1d(cin); s.fc=nn.Linear(256,128,bias=False); s.n=nn.LayerNorm(128)
    def emb(s,x): return F.normalize(s.n(s.fc(s.enc(x).squeeze(-1))),dim=-1)
    def forward(s,x): return s.emb(x)
class DOMCS(nn.Module):
    def __init__(s,cin=14):
        super().__init__(); s.enc=enc1d(cin)
        s.id_fc=nn.Linear(256,128,bias=False); s.id_n=nn.LayerNorm(128)
        s.st_fc=nn.Linear(256,128,bias=False); s.st_n=nn.LayerNorm(128)
    def forward(s,x):
        f=s.enc(x).squeeze(-1); return (F.normalize(s.id_n(s.id_fc(f)),-1),F.normalize(s.st_n(s.st_fc(f.detach())),-1))
    def emb(s,x): return s.forward(x)[0]
class CNNArc(nn.Module):                       # Colab CNN+ArcFace backbone
    def __init__(s,cin=14):
        super().__init__(); s.b=nn.Sequential(CB(cin,64,7),CB(64,128,5),CB(128,256,3),nn.AdaptiveAvgPool1d(1))
        s.e=nn.Sequential(nn.Linear(256,128),nn.BatchNorm1d(128),nn.ReLU(),nn.Linear(128,128),nn.LayerNorm(128))
    def emb(s,x): return F.normalize(s.e(s.b(x).squeeze(-1)),dim=-1)
    def forward(s,x): return s.emb(x)
# ---------- 2D baselines (EEGNet / DeepConvNet / ShallowConvNet) ----------
class EEGNet(nn.Module):
    def __init__(s,cin=14,T=256,d=128):
        super().__init__()
        s.f=nn.Sequential(nn.Conv2d(1,8,(1,64),padding=(0,32),bias=False),nn.BatchNorm2d(8),
            nn.Conv2d(8,16,(cin,1),groups=8,bias=False),nn.BatchNorm2d(16),nn.ELU(),nn.AvgPool2d((1,4)),nn.Dropout(.25),
            nn.Conv2d(16,16,(1,16),padding=(0,8),groups=16,bias=False),nn.Conv2d(16,16,(1,1),bias=False),
            nn.BatchNorm2d(16),nn.ELU(),nn.AvgPool2d((1,8)),nn.Dropout(.25))
        with torch.no_grad(): fl=s.f(torch.zeros(1,1,cin,T)).reshape(1,-1).shape[1]
        s.e=nn.Sequential(nn.Linear(fl,d),nn.LayerNorm(d))
    def emb(s,x): return F.normalize(s.e(s.f(x.unsqueeze(1)).flatten(1)),dim=-1)
    def forward(s,x): return s.emb(x)
class DeepConv(nn.Module):
    def __init__(s,cin=14,T=256,d=128):
        super().__init__()
        s.f=nn.Sequential(nn.Conv2d(1,25,(1,5),bias=False),nn.Conv2d(25,25,(cin,1),bias=False),nn.BatchNorm2d(25),nn.ELU(),nn.MaxPool2d((1,2)),nn.Dropout(.25),
            nn.Conv2d(25,50,(1,5),bias=False),nn.BatchNorm2d(50),nn.ELU(),nn.MaxPool2d((1,2)),nn.Dropout(.25),
            nn.Conv2d(50,100,(1,5),bias=False),nn.BatchNorm2d(100),nn.ELU(),nn.MaxPool2d((1,2)),nn.Dropout(.25),
            nn.Conv2d(100,200,(1,5),bias=False),nn.BatchNorm2d(200),nn.ELU(),nn.MaxPool2d((1,2)),nn.Dropout(.25))
        with torch.no_grad(): fl=s.f(torch.zeros(1,1,cin,T)).reshape(1,-1).shape[1]
        s.e=nn.Sequential(nn.Linear(fl,d),nn.LayerNorm(d))
    def emb(s,x): return F.normalize(s.e(s.f(x.unsqueeze(1)).flatten(1)),dim=-1)
    def forward(s,x): return s.emb(x)
class ShallowConv(nn.Module):
    def __init__(s,cin=14,T=256,d=128):
        super().__init__()
        s.c1=nn.Conv2d(1,40,(1,13),bias=False); s.c2=nn.Conv2d(40,40,(cin,1),bias=False); s.bn=nn.BatchNorm2d(40)
        s.pool=nn.AvgPool2d((1,35),stride=(1,7)); s.drop=nn.Dropout(.25)
        with torch.no_grad():
            z=s.pool(s.bn(s.c2(s.c1(torch.zeros(1,1,cin,T)))**2)); fl=z.reshape(1,-1).shape[1]
        s.e=nn.Sequential(nn.Linear(fl,d),nn.LayerNorm(d))
    def emb(s,x):
        h=s.drop(torch.log(torch.clamp(s.pool(s.bn(s.c2(s.c1(x.unsqueeze(1)))**2)),1e-6)))
        return F.normalize(s.e(h.flatten(1)),dim=-1)
    def forward(s,x): return s.emb(x)

# model registry: class, loss-type
REG={"DOMCS":(DOMCS,"domcs"),"EEGNet":(EEGNet,"ce"),"DeepConvNet":(DeepConv,"ce"),
     "ShallowConvNet":(ShallowConv,"ce"),"ArcFace-only":(CNN1D,"arc"),"SupCon-only":(CNN1D,"supcon"),
     "CNN+ArcFace":(CNNArc,"arc"),"DANN":(CNN1D,"dann"),"PSD+KMeans":(None,"psd")}

# ---------- data ----------
d=np.load(A.bed,allow_pickle=True); X=d["X"].astype(np.float32); y=d["y"].astype(np.int64)
runs=np.asarray([str(r) for r in (d["session"] if "session" in d.files else d["runs"])],dtype=object)
SUBJ=sorted(np.unique(y)); s2i={s:i for i,s in enumerate(SUBJ)}
trm=np.isin(runs,["r01","r02"]); tem=(runs=="r03")
Xtr,ytr,rtr=X[trm],y[trm],runs[trm]; Xte,yte=X[tem],y[tem]

def psd_feat(Xn):                              # bandpower 5 bands x 14 ch -> 70-d
    F_=np.fft.rfft(Xn,axis=-1); P=np.abs(F_)**2; fr=np.fft.rfftfreq(Xn.shape[-1],1/128)
    bands=[(0.5,4),(4,8),(8,13),(13,30),(30,45)]; out=[]
    for lo,hi in bands: out.append(P[:,:,(fr>=lo)&(fr<hi)].mean(-1))
    f=np.concatenate(out,1).astype(np.float32); return f/(np.linalg.norm(f,axis=1,keepdims=True)+1e-9)

def embed(m,Xn,bs=2048):
    m.eval(); o=[]
    with torch.no_grad():
        for i in range(0,len(Xn),bs): o.append(m.emb(torch.from_numpy(Xn[i:i+bs]).to(DEV)).cpu().numpy())
    return np.concatenate(o)

def metrics(zg,yg,zt,yt):
    G=np.zeros((NS,K,zg.shape[1]),np.float32)
    for s in SUBJ:
        c=KMeans(K,n_init=10,random_state=0).fit(zg[yg==s]).cluster_centers_
        G[s2i[s]]=c/(np.linalg.norm(c,axis=1,keepdims=True)+1e-12)
    Gt=torch.tensor(G.reshape(NS*K,-1),device=DEV); P=torch.tensor(zt,device=DEV); gen=[];imp=[];ranks=[]
    with torch.no_grad():
        for i in range(0,len(P),8192):
            S=(P[i:i+8192]@Gt.T).view(-1,NS,K).max(2).values.cpu().numpy()
            yb=yt[i:i+8192]; r=np.arange(len(yb)); gi=np.array([s2i[s] for s in yb])
            gen.append(S[r,gi]); M=S.copy(); M[r,gi]=np.nan; imp.append(M[~np.isnan(M)])
            order=np.argsort(-S,1)
            for j,s in enumerate(yb): ranks.append(int(np.where(order[j]==s2i[s])[0][0]))
    gen=np.concatenate(gen); imp=np.concatenate(imp); ranks=np.array(ranks)
    sc=np.r_[gen,imp].astype(np.float64); lb=np.r_[np.ones_like(gen),np.zeros_like(imp)]
    fpr,tpr,_=roc_curve(lb,sc); fnr=1-tpr; i=np.nanargmin(np.abs(fnr-fpr))
    eer=(fpr[i]+fnr[i])/2*100; auc=float(roc_auc_score(lb,sc))
    cmc=[100*float(np.mean(ranks<k)) for k in range(1,11)]
    return eer,auc,cmc[0],cmc            # CRR=cmc[0]

# ---------- train + eval ----------
Cls,lt=REG[A.model]
if lt=="psd":
    eer,auc,crr,cmc=metrics(psd_feat(Xtr),ytr,psd_feat(Xte),yte)
else:
    rng=np.random.default_rng(A.seed); idx=rng.permutation(len(Xtr)); nv=int(.1*len(idx)); vi,ti=idx[:nv],idx[nv:]
    Xt,yt2,rt=Xtr[ti],ytr[ti],rtr[ti]; Xv,yv=Xtr[vi],ytr[vi]
    m=Cls(14).to(DEV); heads=[]
    arc=ArcHead(NS).to(DEV) if lt in("arc","domcs","dann") else None
    clf=nn.Linear(128,NS).to(DEV) if lt=="ce" else None
    sth=nn.Linear(128,2).to(DEV) if lt=="domcs" else None
    dom=nn.Linear(128,2).to(DEV) if lt=="dann" else None
    for h in (arc,clf,sth,dom):
        if h is not None: heads+=list(h.parameters())
    opt=torch.optim.Adam(list(m.parameters())+heads,lr=LR,weight_decay=WD)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EP,eta_min=1e-6)
    dlab=(rt=="r02").astype(np.int64)          # DANN domain = session r01(0)/r02(1)
    def comp(xb,yb,db=None):                    # IDENTITY loss only (model-fair val metric)
        if lt=="domcs":
            zi,zs=m(xb); return F.cross_entropy(arc(zi,yb),yb)+LSU*supcon(zi,yb)
        z=m.emb(xb)
        if lt=="ce": return F.cross_entropy(clf(z),yb)
        if lt=="arc": return F.cross_entropy(arc(z,yb),yb)
        if lt=="supcon": return supcon(z,yb)
        if lt=="dann": return F.cross_entropy(arc(z,yb),yb)
    best=1e9; best_sd=None
    for ep in range(1,EP+1):
        m.train(); perm=rng.permutation(len(Xt))
        for i in range(0,len(perm),BS):
            b=perm[i:i+BS]; xb=torch.FloatTensor(Xt[b]).to(DEV); yb=torch.LongTensor(yt2[b]).to(DEV)
            opt.zero_grad()
            if lt=="domcs":
                zi,zs=m(xb); L=F.cross_entropy(arc(zi,yb),yb)+LSU*supcon(zi,yb)
                L=L+LS*F.cross_entropy(sth(zs),torch.zeros(len(yb),dtype=torch.long,device=DEV))+LO*orth(zi,zs)
            else:
                z=m.emb(xb)
                if lt=="ce": L=F.cross_entropy(clf(z),yb)
                elif lt=="arc": L=F.cross_entropy(arc(z,yb),yb)
                elif lt=="supcon": L=supcon(z,yb)
                elif lt=="dann":
                    db=torch.LongTensor(dlab[b]).to(DEV)
                    L=F.cross_entropy(arc(z,yb),yb)+0.3*F.cross_entropy(dom(GRL.apply(z,1.0)),db)
            L.backward(); torch.nn.utils.clip_grad_norm_(list(m.parameters())+heads,1.0); opt.step()
        sch.step()
        m.eval()                                # best-val checkpoint (identity loss)
        with torch.no_grad():
            vl=[]
            for i in range(0,len(Xv),BS*4):
                xv=torch.FloatTensor(Xv[i:i+BS*4]).to(DEV); yvb=torch.LongTensor(yv[i:i+BS*4]).to(DEV)
                vl.append(comp(xv,yvb).item())
            v=float(np.mean(vl))
        if v<best: best=v; best_sd={k:vv.cpu().clone() for k,vv in m.state_dict().items()}
    m.load_state_dict(best_sd)                  # restore best-val before eval
    eer,auc,crr,cmc=metrics(embed(m,Xtr),ytr,embed(m,Xte),yte)

res={"model":A.model,"seed":A.seed,"eer":eer,"auc":auc,"crr":crr,"cmc":cmc}
json.dump(res,open(os.path.join(A.out,f"{A.model.replace('+','_')}_s{A.seed}.json"),"w"),indent=2)
print(f"[{A.model}/s{A.seed}] EER={eer:.3f}% AUC={auc:.4f} CRR={crr:.2f}%")
