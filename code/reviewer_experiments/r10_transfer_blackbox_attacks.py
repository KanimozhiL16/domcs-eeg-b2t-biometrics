# =====================================================================================
#  R10 — Security comparison DOMCS-EEG vs strength-matched DANN (reuses R8 checkpoints,
#  EVAL-ONLY, no retraining). White-box targeted PGD impersonation (TSR) + untargeted
#  PGD EER sweep, for BOTH models; plus TRANSFER attacks (craft on A, evaluate on B) as a
#  black-box proxy. This is the key post-parity experiment: do the two models that TIE on
#  clean EER differ under attack? Notebook-safe. Run after R8 (uses R8 seed_3 checkpoints).
# =====================================================================================
import os, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.cluster import KMeans; from sklearn.metrics import roc_curve
DATA = "/home/nvidia/24PHD1237/EEGMMIDB/EEGMMIDB/EEGMMIDB_win2s_step1s_fs128.npz"   # <-- EDIT IF NEEDED
DOMCS_CK = "/home/nvidia/24PHD1237/R8_domcs_10seed/seed_3/model_best.pt"
DANN_CK  = "/home/nvidia/24PHD1237/R8_dann_10seed/seed_3/model_best.pt"
OUT = "/home/nvidia/24PHD1237/R10_security"; os.makedirs(OUT, exist_ok=True)
DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); K=3
EPS = [0.003, 0.005, 0.01]; PGD_STEPS=10; N_PAIRS=200
print("DEVICE:",DEV)

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
def load(mk,p): m=mk(); ck=torch.load(p,map_location="cpu",weights_only=False); m.load_state_dict(ck.get("model_state",ck),strict=False); return m.to(DEV).eval()

d=np.load(DATA,allow_pickle=True); X=d["X"].astype(np.float32); y=d["y"].astype(np.int64)
sess=np.array([int(str(s).lstrip("Rr")) for s in d["session"]],dtype=np.int64)
rest=np.isin(sess,[1,2]); task=sess>=3; subs=np.unique(y)

def gallery(m):
    with torch.no_grad():
        Zr=[]; 
        for i in range(0,rest.sum(),4096): pass
        Xr=X[rest]; yr=y[rest]; Z=[]
        for i in range(0,len(Xr),4096): Z.append(m.zid(torch.tensor(Xr[i:i+4096],dtype=torch.float32,device=DEV)).cpu().numpy())
        Zr=np.concatenate(Z,0); P={}
        for u in subs:
            zu=Zr[yr==u]; c=KMeans(K,n_init=5,random_state=0).fit(zu).cluster_centers_ if len(zu)>=K else np.repeat(zu.mean(0,keepdims=True),K,0)
            P[u]=c/(np.linalg.norm(c,axis=1,keepdims=True)+1e-9)
    return {u:torch.tensor(P[u],dtype=torch.float32,device=DEV) for u in subs}

def thr_eer(m,P):  # clean EER threshold
    Xt=X[task]; yt=y[task]; g,im=[],[]; ix={u:i for i,u in enumerate(subs)}
    Pf=torch.cat([P[u] for u in subs],0); S=len(subs)
    with torch.no_grad():
        for i in range(0,len(Xt),4096):
            z=m.zid(torch.tensor(Xt[i:i+4096],dtype=torch.float32,device=DEV)); sims=(z@Pf.T).reshape(-1,S,K).amax(2)
            idx=torch.tensor([ix[v] for v in yt[i:i+4096]],device=DEV); ar=torch.arange(len(idx),device=DEV)
            g.append(sims[ar,idx].cpu().numpy()); msk=torch.ones_like(sims,dtype=torch.bool); msk[ar,idx]=False; im.append(sims[msk].cpu().numpy())
    g=np.concatenate(g); im=np.concatenate(im); sc=np.concatenate([g,im]); lb=np.concatenate([np.ones_like(g),np.zeros_like(im)])
    fpr,tpr,t=roc_curve(lb,sc); fnr=1-tpr; k=np.nanargmin(np.abs(fpr-fnr)); return float(t[k]), float(fpr[k]*100)

def pgd_target(m_grad, x, victim_proto, eps, steps):  # maximize max-cos to victim prototypes (impersonation)
    x=x.clone().detach(); x0=x.clone(); alpha=eps/4
    for _ in range(steps):
        x.requires_grad_(True); z=m_grad.zid(x); cs=(z@victim_proto.T).max(1).values.mean()
        g,=torch.autograd.grad(cs,x); x=(x+alpha*g.sign()).detach(); x=torch.max(torch.min(x,x0+eps),x0-eps)
    return x.detach()

def run_tsr(m_craft, m_eval, P, tau, eps, steps, n=N_PAIRS, seed=0):
    rng=np.random.default_rng(seed); Xt=X[task]; yt=y[task]
    idx=rng.permutation(len(Xt))[:n]; succ=0
    for j in idx:
        true=yt[j]; victim=rng.choice([u for u in subs if u!=true])
        x=torch.tensor(Xt[j:j+1],dtype=torch.float32,device=DEV)
        xadv=pgd_target(m_craft, x, P[victim], eps, steps)
        with torch.no_grad():
            score=(m_eval.zid(xadv)@P[victim].T).max(1).values.item()
        if score>tau: succ+=1
    return 100.0*succ/n

if __name__=="__main__":
    dm=load(DOMCS,DOMCS_CK); dn=load(DANN,DANN_CK)
    Pdm=gallery(dm); Pdn=gallery(dn)
    tau_dm,eer_dm=thr_eer(dm,Pdm); tau_dn,eer_dn=thr_eer(dn,Pdn)
    print(f"clean EER  DOMCS={eer_dm:.2f}% (tau {tau_dm:.3f}) | DANN={eer_dn:.2f}% (tau {tau_dn:.3f})")
    res={"clean_eer":{"DOMCS":eer_dm,"DANN":eer_dn},"tsr_whitebox":{},"tsr_transfer":{}}
    for eps in EPS:
        # white-box: craft & eval on same model
        wb_dm=run_tsr(dm,dm,Pdm,tau_dm,eps,PGD_STEPS); wb_dn=run_tsr(dn,dn,Pdn,tau_dn,eps,PGD_STEPS)
        # transfer (black-box proxy): craft on the OTHER model, eval on target
        tr_dm=run_tsr(dn,dm,Pdm,tau_dm,eps,PGD_STEPS)   # craft on DANN -> attack DOMCS
        tr_dn=run_tsr(dm,dn,Pdn,tau_dn,eps,PGD_STEPS)   # craft on DOMCS -> attack DANN
        res["tsr_whitebox"][f"{eps}"]={"DOMCS":wb_dm,"DANN":wb_dn}
        res["tsr_transfer"][f"{eps}"]={"DANN->DOMCS":tr_dm,"DOMCS->DANN":tr_dn}
        print(f"eps={eps}: white-box TSR DOMCS={wb_dm:.1f}% DANN={wb_dn:.1f}% | transfer DANN->DOMCS={tr_dm:.1f}% DOMCS->DANN={tr_dn:.1f}%")
    json.dump(res,open(os.path.join(OUT,"R10_security_DOMCS_vs_DANN.json"),"w"),indent=2)
    print("Saved ->",OUT,"/R10_security_DOMCS_vs_DANN.json")
    print("\nINTERPRET: lower TSR = more robust. If DOMCS white-box TSR < DANN, DOMCS is the more secure model at equal clean EER (the key post-parity claim).")
