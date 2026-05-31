"""
ablation_worker_elu.py
======================
Single-job worker: trains ONE (variant, seed) on ONE GPU.
Called by the multi-GPU launcher. Do NOT run directly.

Usage (internal):
  python ablation_worker_elu.py --variant E2 --seed 3 --gpu 2 \
         --data /path/data.npz --out /path/08_ablation_ELU_FINAL
"""

import argparse, os, time, json
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

# ── ARGS ───────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--variant", required=True)
parser.add_argument("--seed",    type=int, required=True)
parser.add_argument("--gpu",     type=int, required=True)
parser.add_argument("--data",    required=True)
parser.add_argument("--out",     required=True)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
DEVICE = torch.device("cuda:0")

# ── HYPERPARAMETERS ────────────────────────────────────────────
N_EPOCHS   = 60
BATCH_SIZE = 256
LR, WD     = 3e-4, 1e-4
ARC_S, ARC_M, TAU = 32.0, 0.50, 0.07
LAM_STATE, LAM_ORTH, LAM_SUP = 0.50, 0.10, 0.30

VARIANT_CFG = {
    # (model_type, use_arcface, lam_sup, use_orth)
    "E1": ("standard",  True,  LAM_SUP, True),
    "E2": ("standard",  True,  LAM_SUP, False),
    "E3": ("nodetach",  True,  LAM_SUP, True),
    "E4": ("standard",  True,  0.0,     True),
    "E5": ("standard",  False, LAM_SUP, True),
}

# ── ARCHITECTURE (ELU — model_definition_FIXED.py) ─────────────
class ConvBlock(nn.Module):
    def __init__(self, ic, oc, k):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(ic, oc, kernel_size=k, padding=k//2, bias=False),
            nn.BatchNorm1d(oc), nn.ELU(inplace=True))
    def forward(self, x): return self.net(x)

def _encoder():
    return nn.Sequential(
        ConvBlock(64,64,7), ConvBlock(64,128,5),
        ConvBlock(128,256,3), nn.AdaptiveAvgPool1d(1))

class DOMCSStandard(nn.Module):
    """f.detach() in state branch (E1/E2/E4/E5)."""
    def __init__(self):
        super().__init__()
        self.enc = _encoder()
        self.id_fc=nn.Linear(256,128,bias=False); self.id_n=nn.LayerNorm(128)
        self.st_fc=nn.Linear(256,128,bias=False); self.st_n=nn.LayerNorm(128)
    def forward(self, x):
        f = self.enc(x).squeeze(-1)
        return (F.normalize(self.id_n(self.id_fc(f)),          dim=-1),
                F.normalize(self.st_n(self.st_fc(f.detach())), dim=-1))

class DOMCSNoDetach(nn.Module):
    """No f.detach() — state gradient reaches encoder (E3)."""
    def __init__(self):
        super().__init__()
        self.enc = _encoder()
        self.id_fc=nn.Linear(256,128,bias=False); self.id_n=nn.LayerNorm(128)
        self.st_fc=nn.Linear(256,128,bias=False); self.st_n=nn.LayerNorm(128)
    def forward(self, x):
        f = self.enc(x).squeeze(-1)
        return (F.normalize(self.id_n(self.id_fc(f)), dim=-1),
                F.normalize(self.st_n(self.st_fc(f)), dim=-1))

class ArcFaceHead(nn.Module):
    def __init__(self, n=109):
        super().__init__()
        self.s=ARC_S; self.m=ARC_M
        self.W=nn.Parameter(torch.FloatTensor(n,128)); nn.init.xavier_uniform_(self.W)
    def forward(self, z, y):
        W=F.normalize(self.W,dim=1); c=F.linear(z,W).clamp(-1+1e-7,1-1e-7)
        oh=torch.zeros_like(c).scatter_(1,y.view(-1,1),1.0)
        return (c*(1-oh)+torch.cos(torch.acos(c)+self.m)*oh)*self.s

class StateHead(nn.Module):
    def __init__(self): super().__init__(); self.fc=nn.Linear(128,2)
    def forward(self, z): return self.fc(z)

def supcon(z, y, tau=TAU):
    n = z.size(0)
    z = F.normalize(z, dim=1)
    sim = torch.mm(z, z.T) / tau
    sim = sim - sim.detach().max(dim=1,keepdim=True)[0]
    eye = torch.eye(n, device=z.device)
    mask = y.unsqueeze(0).eq(y.unsqueeze(1)).float() * (1-eye)
    pos  = mask.sum(1).clamp(min=1)
    exp  = torch.exp(sim) * (1-eye)
    log_p = sim - torch.log(exp.sum(1,keepdim=True)+1e-9)
    return (-(mask*log_p).sum(1)/pos).mean()

def orth_loss(zi, zs):
    return torch.abs(F.cosine_similarity(zi, zs, dim=-1)).mean()

# ── DATA ───────────────────────────────────────────────────────
print(f"[{args.variant}/seed{args.seed}/GPU{args.gpu}] Loading data...")
d    = np.load(args.data, allow_pickle=True)
X    = d['X'].astype(np.float32)
y    = d['y'].astype(np.int64)
sess = np.array([int(str(s).lstrip('Rr')) for s in d['session']], dtype=np.int64)
state = (sess>=3).astype(np.int64)

# ── TRAIN/VAL SPLIT ────────────────────────────────────────────
rng = np.random.default_rng(args.seed)
idx = rng.permutation(len(X))
n_val = int(0.10*len(idx))
vi, ti = idx[:n_val], idx[n_val:]
Xtr,ytr,str_ = X[ti],y[ti],state[ti]
Xvl,yvl,svl  = X[vi],y[vi],state[vi]
rest_tr = (str_==0)

# ── BUILD MODEL ────────────────────────────────────────────────
model_type, use_arc, lam_sup, use_orth = VARIANT_CFG[args.variant]
model = (DOMCSStandard() if model_type=="standard" else DOMCSNoDetach()).to(DEVICE)
arc   = ArcFaceHead().to(DEVICE) if use_arc else None
st_h  = StateHead().to(DEVICE)

params = list(model.parameters()) + list(st_h.parameters())
if arc: params += list(arc.parameters())
opt   = torch.optim.Adam(params, lr=LR, weight_decay=WD)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS, eta_min=1e-6)

best_loss, best_ep, best_state = float('inf'), 0, None
t0 = time.time()

for ep in range(1, N_EPOCHS+1):
    model.train(); st_h.train()
    if arc: arc.train()
    perm = rng.permutation(len(Xtr))
    tl, nb = 0.0, 0

    for i in range(0, len(perm), BATCH_SIZE):
        bi  = perm[i:i+BATCH_SIZE]
        xb  = torch.FloatTensor(Xtr[bi]).to(DEVICE)
        yb  = torch.LongTensor(ytr[bi]).to(DEVICE)
        sb  = torch.LongTensor(str_[bi]).to(DEVICE)
        rb  = (str_[bi]==0)

        opt.zero_grad()
        zi, zs = model(xb)
        loss = torch.tensor(0., device=DEVICE)

        if rb.sum()>1:
            if use_arc and arc:
                loss = loss + F.cross_entropy(arc(zi[rb], yb[rb]), yb[rb])
            if lam_sup>0:
                loss = loss + lam_sup*supcon(zi[rb], yb[rb])

        loss = loss + LAM_STATE*F.cross_entropy(st_h(zs), sb)
        if use_orth:
            loss = loss + LAM_ORTH*orth_loss(zi, zs)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        tl += loss.item(); nb += 1

    sched.step()

    # Validation
    model.eval(); st_h.eval()
    if arc: arc.eval()
    vl = []
    with torch.no_grad():
        for j in range(0, len(Xvl), BATCH_SIZE*4):
            xv=torch.FloatTensor(Xvl[j:j+BATCH_SIZE*4]).to(DEVICE)
            yv=torch.LongTensor(yvl[j:j+BATCH_SIZE*4]).to(DEVICE)
            sv=torch.LongTensor(svl[j:j+BATCH_SIZE*4]).to(DEVICE)
            rv=(svl[j:j+BATCH_SIZE*4]==0)
            zi_v,zs_v=model(xv)
            v=torch.tensor(0.,device=DEVICE)
            if rv.sum()>1 and use_arc and arc:
                v=v+F.cross_entropy(arc(zi_v[rv],yv[rv]),yv[rv])
            if rv.sum()>1 and lam_sup>0:
                v=v+lam_sup*supcon(zi_v[rv],yv[rv])
            v=v+LAM_STATE*F.cross_entropy(st_h(zs_v),sv)
            if use_orth: v=v+LAM_ORTH*orth_loss(zi_v,zs_v)
            vl.append(v.item())

    val = float(np.mean(vl))
    if val < best_loss:
        best_loss=val; best_ep=ep
        best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}

    if ep%10==0 or ep==N_EPOCHS:
        print(f"  [{args.variant}/s{args.seed}/G{args.gpu}] "
              f"ep{ep:3d} train={tl/nb:.4f} val={val:.5f} "
              f"best_ep={best_ep} {(time.time()-t0)/60:.1f}min")

# ── SAVE ───────────────────────────────────────────────────────
ckpt_dir = os.path.join(args.out, args.variant, f"seed_{args.seed}")
os.makedirs(ckpt_dir, exist_ok=True)
torch.save({"model_state": best_state, "best_epoch": best_ep,
            "best_val_loss": best_loss, "variant": args.variant,
            "seed": args.seed, "activation": "ELU",
            "use_orth": use_orth, "use_arcface": use_arc, "lam_sup": lam_sup},
           os.path.join(ckpt_dir, "model_best.pt"))

print(f"  [{args.variant}/s{args.seed}/G{args.gpu}] ✓ DONE "
      f"best_ep={best_ep} val={best_loss:.5f} "
      f"total={(time.time()-t0)/60:.1f}min")
print(f"  SAVED: {ckpt_dir}/model_best.pt")
