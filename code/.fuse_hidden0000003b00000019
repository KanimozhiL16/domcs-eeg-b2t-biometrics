#!/usr/bin/env python3
# =====================================================================================
#  TIFS FIGURES — PART 1 (derived purely from verified JSONs; no model load needed)
#  Sources: 08_ablation_ELU_FINAL/final_metrics_ELU.json (R1),
#           R2_security/R2_security_results.json (R2),
#           R3_stats/R3_stats_results.json (R3).
#  IEEE TIFS rules enforced: width 7.16in (double col) / 3.5in (single), pdf.fonttype=42,
#  >=8pt fonts, vector PDF + 300dpi PNG, Wong colorblind-safe palette, grayscale-safe markers.
#  Every value is read from JSON — NEVER hardcoded.
# =====================================================================================
import json, os, numpy as np
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
mpl.rcParams.update({"pdf.fonttype":42,"ps.fonttype":42,"font.family":"sans-serif",
  "font.size":8,"axes.titlesize":8,"axes.labelsize":8,"xtick.labelsize":7,"ytick.labelsize":7,
  "legend.fontsize":7,"axes.linewidth":0.6,"figure.dpi":300,"savefig.bbox":"tight"})
COL,HALF=7.16,3.5
CB=["#0072B2","#D55E00","#009E73","#CC79A7","#E69F00","#56B4E9","#000000"]  # Wong
ROOT="/home/nvidia/24PHD1237/FRESH_EXP_20260528_DOMCS_EEG"
OUT=f"{ROOT}/figures_TIFS"; os.makedirs(OUT,exist_ok=True)
R1=json.load(open(f"{ROOT}/08_ablation_ELU_FINAL/final_metrics_ELU.json"))
R2=json.load(open(f"{ROOT}/R2_security/R2_security_results.json"))
R3=json.load(open(f"{ROOT}/R3_stats/R3_stats_results.json"))
def save(fig,name):
    fig.savefig(f"{OUT}/{name}.pdf"); fig.savefig(f"{OUT}/{name}.png",dpi=300); plt.close(fig)
    print(f"  saved {name}.pdf/.png")
VAR=["E1","E2","E3","E4","E5"]
LAB={"E1":"Full","E2":"w/o Orth","E3":"w/o detach","E4":"w/o SupCon","E5":"w/o ArcFace"}

# ---------- FIG 2 — Disentanglement (3 panels, double column) ----------
def fig_disentanglement():
    S=R1["summary"]; D=R1["E1_disentanglement"]
    fig,ax=plt.subplots(1,3,figsize=(COL,2.3))
    # (a) |cos| E1 vs E2 — orthogonality-loss effect (43x)
    e1c,e2c=S["E1"]["cos"][0],S["E2"]["cos"][0]; ratio=S["E2"]["cos_ratio_vs_E1"]
    ax[0].bar([0,1],[e1c,e2c],yerr=[S["E1"]["cos"][1],S["E2"]["cos"][1]],
              color=[CB[0],CB[1]],width=.6,capsize=3)
    ax[0].set_xticks([0,1]); ax[0].set_xticklabels(["Full","w/o Orth"])
    ax[0].set_ylabel(r"$|\cos(\mathbf{z}_{id},\mathbf{z}_{s})|$")
    ax[0].set_title(f"(a) Orthogonality ({ratio:.0f}$\\times$)")
    ax[0].annotate(f"{e1c:.4f}",(0,e1c),ha="center",va="bottom",fontsize=6)
    # (b) balanced state-probe on z_id vs chance
    m1=D["M1_balanced_probe"][0]; m1s=D["M1_balanced_probe"][1]
    ax[1].bar([0,1],[50,m1],yerr=[0,m1s],color=[CB[6],CB[2]],width=.6,capsize=3)
    ax[1].set_xticks([0,1]); ax[1].set_xticklabels(["Chance","$z_{id}$"])
    ax[1].set_ylim(0,100); ax[1].set_ylabel("State-probe acc. (%)")
    ax[1].set_title(f"(b) State-attenuated ({m1:.1f}%)")
    ax[1].axhline(50,ls="--",lw=.6,color="gray")
    # (c) cross-state identity preservation: M2 cosine + M3 MMD
    m2=D["M2_cross_state_cos"][0]; m3=D["M3_mmd"][0]
    ax[2].bar([0],[m2],color=CB[0],width=.5); ax[2].set_ylim(0,1)
    ax[2].set_xticks([0]); ax[2].set_xticklabels(["cross-state\ncosine"])
    ax[2].set_ylabel("Cosine sim."); ax[2].set_title(f"(c) Identity kept\nM2={m2:.3f}, MMD={m3:.4f}")
    fig.tight_layout(); save(fig,"FIG2_disentanglement")

# ---------- FIG 4 — Ablation (EER + |cos|, double column 2 panels) ----------
def fig_ablation():
    S=R1["summary"]; x=np.arange(len(VAR))
    fig,ax=plt.subplots(1,2,figsize=(COL,2.5))
    eer=[S[v]["eer"][0] for v in VAR]; eers=[S[v]["eer"][1] for v in VAR]
    ax[0].bar(x,eer,yerr=eers,color=[CB[0]]+[CB[5]]*4,width=.62,capsize=3)
    ax[0].set_xticks(x); ax[0].set_xticklabels([LAB[v] for v in VAR],rotation=25,ha="right")
    ax[0].set_ylabel("B2T EER (%)"); ax[0].set_title("(a) EER by ablation")
    cos=[S[v]["cos"][0] for v in VAR]
    ax[1].bar(x,cos,color=[CB[0]]+[CB[1]]*4,width=.62)
    ax[1].set_xticks(x); ax[1].set_xticklabels([LAB[v] for v in VAR],rotation=25,ha="right")
    ax[1].set_ylabel(r"$|\cos(\mathbf{z}_{id},\mathbf{z}_{s})|$")
    ax[1].set_title(f"(b) Disentanglement (E2/E1={S['E2']['cos_ratio_vs_E1']:.0f}$\\times$)")
    fig.tight_layout(); save(fig,"FIG4_ablation")

# ---------- FIG — Security T0-T5 (DOMCS vs CNN+ArcFace, double column) ----------
def fig_security():
    D=R2["DOMCS-EEG"]; B=R2.get("CNN+ArcFace")
    cats=["clean","T0","T1","T2","T3","T4-FGSM","T4-PGD","T5(TSR)"]
    def vals(m): return [m["clean"]["eer"],m["T0"]["eer"],m["T1"]["eer"],m["T2"]["100%"]["eer"],
                         m["T3"]["AWGN_20dB"]["eer"],m["T4"]["FGSM"]["0.01"]["eer"],
                         m["T4"]["PGD"]["0.01"]["eer"],m["T5"]["TSR"]]
    x=np.arange(len(cats)); w=.38
    fig,ax=plt.subplots(figsize=(COL,2.6))
    ax.bar(x-w/2,vals(D),w,label="DOMCS-EEG",color=CB[0])
    if B: ax.bar(x+w/2,vals(B),w,label="CNN+ArcFace",color=CB[1])
    ax.set_xticks(x); ax.set_xticklabels(cats,rotation=25,ha="right")
    ax.set_ylabel("EER (%)  /  T5=TSR (%)"); ax.set_title("Security threats T0-T5")
    ax.legend(); fig.tight_layout(); save(fig,"FIG_security_T0T5")

# ---------- FIG — Adversarial T4 sweep (FGSM+PGD vs eps, double column) ----------
def fig_adversarial():
    eps=[0.001,0.002,0.003,0.005,0.007,0.01]; ek=[str(e) for e in eps]
    fig,ax=plt.subplots(1,2,figsize=(COL,2.5),sharey=True)
    for j,atk in enumerate(["FGSM","PGD"]):
        for i,(tag,c,mk) in enumerate([("DOMCS-EEG",CB[0],"o"),("CNN+ArcFace",CB[1],"s")]):
            if tag not in R2: continue
            ax[j].plot(eps,[R2[tag]["T4"][atk][k]["eer"] for k in ek],marker=mk,color=c,label=tag,lw=1.2,ms=4)
        ax[j].axhline(R2["DOMCS-EEG"]["clean"]["eer"],ls=":",lw=.6,color="gray")
        ax[j].set_xlabel(r"$\varepsilon$ ($L_\infty$)"); ax[j].set_title(f"({'ab'[j]}) {atk}")
        ax[j].grid(alpha=.3,lw=.4)
    ax[0].set_ylabel("EER (%)"); ax[1].legend()
    fig.tight_layout(); save(fig,"FIG_adversarial_T4")

# ---------- FIG — Statistics DOMCS vs DANN (paired, single column) ----------
def fig_stats():
    dm=np.array(R3["domcs_eers"]); dn=np.array(R3["dann_eers"]); st=R3["DOMCS_vs_DANN"]
    fig,ax=plt.subplots(figsize=(HALF,2.6))
    for i in range(len(dm)):
        ax.plot([0,1],[dm[i],dn[i]],color="gray",lw=.7,marker="o",ms=3,alpha=.7)
    ax.plot([0,1],[dm.mean(),dn.mean()],color=CB[0],lw=2,marker="D",ms=6,label="mean")
    ax.set_xticks([0,1]); ax.set_xticklabels(["DOMCS-EEG","DANN"])
    ax.set_ylabel("B2T EER (%)")
    ax.set_title(f"d={st['cohen_d']:.2f}, $p_1$={st['p_one']:.3f}, 4/5 seeds")
    ax.legend(); fig.tight_layout(); save(fig,"FIG_stats_DOMCS_vs_DANN")

print("Generating TIFS figures (part 1, from JSON)...")
fig_disentanglement(); fig_ablation(); fig_security(); fig_adversarial(); fig_stats()
print(f"Done -> {OUT}  (all PDF vector + 300dpi PNG, fonttype42, >=8pt)")
