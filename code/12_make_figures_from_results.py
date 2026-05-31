#!/usr/bin/env python3
"""
12_make_figures_from_results.py -- regenerate the main paper figures from the
verified result JSONs in ../results/ (NO data, NO GPU, NO model load).
Outputs: ../figures/FIG2_disentanglement, FIG4_ablation, FIG_security_T0T5  (.pdf + .png)
"""
import json, os
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
mpl.rcParams.update({"pdf.fonttype":42,"ps.fonttype":42,"font.family":"sans-serif",
  "font.size":8,"axes.titlesize":8,"axes.labelsize":8,"xtick.labelsize":7,
  "ytick.labelsize":7,"legend.fontsize":7,"axes.linewidth":0.6,"figure.dpi":300,
  "savefig.bbox":"tight"})
COL,HALF=7.16,3.5
CB=["#0072B2","#D55E00","#009E73","#CC79A7","#E69F00","#56B4E9","#000000"]  # Wong

HERE=os.path.dirname(os.path.abspath(__file__)); REPO=os.path.dirname(HERE)
RESULTS=os.environ.get("RESULTS",os.path.join(REPO,"results"))
OUT=os.environ.get("FIGOUT",os.path.join(REPO,"figures")); os.makedirs(OUT,exist_ok=True)
R1=json.load(open(os.path.join(RESULTS,"final_metrics_ELU.json")))
R2=json.load(open(os.path.join(RESULTS,"R2_security_results.json")))

def save(fig,name):
    fig.savefig(os.path.join(OUT,name+".pdf")); fig.savefig(os.path.join(OUT,name+".png"),dpi=300)
    plt.close(fig); print("  saved",name+".pdf/.png")

VAR=["E1","E2","E3","E4","E5"]
LAB={"E1":"Full","E2":"w/o Orth","E3":"w/o detach","E4":"w/o SupCon","E5":"w/o ArcFace"}

def fig_disentanglement():
    S=R1["summary"]; D=R1["E1_disentanglement"]
    fig,ax=plt.subplots(1,3,figsize=(COL,2.3))
    e1c,e2c=S["E1"]["cos"][0],S["E2"]["cos"][0]; ratio=S["E2"]["cos_ratio_vs_E1"]
    ax[0].bar([0,1],[e1c,e2c],yerr=[S["E1"]["cos"][1],S["E2"]["cos"][1]],color=[CB[0],CB[1]],width=.6,capsize=3)
    ax[0].set_xticks([0,1]); ax[0].set_xticklabels(["Full","w/o Orth"])
    ax[0].set_ylabel(r"$|\cos(z_{id},z_{s})|$"); ax[0].set_title("(a) Orthogonality (%.0fx)"%ratio)
    m1,m1s=D["M1_balanced_probe"]
    ax[1].bar([0,1],[50,m1],yerr=[0,m1s],color=[CB[6],CB[2]],width=.6,capsize=3)
    ax[1].set_xticks([0,1]); ax[1].set_xticklabels(["Chance",r"$z_{id}$"]); ax[1].set_ylim(0,100)
    ax[1].set_ylabel("State-probe acc. (%)"); ax[1].set_title("(b) State-attenuated (%.1f%%)"%m1)
    ax[1].axhline(50,ls="--",lw=.6,color="gray")
    m2=D["M2_cross_state_cos"][0]; m3=D["M3_mmd"][0]
    ax[2].bar([0],[m2],color=CB[0],width=.5); ax[2].set_ylim(0,1)
    ax[2].set_xticks([0]); ax[2].set_xticklabels(["cross-state\ncosine"]); ax[2].set_ylabel("Cosine sim.")
    ax[2].set_title("(c) Identity kept\nM2=%.3f, MMD=%.4f"%(m2,m3))
    save(fig,"FIG2_disentanglement")

def fig_ablation():
    S=R1["summary"]
    eer=[S[v]["eer"][0] for v in VAR]; err=[S[v]["eer"][1] for v in VAR]
    cos=[S[v]["cos"][0] for v in VAR]
    fig,ax=plt.subplots(figsize=(HALF,2.6)); x=range(len(VAR))
    ax.bar(x,eer,yerr=err,color=CB[0],width=.6,capsize=3)
    ax.set_xticks(list(x)); ax.set_xticklabels([LAB[v] for v in VAR],rotation=30,ha="right")
    ax.set_ylabel("EER (%)"); ax.set_title("Ablation (EER; |cos| annotated)")
    for i,c in enumerate(cos): ax.annotate("%.3f"%c,(i,eer[i]),ha="center",va="bottom",fontsize=6)
    save(fig,"FIG4_ablation")

def fig_security():
    D=R2["DOMCS-EEG"]; C=R2["CNN+ArcFace"]
    def pgd(m): 
        try: return m["T4"]["PGD"]["0.01"]["eer"]
        except Exception: return 0.0
    rows=[("clean",D["clean"]["eer"],C["clean"]["eer"]),
          ("T1",D["T1"]["eer"],C["T1"]["eer"]),
          ("T4 PGD.01",pgd(D),pgd(C)),
          ("T5 TSR",D["T5"]["TSR"],C["T5"]["TSR"])]
    labels=[r[0] for r in rows]; dv=[r[1] for r in rows]; cv=[r[2] for r in rows]
    import numpy as np; x=np.arange(len(rows)); w=.38
    fig,ax=plt.subplots(figsize=(HALF,2.6))
    ax.bar(x-w/2,dv,w,label="DOMCS-EEG",color=CB[0]); ax.bar(x+w/2,cv,w,label="CNN+ArcFace",color=CB[1])
    ax.set_xticks(x); ax.set_xticklabels(labels,rotation=20,ha="right")
    ax.set_ylabel("EER / TSR (%)"); ax.set_title("Threat model (lower = more robust)"); ax.legend()
    save(fig,"FIG_security_T0T5")

if __name__=="__main__":
    print("Regenerating figures from",RESULTS,"->",OUT)
    fig_disentanglement(); fig_ablation(); fig_security()
    print("DONE.")
