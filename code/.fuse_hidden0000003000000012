#!/usr/bin/env python3
# =====================================================================================
#  R7 LAUNCHER — runs the BED benchmark across 8 GPUs, then aggregates.
#  45 jobs = 9 models (DOMCS + 8 baselines) x 5 seeds. Calls 07_bed_benchmark_worker.py.
#  Outputs: per-job JSON, aggregated BED_benchmark.json + LaTeX table (mean +/- std).
# =====================================================================================
import os, json, time, subprocess, numpy as np, glob
HERE   = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, "07_bed_benchmark_worker.py")
BED    = "/home/nvidia/24PHD1237/BED_DATASET/BED_win2s_step1s_fs128.npz"
OUT    = "/home/nvidia/24PHD1237/FRESH_EXP_20260528_DOMCS_EEG/R7_bed_benchmark"; os.makedirs(OUT,exist_ok=True)
MODELS = ["DOMCS","EEGNet","DeepConvNet","ShallowConvNet","ArcFace-only","SupCon-only","CNN+ArcFace","DANN","PSD+KMeans"]
SEEDS  = [1,2,3,4,5]; NGPU = 8
PSD_SINGLE = True   # PSD has no training -> run 1 seed only (deterministic)

jobs=[]
for m in MODELS:
    sd = [1] if (m=="PSD+KMeans" and PSD_SINGLE) else SEEDS
    for s in sd: jobs.append((m,s))
print(f"Launching {len(jobs)} jobs across {NGPU} GPUs -> {OUT}")

t0=time.time(); i=0
while i < len(jobs):
    batch=jobs[i:i+NGPU]; procs=[]
    for g,(m,s) in enumerate(batch):
        log=open(os.path.join(OUT,f"log_{m.replace('+','_')}_s{s}.txt"),"w")
        p=subprocess.Popen(["python",WORKER,"--model",m,"--seed",str(s),"--gpu",str(g),
                            "--bed",BED,"--out",OUT],stdout=log,stderr=subprocess.STDOUT)
        procs.append((p,m,s,log)); print(f"  ▶ {m}/s{s} GPU{g} PID{p.pid}")
    for p,m,s,log in procs:
        rc=p.wait(); log.close(); print(f"  {'✓' if rc==0 else '✗'} {m}/s{s} rc={rc}")
    i+=NGPU
print(f"All jobs done in {(time.time()-t0)/60:.1f} min")

# ---------- aggregate ----------
agg={}
for m in MODELS:
    rows=[json.load(open(f)) for f in glob.glob(os.path.join(OUT,f"{m.replace('+','_')}_s*.json"))]
    if not rows: agg[m]={"missing":True}; continue
    eer=np.array([r["eer"] for r in rows]); auc=np.array([r["auc"] for r in rows]); crr=np.array([r["crr"] for r in rows])
    cmc=np.mean([r["cmc"] for r in rows],0)
    sd=lambda a: float(a.std(ddof=1)) if len(a)>1 else 0.0
    agg[m]={"n":len(rows),"eer":[float(eer.mean()),sd(eer)],"auc":[float(auc.mean()),sd(auc)],
            "crr":[float(crr.mean()),sd(crr)],"cmc":cmc.tolist()}
json.dump(agg,open(os.path.join(OUT,"BED_benchmark.json"),"w"),indent=2)

# ---------- print + LaTeX (sorted best EER first) ----------
order=sorted([m for m in MODELS if "eer" in agg[m]], key=lambda m: agg[m]["eer"][0])
print("\n================ BED BENCHMARK (enroll r01+r02 -> probe r03, exhaustive EER) ================")
print(f"{'Model':<16}{'EER %':>14}{'AUC':>10}{'CRR %':>12}{'Rank-5 %':>10}")
for m in order:
    a=agg[m]; star=" (proposed)" if m=="DOMCS" else ""
    print(f"{m:<16}{a['eer'][0]:>7.2f}±{a['eer'][1]:<5.2f}{a['auc'][0]:>10.4f}{a['crr'][0]:>9.2f}{'':2}{a['cmc'][4]:>9.2f}{star}")
with open(os.path.join(OUT,"TABLE_BED_benchmark.tex"),"w") as f:
    f.write("% BED benchmark (enroll r01+r02, probe r03, exhaustive EER). DOMCS = proposed.\n")
    f.write("\\begin{tabular}{lccc}\n\\toprule\nMethod & EER (\\%)$\\downarrow$ & AUC$\\uparrow$ & CRR (\\%)$\\uparrow$ \\\\\n\\midrule\n")
    for m in order:
        a=agg[m]; nm=("\\textbf{%s}"%m) if m=="DOMCS" else m
        f.write(f"{nm} & {a['eer'][0]:.2f}\\,$\\pm$\\,{a['eer'][1]:.2f} & {a['auc'][0]:.3f} & {a['crr'][0]:.2f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
print(f"\nSaved -> {OUT}/BED_benchmark.json + TABLE_BED_benchmark.tex")
print("Verify: DOMCS should have the LOWEST EER and HIGHEST AUC/CRR for the proposed-method claim.")
