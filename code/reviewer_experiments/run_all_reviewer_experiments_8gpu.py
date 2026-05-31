# Run this ONE cell. Spawns DOMCS x10 + DANN x10 + EEGNet x3 across 8 GPUs (waves), then eval+aggregate.
import os, sys, subprocess, time, json, glob
WDIR = os.getcwd()                       # folder containing the worker .py files (the notebook's cwd)
PY   = sys.executable
DATA = "/home/nvidia/24PHD1237/EEGMMIDB/EEGMMIDB/EEGMMIDB_win2s_step1s_fs128.npz"   # <-- EDIT IF NEEDED
BASE = "/home/nvidia/24PHD1237"
DOUT, NOUT, EOUT, SOUT = f"{BASE}/R8_domcs_10seed", f"{BASE}/R8_dann_10seed", f"{BASE}/R9_eegnet_sametask", f"{BASE}/R8_stats"
NGPU = 8
for p in (DOUT,NOUT,EOUT,SOUT): os.makedirs(p,exist_ok=True)
jobs=[("r8_domcs_worker.py",s,DOUT) for s in range(1,11)] \
   + [("r8_dann_worker.py", s,NOUT) for s in range(1,11)] \
   + [("r9_eegnet_sametask_worker.py",s,EOUT) for s in (1,2,3)]
print(f"Launching {len(jobs)} jobs across {NGPU} GPUs from {WDIR}"); t0=time.time(); i=0
while i < len(jobs):
    wave=jobs[i:i+NGPU]; procs=[]
    for g,(w,s,od) in enumerate(wave):
        lg=open(os.path.join(od,f"log_s{s}.txt"),"w")
        p=subprocess.Popen([PY, os.path.join(WDIR,w), "--seed",str(s), "--gpu",str(g), "--data",DATA, "--out",od],
                           stdout=lg, stderr=subprocess.STDOUT)
        procs.append((p,w,s,lg)); print(f"  > {w} seed{s} -> GPU{g} PID{p.pid}")
    for p,w,s,lg in procs:
        rc=p.wait(); lg.close(); print(f"  {'OK' if rc==0 else 'FAIL rc=%d'%rc} {w} seed{s}")
    i+=NGPU
print(f"All training done in {(time.time()-t0)/60:.1f} min")
# ---- eval + stats (fast vectorized) ----
print("\nEvaluating 10-seed stats..."); 
subprocess.run([PY, os.path.join(WDIR,"r8_eval_10seed_stats.py"), "--data",DATA, "--domcs",DOUT, "--dann",NOUT, "--out",SOUT])
# ---- aggregate EEGNet same-task ----
ej=[json.load(open(f)) for f in glob.glob(f"{EOUT}/seed_*.json")]
if ej:
    import numpy as np
    acc=np.array([e["id_acc"] for e in ej]); eer=np.array([e["verif_eer"] for e in ej])
    agg={"id_acc_mean_std":[float(acc.mean()),float(acc.std(ddof=1))],"verif_eer_mean_std":[float(eer.mean()),float(eer.std(ddof=1))],
         "b2t_eer_reference":47.88,"note":"EEGNet same-task; B2T collapse is protocol-driven"}
    json.dump(agg,open(f"{EOUT}/R9_eegnet_sametask.json","w"),indent=2)
    print(f"EEGNet same-task: id-acc={acc.mean():.2f}% verif-EER={eer.mean():.2f}% (vs B2T 47.88%)")
print("\nDONE. Outputs:")
print(f"  {SOUT}/R8_10seed_stats.json"); print(f"  {EOUT}/R9_eegnet_sametask.json")
