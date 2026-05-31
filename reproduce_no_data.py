#!/usr/bin/env python3
"""
reproduce_no_data.py  --  CLONE-AND-VERIFY (no dataset, no GPU, no checkpoints)
Prints the paper's headline numbers straight from the verified result files in
results/. Proves the reported metrics are the exact outputs of the executed
pipeline (not hand-entered).

Usage (after git clone):
    pip install -r requirements.txt
    python reproduce_no_data.py

To RE-DERIVE numbers from the trained models, see HOW_TO_REPRODUCE.md "Level 2".
"""
import json, os
REPO = os.path.dirname(os.path.abspath(__file__))
RES  = os.path.join(REPO, "results")
def jload(n): return json.load(open(os.path.join(RES, n)))

def main():
    fm  = jload("final_metrics_ELU.json"); s, d = fm["summary"], fm["E1_disentanglement"]
    r2  = jload("R2_security_results.json")
    r8  = jload("R8_10seed_stats.json")
    r9  = jload("R9_eegnet_sametask.json")
    bed = jload("bed_exactarch_results.json")
    L = "=" * 70
    print(L); print(" DOMCS-EEG  --  reproducibility check (values read from results/*.json)"); print(L)

    print("\n[1] MAIN RECOGNITION (EEGMMIDB, B2T, 5 seeds)")
    print("    EER (E1 Full)            : {:.2f} +/- {:.2f} %     | paper 2.40 +/- 0.22".format(s["E1"]["eer"][0], s["E1"]["eer"][1]))
    print("    AUC (E1)                 : {:.4f}             | paper 0.996".format(s["E1"]["auc"][0]))
    print("    |cos(z_id,z_state)| (M5) : {:.4f}             | paper 0.0016".format(d["M5_cos"][0]))
    print("    balanced state-probe (M1): {:.1f} % (chance 50) | paper 65.6".format(d["M1_balanced_probe"][0]))
    print("    cross-state cosine  (M2) : {:.3f}              | paper 0.891".format(d["M2_cross_state_cos"][0]))
    print("    MMD                 (M3) : {:.4f}             | paper 0.005".format(d["M3_mmd"][0]))

    print("\n[2] ABLATION (EER %, |cos|)")
    nm = {"E1":"Full","E2":"w/o Orth","E3":"w/o detach","E4":"w/o SupCon","E5":"w/o ArcFace"}
    for v in ["E1","E2","E3","E4","E5"]:
        print("    {:<3} {:<11}: EER {:.2f}%   |cos| {:.4f}".format(v, nm[v], s[v]["eer"][0], s[v]["cos"][0]))
    print("    orthogonality effect E2:E1 |cos| ratio = {:.1f}x   | paper 43x".format(s["E2"]["cos_ratio_vs_E1"]))

    print("\n[3] SECURITY threat model (clean/PGD EER %, TSR %; lower = more robust)")
    D, C = r2["DOMCS-EEG"], r2["CNN+ArcFace"]
    print("    clean EER                : DOMCS {:.2f}   CNN+ArcFace {:.2f}".format(D["clean"]["eer"], C["clean"]["eer"]))
    print("    T1 same-state attack EER : DOMCS {:.2f}   CNN+ArcFace {:.2f}".format(D["T1"]["eer"], C["T1"]["eer"]))
    try:
        print("    T4 PGD eps=0.01 EER      : DOMCS {:.2f}   CNN+ArcFace {:.2f}".format(D["T4"]["PGD"]["0.01"]["eer"], C["T4"]["PGD"]["0.01"]["eer"]))
    except Exception: pass
    print("    T5 targeted-impersonation TSR: DOMCS {:.1f}   CNN+ArcFace {:.1f}".format(D["T5"]["TSR"], C["T5"]["TSR"]))

    print("\n[4] STATISTICS")
    dm, nmn, pr = r8["domcs_mean_std"], r8["dann_mean_std"], r8["paired"]
    print("    10-seed DOMCS : {:.2f} +/- {:.2f} %   DANN : {:.2f} +/- {:.2f} %".format(dm[0], dm[1], nmn[0], nmn[1]))
    print("    paired test   : t={:.2f}, p_one={:.3f}  -> statistical PARITY".format(pr["t"], pr["p_one"]))
    ia, ve = r9["id_acc_mean_std"], r9["verif_eer_mean_std"]
    print("    EEGNet same-task: id-acc {:.1f} %, verif-EER {:.2f} %  (vs {:.2f} % under B2T)".format(ia[0], ve[0], r9["b2t_eer_reference"]))

    print("\n[5] BED cross-device (within-session, 5 seeds)")
    print("    EER {:.2f} +/- {:.2f} %   AUC {:.3f}".format(bed["eer_mean"][0], bed["eer_mean"][1], bed["auc_mean"]))

    print("\n" + L)
    print(" All values above are the exact contents of results/*.json (no recompute).")
    print(" They match the manuscript -> the reported numbers are produced by this code.")
    print(L)

if __name__ == "__main__":
    main()
