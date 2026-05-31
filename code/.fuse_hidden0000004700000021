"""
setup_paper_figures.py
======================
Copies all publication-ready figures from the project folder
into the LaTeX 'figures/' subfolder so \includegraphics works.

Run once before compiling the paper:
    python setup_paper_figures.py
"""

import os
import shutil

# Source: paper_ready/ folder in the project
SRC = r"C:\Users\L.KANIMOZHI\OneDrive\27may26 3.00pm\DOMCS_EEG_COMPLETE\DOMCS_EEG_FINAL_LOCKED_TIFS_VERSION\paper_ready"

# Destination: figures/ subfolder next to this script
DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(DST, exist_ok=True)

figures = [
    "FIG_1_DET_curve.pdf",
    "FIG_2_ablation_bar.pdf",
    "FIG_3_eer_vs_noise.pdf",
    "FIG_4_tsne_disentangle.pdf",
    "FIG_5_security_bar.pdf",
]

print("Copying figures to:", DST)
for fig in figures:
    src_path = os.path.join(SRC, fig)
    dst_path = os.path.join(DST, fig)
    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
        print("  OK:", fig)
    else:
        print("  MISSING:", fig, "(run generate_tifs_paper_figures.py on A100 first)")

print("\nDone. Now compile with:")
print("  pdflatex  main.tex")
print("  pdflatex  main.tex   (second pass for cross-refs)")
