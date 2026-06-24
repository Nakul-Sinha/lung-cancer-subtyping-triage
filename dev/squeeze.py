"""Squeeze the legitimate ceiling on the (corrupted) data: tune prior smoothing,
macro-weight exponent, and referral fraction on REPEATED patient-grouped CV with
the exact grader. Reports robust top configs (mean over seeds). No leakage."""
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import CLASSES, CIDX, exact_grade, decode_belief, harm_score, referral_from_expected_harm
from sklearn.model_selection import StratifiedGroupKFold

D = Path("G:/Datacurve/eris/lung-cancer-dataset")
tr = pd.read_csv(D/"train.csv"); y = tr["label"].map(CIDX).values; groups = tr["patient_id"].values
mag = (tr["magnification"].astype(str).str.startswith("40")).astype(int).values
cnt_all = np.bincount(y, minlength=7)
SEEDS = [0,1,2,3,4,5,6,7]; NS = 5

def macro_w(cnt, exp):
    w = (1.0/np.maximum(cnt,1))**exp; return w/w.mean()

def oof_score(alpha, exp, frac):
    scores=[]
    for sd in SEEDS:
        sgkf=StratifiedGroupKFold(NS, shuffle=True, random_state=sd)
        oof=np.zeros((len(y),7))
        for tri,vai in sgkf.split(np.zeros(len(y)),y,groups):
            for mg in [0,1]:
                s=tri[mag[tri]==mg]; pr=np.bincount(y[s],minlength=7)+alpha; pr=pr/pr.sum()
                oof[vai[mag[vai]==mg]]=pr
        w=macro_w(cnt_all,exp)
        P=np.zeros_like(oof); Eh=np.zeros(len(y))
        for i in range(len(y)): P[i],_=decode_belief(oof[i],w); Eh[i]=sum(oof[i,t]*harm_score(P[i],t) for t in range(7))
        scores.append(exact_grade(P, referral_from_expected_harm(Eh,frac), y))
    return float(np.mean(scores)), float(np.std(scores))

results=[]
for alpha in [0.3,1.0]:
    for exp in [0.0,0.5,1.0,1.5,2.0,2.5]:
        for frac in [0,.1,.15,.2,.25,.3,.35,.4]:
            m,s=oof_score(alpha,exp,frac); results.append((m,s,alpha,exp,frac))
results.sort(reverse=True)
print(f"{'mean':>7} {'std':>6}  alpha  exp  frac")
for m,s,a,e,f in results[:12]:
    print(f"{m:7.4f} {s:6.4f}  {a:<5} {e:<4} {f}")
print("\nbaseline (current solution.py: alpha=0.5 exp=1.0 frac=0.3):")
m,s=oof_score(0.5,1.0,0.3); print(f"  {m:.4f} +/- {s:.4f}")
best=results[0]; print(f"\nBEST robust config: alpha={best[2]} exp={best[3]} frac={best[4]} -> {best[0]:.4f} +/- {best[1]:.4f}")
