"""Distribution-shift stress test: the public score (0.668) < OOF (0.692) means the
submission is overfit to the train label distribution. Here we evaluate each
(macro-weight exp, referral frac, belief-temper lambda) config against MANY plausible
test label distributions and pick the one with the best worst-case / mean robust score.
Belief is constant-per-magnification (images uninformative), so this is exact."""
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import CLASSES, CIDX, A, exact_grade, decode_belief, harm_score, referral_from_expected_harm

D = Path("G:/Datacurve/eris/lung-cancer-dataset")
tr = pd.read_csv(D/"train.csv"); te = pd.read_csv(D/"test.csv")
y = tr["label"].map(CIDX).values
magtr = (tr["magnification"].astype(str).str.startswith("40")).astype(int).values
magte = (te["magnification"].astype(str).str.startswith("40")).astype(int).values
cnt = np.bincount(y, minlength=7)
rng = np.random.default_rng(0)

def prior_per_mag(alpha=1.0):
    out={}
    for mg in [0,1]:
        c=np.bincount(y[magtr==mg],minlength=7)+alpha; out[mg]=c/c.sum()
    return out
PM = prior_per_mag()

def belief_for_mag(mg, exp, lam):
    """decode the per-mag prior with macro-weight exp, then blend toward the prior by lam."""
    w = None if exp==0 else (lambda v: v/v.mean())((1.0/np.maximum(cnt,1))**exp)
    b_dec,_ = decode_belief(PM[mg], w)
    b = (1-lam)*b_dec + lam*PM[mg]
    return b/b.sum()

# candidate test label distributions (per magnification) to stress-test against
def test_dists():
    base = {mg: PM[mg] for mg in [0,1]}
    dists = {"train_prior": base,
             "uniform": {mg: np.ones(7)/7 for mg in [0,1]},
             "marginal": {mg: (np.bincount(y,minlength=7)/len(y)) for mg in [0,1]}}
    # class-heavy shifts (double one class)
    for c in range(7):
        d={}
        for mg in [0,1]:
            p=PM[mg].copy(); p[c]*=3; p/=p.sum(); d[mg]=p
        dists[f"heavy_{CLASSES[c]}"]=d
    # benign vs malignant skews
    for name,boost in [("benign_heavy",[3,1,1,1,1,1,1]),("malig_heavy",[1,2,2,2,2,2,2])]:
        d={}
        for mg in [0,1]:
            p=PM[mg]*np.array(boost,float); p/=p.sum(); d[mg]=p
        dists[name]=d
    return dists

def eval_config(exp, frac, lam, dists, n_sim=400, n_test=219):
    beliefs={mg:belief_for_mag(mg,exp,lam) for mg in [0,1]}
    Eh={mg:sum(PM[mg][t]*harm_score(beliefs[mg],t) for t in range(7)) for mg in [0,1]}
    scores={}
    for dname,d in dists.items():
        ss=[]
        for _ in range(n_sim):
            # sample magnifications like the real test, then labels from d|mag
            mg=magte.copy() if len(magte)==n_test else rng.integers(0,2,n_test)
            yy=np.array([rng.choice(7,p=d[m]) for m in mg])
            P=np.array([beliefs[m] for m in mg]); E=np.array([Eh[m] for m in mg])
            ref=referral_from_expected_harm(E,frac)
            ss.append(exact_grade(P,ref,yy))
        scores[dname]=float(np.mean(ss))
    return scores

dists=test_dists()
configs=[(e,f,l) for e in [0.5,1.0,1.5] for f in [0,0.1,0.2,0.3,0.4] for l in [0,0.25,0.5,0.75,1.0]]
rows=[]
for e,f,l in configs:
    sc=eval_config(e,f,l,dists)
    vals=np.array(list(sc.values()))
    rows.append((vals.min(),vals.mean(),e,f,l,sc))
# rank by worst-case (robust), tiebreak by mean
rows.sort(key=lambda r:(r[0],r[1]),reverse=True)
print(f"{'worst':>7}{'mean':>8}  exp  frac  lam")
for wc,mn,e,f,l,_ in rows[:10]:
    print(f"{wc:7.4f}{mn:8.4f}  {e:<4}{f:<5} {l}")
print("\ncurrent submission config (exp=1.0, frac=0.4, lam=0):")
sc=eval_config(1.0,0.4,0.0,dists); v=np.array(list(sc.values()))
print(f"  worst={v.min():.4f} mean={v.mean():.4f}")
print("  per-distribution:", {k:round(x,3) for k,x in sc.items()})
best=rows[0]
print(f"\nROBUST PICK: exp={best[2]} frac={best[3]} lam={best[4]}  worst={best[0]:.4f} mean={best[1]:.4f}")
print("  per-distribution:", {k:round(x,3) for k,x in best[5].items()})
