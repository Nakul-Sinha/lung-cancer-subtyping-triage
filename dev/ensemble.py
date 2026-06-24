"""Offline ensembling + decode-temper tuning over saved oof.npz artifacts.
Usage: python dev/ensemble.py kout_*/oof.npz   (globs multiple run artifacts)
Combines calibrated posteriors, tunes decode-temperature + referral frac on the
patient-OOF exact grader, reports per-run and ensemble OOF, writes best submission."""
import sys, glob
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import (CLASSES, A, harm_score, exact_grade, decode_belief,
                     referral_from_expected_harm)

def softmax(z, T=1.0):
    z = z/np.float64(T); z = z - z.max(1, keepdims=True); e = np.exp(z); return e/e.sum(1, keepdims=True)

def temper(q, Td):
    if Td == 1.0: return q
    qt = np.clip(q, 1e-9, 1)**(1.0/Td); return qt/qt.sum(1, keepdims=True)

def decode_batch_temper(Q, Td, w=None):
    Qt = temper(Q, Td); P = np.zeros_like(Qt); Eh = np.zeros(len(Qt))
    for i in range(len(Qt)):
        P[i], _ = decode_belief(Qt[i], w)
        Eh[i] = sum(Q[i, t]*harm_score(P[i], t) for t in range(7))  # expected harm under *real* q
    return P, Eh

def best_decode(Q, y, w_macro, td_grid=(1.0,1.25,1.5,2.0,3.0), fr_grid=(0,.05,.1,.15,.2,.25,.3)):
    best = (-1, None)
    for w,wn in [(None,"none"),(w_macro,"macroW")]:
        for Td in td_grid:
            P, Eh = decode_batch_temper(Q, Td, w)
            for fr in fr_grid:
                ref = referral_from_expected_harm(Eh, fr)
                s = exact_grade(P, ref, y)
                if s > best[0]: best = (s, dict(Td=Td, frac=fr, w=wn))
    return best

def main(paths):
    files = []
    for p in paths: files += glob.glob(p)
    files = sorted(set(files))
    assert files, "no oof.npz found"
    runs = [dict(np.load(f, allow_pickle=True)) for f in files]
    y = runs[0]["y"].astype(int); ids = runs[0]["ids"].astype(str)
    cnt = np.bincount(y, minlength=7); w_macro = (1.0/np.maximum(cnt,1)); w_macro/=w_macro.mean()

    print(f"{'run':40s} {'val_acc':>7} {'OOFbest':>8}")
    oof_list, test_list = [], []
    for f, r in zip(files, runs):
        T = float(r["T"]); oofQ = softmax(r["oof_logits"], T); teQ = softmax(r["test_logits"], T)
        oof_list.append(oofQ); test_list.append(teQ)
        acc = (oofQ.argmax(1)==y).mean(); s,_ = best_decode(oofQ, y, w_macro)
        print(f"{Path(f).parent.name:40s} {acc:7.3f} {s:8.4f}")

    # ensemble = mean of calibrated posteriors
    oofE = np.mean(oof_list,0); teE = np.mean(test_list,0)
    accE = (oofE.argmax(1)==y).mean(); sE, cfg = best_decode(oofE, y, w_macro)
    print(f"\n{'ENSEMBLE(mean)':40s} {accE:7.3f} {sE:8.4f}  cfg={cfg}")

    # write best ensemble submission
    w = w_macro if cfg["w"]=="macroW" else None
    P, Eh = decode_batch_temper(teE, cfg["Td"], w)
    ref = referral_from_expected_harm(Eh, cfg["frac"])
    sub = pd.DataFrame({"id": ids});
    for j,c in enumerate(CLASSES): sub["p_"+c]=P[:,j]
    sub["referral"]=ref
    out = Path("working"); out.mkdir(exist_ok=True); sub.to_csv(out/"submission.csv", index=False)
    print("wrote", out/"submission.csv", sub.shape, "| OOF est:", round(sE,4))

if __name__ == "__main__":
    main(sys.argv[1:] or ["dev/kout_*/oof.npz"])
