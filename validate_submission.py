"""Strict validator for the Lung Cancer Subtyping & Grading submission."""
import sys
import numpy as np, pandas as pd
from pathlib import Path

PROB_COLS = ["p_nor","p_aca_bd","p_aca_md","p_aca_pd","p_scc_bd","p_scc_md","p_scc_pd"]
COLS = ["id"] + PROB_COLS + ["referral"]

def validate(sub_path="working/submission.csv", sample_path=None, data_dir=None):
    sub = pd.read_csv(sub_path)
    assert set(COLS) == set(sub.columns), f"columns mismatch: {sub.columns.tolist()}"
    # test ids
    if data_dir is None:
        for c in [Path("dataset/public"), Path("G:/ml/data/runnerloaddataset/public")]:
            if (c/"test.csv").exists(): data_dir=c; break
    te = pd.read_csv(Path(data_dir)/"test.csv")
    assert len(sub)==len(te), f"row count {len(sub)} != test {len(te)}"
    assert sub["id"].is_unique, "duplicate ids"
    assert set(sub["id"])==set(te["id"]), "id set mismatch with test.csv"
    P = sub[PROB_COLS].to_numpy(float)
    assert np.isfinite(P).all(), "NaN/Inf in probabilities"
    assert (P>=-1e-9).all() and (P<=1+1e-9).all(), "probabilities out of [0,1]"
    s = P.sum(1)
    assert np.allclose(s,1.0,atol=0.02), f"prob rows not summing to 1 (±0.02): min {s.min():.4f} max {s.max():.4f}"
    r = sub["referral"].to_numpy(float)
    assert np.isfinite(r).all() and (r>=-1e-9).all() and (r<=1+1e-9).all(), "referral out of [0,1] or NaN"
    n_ref_flag = int((r>0.5).sum())
    print(f"OK  rows={len(sub)}  prob-sum[min,max]=[{s.min():.4f},{s.max():.4f}]  "
          f"referral>0.5: {n_ref_flag}  ({100*n_ref_flag/len(sub):.0f}% flagged; budget~{int(0.2*len(sub))})")
    return True

if __name__=="__main__":
    p = sys.argv[1] if len(sys.argv)>1 else "working/submission.csv"
    validate(p)
