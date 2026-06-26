"""
CPU spine for the Lung Cancer Subtyping & Grading challenge.

Contains (all verified against the published grader):
  - exact grader (referral allocation -> harm -> macro -> rescale [0.02,1])
  - Bayes-optimal belief decoder (closed-form, proven gap=0 vs brute force)
  - expected-harm referral policy
  - StratifiedGroupKFold OOF scoring helper
  - baseline anchors (uniform, prior, oracle posterior) for sanity

Run:  python dev/harness.py
"""
import numpy as np
import pandas as pd
from math import ceil
from pathlib import Path

CLASSES = ["nor", "aca_bd", "aca_md", "aca_pd", "scc_bd", "scc_md", "scc_pd"]
CIDX = {c: i for i, c in enumerate(CLASSES)}
MALIG = np.array([0, 1, 1, 1, 1, 1, 1], dtype=bool)

GAMMA_MISS, GAMMA_FA = 1.0, 0.5
S_REF = 0.40
REFERRAL_BUDGET = 0.20
REFERRAL_THRESHOLD = 0.5
HARM_FLOOR, HARM_CEIL = -2.0, 1.0
SCORE_FLOOR = 0.02

DATA = Path("G:/ml/data/runnerloaddataset/public")


# ----------------------------- affinity / harm -----------------------------
def _subtype(c): return c.split("_")[0]
def _grank(c): return {"bd": 1, "md": 2, "pd": 3}.get(c.split("_")[1], 0) if "_" in c else 0

def _affinity(t, k):
    if k == t: return 1.0
    if t == "nor" or k == "nor": return -1.0
    if _subtype(t) != _subtype(k): return 0.0
    return {1: 0.5, 2: 0.2}[abs(_grank(t) - _grank(k))]

A = np.array([[_affinity(t, k) for k in CLASSES] for t in CLASSES])  # A[t,k]


def harm_score(probs, t_idx):
    """Exact per-patch auto-report harm (clipped)."""
    p = np.asarray(probs, float)
    hr = float(p @ A[t_idx])
    if t_idx != CIDX["nor"]:
        pen = GAMMA_MISS * p[CIDX["nor"]] ** 2
    else:
        pen = GAMMA_FA * (p[MALIG].sum()) ** 2
    return max(HARM_FLOOR, min(HARM_CEIL, hr - pen))


def exact_grade(probs, referral, gt_idx):
    """Faithful re-implementation of the published grade() function.
    probs: (N,7) ; referral: (N,) ; gt_idx: (N,) int labels. Returns rescaled score."""
    probs = np.asarray(probs, float); referral = np.asarray(referral, float)
    gt_idx = np.asarray(gt_idx, int)
    N = len(gt_idx)
    K = int(REFERRAL_BUDGET * N)
    max_per = max(1, ceil(K / len(CLASSES)))
    eligible = [i for i in range(N) if referral[i] > REFERRAL_THRESHOLD]
    eligible.sort(key=lambda i: (-referral[i], i))
    referred = set(); cc = {c: 0 for c in range(len(CLASSES))}
    for i in eligible:
        if len(referred) >= K: break
        t = gt_idx[i]
        if cc[t] < max_per:
            referred.add(i); cc[t] += 1
    by = {c: [] for c in range(len(CLASSES))}
    for i in range(N):
        s = S_REF if i in referred else harm_score(probs[i], gt_idx[i])
        by[gt_idx[i]].append(s)
    class_means = [np.mean(v) for v in by.values() if v]
    macro = float(np.mean(class_means))
    score = SCORE_FLOOR + (macro - HARM_FLOOR) / (HARM_CEIL - HARM_FLOOR) * (1.0 - SCORE_FLOOR)
    return max(SCORE_FLOOR, min(1.0, score))


# ----------------------------- belief decoder -----------------------------
def decode_belief(q, class_weight=None):
    """Closed-form Bayes-optimal belief that maximizes E_{t~q}[harm(p,t)].
    q: (7,) calibrated posterior. class_weight: optional (7,) macro inverse-freq weights
    applied to each true-class term (q_t -> q_t*w_t, renormalized). Returns (p(7,), E_harm)."""
    q = np.asarray(q, float)
    w = np.ones(7) if class_weight is None else np.asarray(class_weight, float)
    qw = q * w
    qw = qw / qw.sum()
    a = A.T @ qw                       # a_k = sum_t qw_t affinity(t,k)
    qn = qw[CIDX["nor"]]
    a_nor = a[CIDX["nor"]]             # == 2*qn - 1
    jmal = 1 + int(np.argmax(a[1:]))
    a_mal = a[jmal]
    denom = (qn - 2.0)
    m = (a_nor - a_mal - 2.0 * (1.0 - qn)) / denom
    m = min(1.0, max(0.0, m))
    p = np.zeros(7); p[CIDX["nor"]] = 1.0 - m; p[jmal] = m
    # expected harm under the *true* (unweighted) posterior q
    Eh = sum(q[t] * harm_score(p, t) for t in range(7))
    return p, Eh


def decode_batch(Q, class_weight=None):
    P = np.zeros_like(Q, float); Eh = np.zeros(len(Q))
    for i in range(len(Q)):
        P[i], Eh[i] = decode_belief(Q[i], class_weight)
    return P, Eh


def referral_from_expected_harm(Eh, frac=0.25):
    """Flag worst `frac` by expected harm for escalation (referral>0.5), ranked desc by neediness."""
    Eh = np.asarray(Eh, float); N = len(Eh)
    upl = S_REF - Eh                          # >0 => escalation expected to help
    order = np.argsort(-upl)                  # neediest first
    k = int(round(frac * N))
    ref = np.zeros(N)
    sel = [i for i in order[:k] if upl[i] > 0]
    for r, i in enumerate(sel):
        ref[i] = 1.0 - 0.45 * (r / max(1, len(sel) - 1)) if len(sel) > 1 else 0.9
    return ref


# ----------------------------- CV helper -----------------------------
def stratified_group_oof_indices(labels, groups, n_splits=5, seed=0):
    from sklearn.model_selection import StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = [None] * len(labels)
    for f, (_, va) in enumerate(sgkf.split(np.zeros(len(labels)), labels, groups)):
        for i in va: folds[i] = f
    return np.array(folds)


def score_posteriors_oof(Q, gt_idx, frac=0.25, class_weight=None):
    """Decode + referral on a full set of OOF posteriors, score with exact grader."""
    P, Eh = decode_batch(Q, class_weight)
    ref = referral_from_expected_harm(Eh, frac)
    return exact_grade(P, ref, gt_idx), P, ref


# ----------------------------- baselines / self-test -----------------------------
def main():
    tr = pd.read_csv(DATA / "train.csv")
    y = tr["label"].map(CIDX).values
    groups = tr["patient_id"].values
    N = len(y)
    prior = np.bincount(y, minlength=7) / N
    print(f"train N={N}  classes={dict(zip(CLASSES, np.bincount(y, minlength=7)))}")
    print(f"prior={np.round(prior,3)}")

    rng = np.random.default_rng(0)

    # Anchor 1: uniform belief + sample-style referral=1 everywhere (the sample submission)
    Pu = np.full((N, 7), 1 / 7)
    s_unif_ref1 = exact_grade(Pu, np.ones(N), y)
    s_unif_ref0 = exact_grade(Pu, np.zeros(N), y)
    print(f"\n[anchor] uniform 1/7, referral=1 all : {s_unif_ref1:.4f}")
    print(f"[anchor] uniform 1/7, referral=0 all : {s_unif_ref0:.4f}")

    # Anchor 2: predict prior (constant) for everyone, decoded vs raw
    Qprior = np.tile(prior, (N, 1))
    s_prior_raw = exact_grade(Qprior, np.zeros(N), y)
    s_prior_dec, _, _ = score_posteriors_oof(Qprior, y, frac=0.0)
    print(f"\n[anchor] constant prior, raw submit    : {s_prior_raw:.4f}")
    print(f"[anchor] constant prior, decoded        : {s_prior_dec:.4f}")

    # Anchor 3: 'noisy oracle' posterior at a few accuracy levels (sanity that grader+decode behave)
    print("\n[sanity] noisy-oracle posteriors (calibrated): decoded+referral vs raw-softmax")
    for conc in [2.0, 5.0, 12.0]:
        # calibrated: draw q from Dirichlet, sample label FROM q
        Q = rng.dirichlet(np.ones(7) * 1.0, size=N)  # placeholder; replaced below per-row
        # build calibrated: start from true one-hot smoothed by conc, then it's a proxy
        Q = np.zeros((N, 7))
        for i in range(N):
            alpha = np.ones(7) + conc * np.eye(7)[y[i]]
            Q[i] = rng.dirichlet(alpha)
        acc = (Q.argmax(1) == y).mean()
        s_raw = exact_grade(Q, np.zeros(N), y)
        s_dec, _, ref = score_posteriors_oof(Q, y, frac=0.25)
        s_dec_noref, _, _ = score_posteriors_oof(Q, y, frac=0.0)
        print(f"  conc={conc:>4} acc~{acc:.2f} | raw {s_raw:.4f} | decoded(noref) {s_dec_noref:.4f} "
              f"| decoded+ref {s_dec:.4f}  (referred {int((ref>0.5).sum())})")

    # CV fold sanity
    folds = stratified_group_oof_indices(y, groups, 5, 0)
    print(f"\n[cv] StratifiedGroupKFold fold sizes: {np.bincount(folds)}  "
          f"(patients/fold ~{tr.groupby(folds)['patient_id'].nunique().tolist()})")
    # verify no patient spans folds
    leak = tr.groupby('patient_id').apply(lambda d: len(np.unique(folds[d.index]))).max()
    print(f"[cv] max folds any patient appears in (must be 1): {leak}")


if __name__ == "__main__":
    main()
