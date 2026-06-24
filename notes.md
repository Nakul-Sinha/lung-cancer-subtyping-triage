# Lung Cancer Subtyping & Grading — Notes

## Challenge facts (verified locally)
- **Dataset = LungHist700** (691 H&E lung patches total; here 472 train + 219 test).
- Images: all **1600×1200 RGB** JPEGs, magnification **20x or 40x** (same pixel size both mags).
- 7 classes (clinical hierarchy): `nor`, `aca_bd`, `aca_md`, `aca_pd`, `scc_bd`, `scc_md`, `scc_pd`.
  - Tier1 benign(nor) vs malignant; Tier2 subtype aca vs scc; Tier3 ordinal grade bd<md<pd.
- Train label counts: nor 115, aca_bd 83, aca_md 74, scc_pd 60, scc_md 51, scc_bd 48, aca_pd 41.
- **Only 30 train patients** (patient_id 2–45); test patients disjoint (~15). Patches/patient: 2–51 (median 10).
- 9/30 patients have >1 label; 7 have >1 superclass; 20/30 span both magnifications.
- Magnification near-balanced: train 241/231 (20x/40x), test 118/101.
- Data location (local): `eris/runnerloaddataset/public/` (misnamed folder). Cols: id,image,magnification,patient_id,superclass,subclass,label.

## Metric (FULLY SPECIFIED — this is the key lever)
Submit per test patch: 7 probs (sum→1, ±0.02 renorm) + `referral`∈[0,1].
- **Referral**: budget K=int(0.20·N)≈43; eligible only if referral>0.5; sorted desc; per-TRUE-class cap max(1,ceil(K/7))=7; referred patch scores flat **S_REF=0.40**.
- **Auto-reported patch** scores `harm = clip(hr − penalty, −2, 1)`:
  - `hr = Σ_k p_k·affinity(t,k)`; affinity: exact +1, same-subtype grade±1 +0.5, ±2 +0.2, wrong subtype 0, cross benign↔malignant −1.
  - `penalty`: if t malignant → `1.0·p_nor²` (miss); if t=nor → `0.5·(Σ malignant p)²` (false alarm).
- Macro-average over 7 TRUE classes, rescale [−2,1]→[0.02,1.0]. Higher better, max 1.0.
- Compute: A10G 24GB, 30 min train+infer. Train only on provided data; no external labeled lung histopath; no id/filename hardcoding; public reproducible pretrained weights OK.

## Strategic analysis (my own — to verify/expand via research)
**Core thesis: score gains come from (1) generalizable features on tiny data + (2) decision-theoretic decoding of the exact metric — NOT from overfitting.**

### Lever 1 — Bayes-optimal belief decoding (metric is non-proper)
Optimal submitted belief p ≠ model posterior q. Maximize `E_{t~q}[harm(p,t)]` on the simplex.
- E[hr] = Σ_k p_k a_k, with `a_k = Σ_t q_t·affinity(t,k)` (linear in p).
- E[penalty] = q_nor·0.5·(1−p_nor)² + (1−q_nor)·1.0·p_nor².
- Concave ⇒ unique max. Reward linear in malignant sub-mass ⇒ put all malignant mass on argmax_k a_k (malignant), then solve 1-D concave quadratic for p_nor vs malignant mass m. a_nor = 2q_nor−1.
- **Macro-aware refinement**: macro-average weights true class c by 1/n_c ⇒ reweight affinity/penalty by est. inverse test-class freq (from train prior). Pushes mass toward rare classes. Validate on macro CV.
- Requires **calibrated q** ⇒ calibration is important.

### Lever 2 — Referral = selective prediction
Refer patches where expected auto score < 0.40 (replace with flat 0.40). Rank by uplift = max(0, 0.40 − E[harm]), optionally × inverse class freq. Set referral>0.5 for top ~K, ≤0.5 otherwise. Per-true-class cap (7) can't be gamed → spread across predicted-hard classes. Tune count on CV.

### Lever 3 — Generalizable model on 472 patches / 30 patients
- Fine-tuning big nets overfits. Prefer **frozen pathology foundation-model embeddings + light regularized head** (logreg/MLP/prototype).
- 1600×1200 → encoder tiles (224). Multi-crop mean-pool / MIL aggregation preserves nuclear/texture detail (needed for grade).
- Magnification covariate: feed mag to head and/or mag-aware tiling; TTA.
- Stain robustness: stain augmentation (RandStainNA/HED jitter) or Macenko norm for cross-patient generalization.
- Ensemble diverse encoders + multi-crop TTA (cheap, frozen ⇒ fits A10G/30min).

### Validation
GroupKFold by patient_id (repeated, since only 30 patients ⇒ high variance). Score folds with the EXACT metric incl. referral simulation. Tune only decision hyperparams + head regularization on CV; never on public LB.

## Research RESOLVED (deep-research, 23 claims verified 3-0; synthesis step failed, synthesized manually)
A. **Foundation models — GATING is decisive:**
   - ❌ GATED (avoid, violate compliance): UNI, Virchow, Virchow2, H-optimus-0, Hibou-L, Prov-GigaPath (institutional-email approval / HF login).
   - ✅ NON-GATED (safe, reproducible offline): **Phikon-v2** (owkin, ViT-L/1024, 224@20x, non-commercial lic) = PRIMARY; **Lunit-DINO** (torch.hub from GitHub releases, no auth; ViT-S/8 & /16 = 384-d; mirror 1aurent/*) = ensemble; Phikon-v1 (768). CTransPath non-gated but GDrive download ⇒ pre-stage only.
   - De-risk: encoder perf does NOT scale with model size; Phikon-v2 0.874 ≈ UNI 0.873 ≈ GigaPath 0.883 AUC ⇒ avoiding gated models costs ~0.01 AUC.
   - License caveat: Phikon-v2/Lunit are non-commercial-research (non-gated ⇒ satisfies the gating rule; flag for platform). Permissive fallback = ImageNet ConvNeXt/ViT.
B. LungHist700: **45 patients**, 691 imgs, 1200×1600, Valladolid 2023 (= 30 train + ~15 test). Grade is hardest axis. Methodology only — never use its labels (test patches are a subset ⇒ leakage).
C. Small-data: frozen FM + light head >> full fine-tune on <500 imgs; stain AUGMENT (RandStainNA/HED) for cross-patient > fixed norm alone; multi-crop mean-pool tiles for 1600×1200; mag as head covariate + scale-normalized 40x view + TTA.
D. Ordinal/hier heads: NOT worth it — metric already gives partial credit and decode exploits hierarchy; flat-7 calibrated + decode is lower-variance. Optional ablation only.
E. Calibration: single global **temperature** (Guo 2017) on OOF logits — most data-efficient; avoid Dirichlet/vector scaling on <500. Regularized logreg ~calibrated.
F. Decision theory: confirmed — plug-in Bayes-optimal under estimated posterior + threshold-at-budget for referral (selective classification / reject option, Geifman-ElYaniv, SelectiveNet). Matches my verified derivation.

## DELIVERABLE: Approach.md written (comprehensive, citation-backed, verified numbers). notes.md = working log.

## IMPLEMENTATION LOG
### Env
- Local machine is **CPU-only** (torch 2.6+cu124 but no CUDA). GPU work → Kaggle T4 (headless, `dev/krun.py`) or Lightning A100 (SSH, 3h cap, reserve for sweeps).
- Kaggle auth OK (user `nakuls1nha`). Git remote: github.com/Nakul-Sinha/Lung-Cancer-Subtyping-Grading (main).
- Backbone decision: **ImageNet `convnext_tiny.fb_in22k_ft_in1k`** for official deliverable (zero DQ risk per guide); pathology-SSL only as side experiment.

### Files
- `dev/harness.py` — exact grader + decoder + referral + StratifiedGroupKFold (verified; sample-sub anchor=0.6776).
- `dev/train.py` — self-contained pipeline (→ becomes solution.py). Backbone+mag covariate, HED+geom aug, 5-fold SGKF, EMA, temp calibration, decode+referral, fold-ensemble test. Env-configurable. SMOKE test passed on CPU.
- `validate_submission.py` — strict schema/range/sum checks.
- `dev/krun.py` — Kaggle dataset upload + script-kernel push/status/fetch.

### Iteration 1 (in progress)
- Dataset uploading to Kaggle (~405MB). Next: push T4 kernel, fetch OOF score + submission, commit/push.
- Anchors to beat (local, exact grader): sample-sub (uniform+ref=1) **0.6776**; decoded-prior 0.6967.

## 🚨 BLOCKER — LOCAL IMAGE DATA IS CORRUPTED (discovered iter-1)
- `images/` has **691 files but only 134 UNIQUE contents** (md5). Most-replicated image appears 14×.
- Image content is **decoupled from labels**: each unique image maps to **2.78 distinct labels on avg (max 6)**; only 31% of unique images have a single consistent label. 92/93 dup-groups span multiple labels AND patients AND magnifications.
- All 91 unique TEST images are duplicates of TRAIN images (train/test content overlap=91).
- ⇒ The vision task is **unlearnable on this copy** (same pixels → up to 6 different labels). This explains all random results: v1 finetune acc 0.025, frozen ImageNet acc 0.11, benign/malignant AUC 0.186 (anti-correlated), logreg train-fit acc 0.30.
- Real LungHist700 = 691 UNIQUE images. So ~557 unique images are MISSING from this local export, replaced by duplicates. **Need correct dataset.**
- Cannot fix via figshare LungHist700 (= banned external-label leakage on the test set).
- All infra (exact grader, StratifiedGroupKFold, Bayes decoder, referral, Kaggle pipeline) is built & verified — only correct pixels are missing.

## ITERATION 1 (committed) — robust self-calibrating solution.py
- Data still corrupted (user re-provided same data at `eris/lung-cancer-dataset/`, verified identical: 134 unique/691). Building robust solution that works either way.
- `solution.py`: mag-conditioned prior (always) + vision fine-tune (GPU only) + blends → self-selects best on patient-grouped OOF exact grader → Bayes decode + referral. CPU path & GPU vision path both smoke-verified.
- **OOF results (patient-grouped, exact grader): single-seed 0.708; HONEST repeated 8×5-CV = 0.692 ± 0.011** (mag_prior+decode+referral, robust config exp=1.0 frac=0.4); flat prior ~0.682; sample-sub anchor 0.6776; perfect ceiling 1.0. Magnification is the only legit signal ⇒ ~0.692 is the Bayes ceiling on corrupted data (user chose to squeeze it; `dev/squeeze.py`).
- Submission: 219 rows valid (66 flagged @ frac 0.3, grader caps at K=43).
- Pipeline fixes applied: EMA warmup (fixes random-head leak), discriminative LR (bb 0.3×), backbone-freeze warmup 3ep — but unmeasurable on corrupted data; will re-evaluate on intact data.
- On intact platform data, solution.py auto-selects vision if it beats prior on OOF → captures headroom (0.707→1.0).

## PUBLIC SCORE 0.6884 — CEILING PROVEN
- Public 0.6884 ≈ OOF 0.6926 (gap 0.004, within CV std 0.01) ⇒ submission generalizes, NOT overfit. Good for private.
- User confirmed PLATFORM data is also corrupted ⇒ this is a pure decision-theory challenge for everyone; "barely above AI baseline" is structural (all capped by corruption).
- **Ceiling proof (patient-grouped CV, 6 seeds, exact grader):** P(label|magnification)=0.6926; P(label|image content)=0.6704 (WORSE); P(label|image+mag)=0.6731 (WORSE). ⇒ image pixels carry NO transferable signal (same image → independently random labels per occurrence; this is why the CNN went below-random/anti-correlated at 0.025).
- Conclusion: magnification is the ONLY transferable signal; decode is Bayes-optimal; macro-weight is exactly macro-optimal. **0.688–0.693 is the provable ceiling.** No legitimate lever remains. Higher would require intact data (unavailable — platform corrupted) or prohibited image/id fingerprinting (which also fails on private: anti-generalizes). Current submission is already the private-optimal choice.

## FINALIZED (shared-pool) — solution.py v2
- Org confirmed dataset stays as-is (duplicated, one pool); others score ~0.72.
- KEY: same magnification approach scores **0.6915 under patient-GROUPED CV but 0.7221 under shared-pool (random) CV**. The 0.72 is just the shared-pool regime. My earlier 0.6884 used grouped-CV framing (too pessimistic for a shared pool).
- Exhaustive proof magnification is optimal: image-content lookup 0.695, blends ≤0.722, id-hash random — nothing beats magnification under shared-pool CV.
- Finalized solution.py: magnification-conditioned prior (alpha=0.3) + Bayes decode (macro-exp=1.0) + cap-aware referral (frac=0.4, plateau ≥0.4) tuned on **shared-pool StratifiedKFold** (10 seeds). Vision path removed (images proven uninformative). Comment-free, deterministic, CPU-fast.
- **Shared-pool OOF = 0.7221.** Submission: 219 rows valid, 88 flagged (grader refers K=43).
- "Perfect bias point" = magnification class-distribution (stable across every pool sample → lifts public AND private equally, no overfit) + referral frac=0.4. Already at it.
- Next: user resubmits this file. If →~0.72 the test is shared-pool (holds on private, same pool); if stays ~0.69 the test holds out patients and 0.69 is honest (others overfitting public).

### Open risks / TODO
- **Vision GPU path verified only by CPU smoke** — should confirm on Kaggle T4 once (no crash), and on intact data measure real vision OOF.
- Get intact images (export tool is the likely culprit — both copies identical-corrupted).
- **Offline weights for official A10G run**: if Eris runtime has NO internet, timm can't download convnext weights → must stage weights as packaged asset. Confirm whether Eris A10G has internet. (Dev on Kaggle uses internet.)
- Official 30-min A10G budget: measure 5-fold×28ep timing on T4 (A10G ~1.5-2× faster); reduce folds/epochs if needed.
- Decode currently emits ~one-hot beliefs (theoretical optimum); test concentrate-vs-tempered on real OOF (robustness under imperfect calibration).
- Train/test view mismatch (train RandomResizedCrop vs test full-resize) — standard, revisit if OOF poor.

## VERIFIED decision-theory results (own simulation vs exact grader)
- **Closed-form Bayes-optimal belief decoder is provably exact**: gap vs brute force = **0.0** over 150 posteriors × 450k candidates. Formula: put all malignant mass on argmax_k a_k (k≠nor), where a=Aᵀq; solve 1-D concave quadratic for malignant mass m: `m = (a_nor − a_mal − 2(1−q_nor)) / (q_nor − 2)`, clip [0,1]; a_nor = 2q_nor−1.
- **Uplift of optimal decode vs submitting raw posterior q ≈ +0.053 LB** (vs argmax one-hot +0.028), avg over random posteriors.
- **Emergent metric-aware behaviors** (verified): grade-unsure within a subtype → bet the **middle grade** (md, never off-by-2); subtype tossup → pick one (cross-subtype reward=0 either way), E[harm]→0.40 (referral margin); benign/malignant unsure → split nor vs best malignant, low E[harm] → referral candidate. Ties belief & referral layers together.
- **Referral is calibration-contingent**: with MIScalibrated posteriors, escalation HURTS (−0.01 to −0.03). With CALIBRATED posteriors (true label sampled from q), targeted referral of lowest-E[harm] patches gives **+0.017–0.018 LB**, optimum spending most of K≈43 budget. ⇒ Naive `referral=1.0` for all (the sample_submission!) is actively harmful. Refer only bottom-K by expected harm, and ONLY if calibrated; tune count on exact-grader patient-CV.
- **Calibration is load-bearing** — both levers depend on it. Invest in temperature/Dirichlet calibration; validate ECE on grouped CV.
- Combined decision-layer uplift over naive softmax+random-referral baseline ≈ **+0.05–0.07 LB**, fully overfitting-proof (optimizes the known metric given calibrated q).

## Compliance reminders
- Do NOT download/use LungHist700 labels (test patches are a subset → leakage). Methodology only.
- No gated/credentialed/API models in the official pipeline. Verify each backbone is freely downloadable & reproducible.
- Train only on provided public data.
