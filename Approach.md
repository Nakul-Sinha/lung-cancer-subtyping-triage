# Approach — Lung Cancer Subtyping & Grading (LungHist700)

## 0. TL;DR / headline strategy

The points in this challenge do **not** come from a bigger network — they come from two things a naive solver leaves on the table:

1. **A representation that generalizes across unseen patients** on ~472 images / **30 patients**. Train-from-scratch or fine-tuning a large net on this little data memorizes patients and collapses on the disjoint test patients. Instead: **frozen, publicly-downloadable pathology foundation-model embeddings + a small, heavily-regularized head**, validated with **patient-grouped CV**.
2. **A decision layer that optimizes the *exact, fully-specified* scoring metric.** The metric is published in full and is **not a proper scoring rule**, so the score-maximizing belief distribution is *not* the model's posterior. Converting a calibrated posterior into the Bayes-optimal belief, plus a budget-aware referral policy, is worth **≈ +0.05–0.07 leaderboard** on its own — and it is **mathematically incapable of overfitting** because it optimizes a *known* function, not the data.

Everything below is engineered so that **every gain transfers to the private test set**: generalizable features, decision theory against the published metric, and honest patient-stratified validation. No leaderboard probing, no metadata shortcuts, no gated models.

**Verified up front (own simulation against the exact grader code):** the closed-form belief decoder matches brute-force optimization with **gap = 0.0**; it gains **+0.053** leaderboard vs. submitting the raw posterior; targeted referral on calibrated posteriors gains **+0.017–0.018**; and naive `referral = 1.0` for all patches (as in `sample_submission.csv`) *loses* score.

---

## 0b. Implementation status, data-integrity finding & results (iteration 1)

**⚠️ Critical data finding.** The provided `images/` folder contains **691 files but only 134 unique images** (verified by content hash). The image content is **decoupled from the labels**: a byte-identical image appears under multiple hash-token filenames with *contradictory* labels (e.g. one identical file labeled `nor`, `aca_bd`, `scc_md`, `scc_bd` across different rows). Each unique image carries **2.78 distinct labels on average (max 6)**; all 91 unique test images duplicate train images. The real LungHist700 has 691 *unique* images, so ~557 are missing from this export. **Consequence: no vision model can exceed chance on this copy** (confirmed empirically — fine-tune acc 0.025, frozen-feature acc 0.11, benign/malignant AUC 0.186). This is almost certainly an export/download corruption, not the intended task. Fixing it by pulling LungHist700 from figshare is disallowed (external test-overlapping labels = leakage). See `notes.md` and `dev/verify_data.py`.

**Robust response — a self-calibrating solution (`solution.py`).** Rather than depend on image quality, the pipeline:
1. Always builds a **magnification-conditioned class prior** (a legitimate, test-available covariate — not a fingerprint).
2. If a GPU is present, also fine-tunes the ImageNet backbone (with the magnification covariate, EMA-warmup, discriminative LR, backbone-freeze warmup).
3. On **patient-grouped OOF**, scores every candidate posterior (prior / vision / blends) with the **exact competition grader** and keeps whichever wins.
4. Applies the verified Bayes-optimal belief decode + budget-aware expected-harm referral.

So if the platform's images carry signal, the vision model is selected automatically and the score climbs toward the headroom below; if they don't (as in this corrupted copy), it floors on the magnification prior — never below a strong post-processing baseline.

**Verified results (patient-grouped OOF, exact grader):**
| Submission | OOF score |
|---|---|
| Sample submission (uniform + referral=1) — anchor | 0.6776 |
| Flat prior + decode + referral | 0.687 |
| **Magnification-prior + decode + referral (selected on corrupted data)** | **0.7070** |
| Perfect classifier ceiling (true-label one-hot) | 1.0000 |

Iteration 1 ships the robust solution at **0.7070** (beats the anchor, fully legitimate, no overfitting). The 0.707→1.0 gap is the headroom that the vision path will capture **automatically once intact images are available** — the architecture, decision layer, validation harness, and Kaggle/A10G pipeline are all built and verified; only correct pixels are missing.

---

## 1. Problem & data understanding

**Dataset = LungHist700** (verified: 691 images, 1200×1600 px, **45 patients**, Hospital Clínico de Valladolid 2023). [PMC11455975](https://pmc.ncbi.nlm.nih.gov/articles/PMC11455975/), [Sci Data 2024](https://www.nature.com/articles/s41597-024-03944-3). Locally: **472 train / 219 test**, all **1600×1200 RGB** JPEG, magnification **20x or 40x**.

**Targets — 7 classes in a clinical hierarchy:**
```
                 patch
        ┌──────────┴───────────┐
    benign (nor)            malignant
                       ┌────────┴────────┐
                  ACA (aca)          SCC (scc)
                  bd < md < pd       bd < md < pd   ← ordinal grade
```
`nor, aca_bd, aca_md, aca_pd, scc_bd, scc_md, scc_pd`.

**Train label counts (local):** nor 115, aca_bd 83, aca_md 74, scc_pd 60, scc_md 51, scc_bd 48, aca_pd 41. Imbalanced; the leaderboard **macro-averages over the 7 true classes** (rare classes matter as much as common ones).

**Three structural difficulties (and our answer to each):**
| Difficulty | Consequence | Our countermeasure |
|---|---|---|
| **Scarce data** (~472 patches) | From-scratch / full fine-tune overfits | Frozen foundation features + small regularized head |
| **Patient-stratified split** (30 train patients, disjoint test; only ~16 patches/patient) | Test score = cross-patient generalization; staining/morphology shift | Patient `GroupKFold`, stain augmentation, ensemble of encoders |
| **Joint belief + referral metric** | Both outputs scored; metric is non-proper | Bayes-optimal decode + budget-aware referral against the exact grader |

**Constraints:** single A10G (24 GB), **30 min train+inference**, offline. Train **only** on provided data. No `id`/filename hardcoding. Public, reproducible pretrained weights allowed.

---

## 2. The metric is the main lever — read it as a decision problem

Per test patch we submit 7 probabilities `p` (sum→1) **and** a `referral`∈[0,1]. Scoring (exact, from the description):

- **Referral allocation:** budget `K = int(0.20·N)` ≈ 43; a patch is eligible only if `referral > 0.5`; among eligible, escalate by **descending referral**, subject to a **per-true-class cap** `max(1, ceil(K/7)) = 7`; a referred patch scores a **flat `S_REF = 0.40`**.
- **Auto-reported patch** scores `harm = clip(hr − penalty, −2, +1)`:
  - `hr = Σ_k p_k · affinity(t, k)` — affinity: exact **+1**, same-subtype grade off-by-1 **+0.5**, off-by-2 **+0.2**, wrong subtype **0**, crosses benign↔malignant **−1**.
  - `penalty` — if `t` malignant: `1.0·p_nor²` (missed cancer); if `t = nor`: `0.5·(Σ malignant p)²` (false alarm). **Miss penalized 2× a false alarm.**
- **Macro-average** the per-patch scores over the 7 true classes, then linearly rescale `[−2, +1] → [0.02, 1.0]`.

Two facts drop out of this:

**(a) The metric is not a proper scoring rule.** A proper rule (e.g. log-loss) is maximized by reporting your true posterior. This one is maximized by a *different*, generally **peakier** distribution, because the reward is linear in `p` while the penalty is quadratic. ⟹ **Submitting softmax probabilities is leaving points on the table.** (§6.)

**(b) Referral is a *selective-prediction / reject-option* problem** ([Geifman & El-Yaniv 2017](https://www.researchgate.net/publication/317100919_Selective_Classification_for_Deep_Neural_Networks), [SelectiveNet 2019](https://arxiv.org/pdf/1905.09509)): escalate a patch iff its expected auto-report score is below the flat referral value `S_REF = 0.40`, under a fixed budget. This is a *threshold-at-budget* rule on expected harm. (§6.)

---

## 3. Representation: frozen pathology foundation features (compliance-ranked)

With 30 patients, the model must not learn patient identity. The single most important design choice is to **stand on a self-supervised pathology encoder pretrained on huge external WSI corpora** (no LungHist700 labels involved → no leakage), use it **frozen** as a feature extractor, and train only a tiny head. This is the standard, robust recipe for small medical datasets and it fits the compute budget trivially (one forward pass per crop).

### 3.1 Compliance gate (decisive) — which weights are actually usable

The SKILL rules and the challenge both ban **gated / credentialed / non-reproducible** weights. Research verdict (each claim adversarially verified 3–0):

| Model | Gated on HF? | License | Arch / dim | Verdict |
|---|---|---|---|---|
| **Phikon-v2** (`owkin/phikon-v2`) | **No** — directly downloadable | Owkin **non-commercial** | ViT-L/16, 224px@20x, **1024** | ✅ **PRIMARY** |
| **Lunit-DINO** (`lunit-io/benchmark-ssl-pathology`; mirror `1aurent/vit_small_patch{8,16}_224.lunit_dino`) | **No** — `torch.hub`/timm, no auth | Lunit **non-commercial** | ViT-S/8 & /16, **384**; also RN50 | ✅ **ENSEMBLE** |
| **Phikon-v1** (`owkin/phikon`) | **No** | Owkin non-commercial | iBOT ViT-B, **768** | ✅ optional |
| **CTransPath** (GitHub/TransPath) | Not HF-gated, but weights via Google-Drive | GPL-ish | Swin-Tiny (SRCL), **768** | ⚠️ only if **pre-staged** as an offline asset |
| **UNI / UNI2** (`MahmoodLab/UNI`) | **Yes** — institutional-email approval | CC-BY-NC-ND-4.0 | ViT-L/H | ❌ **AVOID** (gated) |
| **Virchow / Virchow2** (`paige-ai/*`) | **Yes** — institutional-email approval | CC-BY-NC-ND-4.0 | ViT-H | ❌ **AVOID** (gated) |
| **H-optimus-0/1** (`bioptimus/*`) | **Yes** — HF login | Apache-2.0 | ViT-g, **1536** | ❌ **AVOID** (gated access) |
| **Hibou-L** (`histai/hibou-L`) | **Yes** | — | ViT-L | ❌ **AVOID** (gated) |
| **Prov-GigaPath** (Microsoft) | **Yes** | — | ViT-g, **1536** | ❌ **AVOID** (gated) |

Sources: [UNI](https://huggingface.co/MahmoodLab/UNI), [Phikon-v2](https://huggingface.co/owkin/phikon-v2) + [paper](https://arxiv.org/html/2409.09173v1), [Virchow](https://huggingface.co/paige-ai/Virchow), [Virchow2](https://arxiv.org/pdf/2408.00738), [H-optimus-0](https://huggingface.co/bioptimus/H-optimus-0), [Hibou-L](https://huggingface.co/histai/hibou-L), [Lunit](https://github.com/lunit-io/benchmark-ssl-pathology).

**The crucial de-risking fact:** pathology tile-encoder performance **does not scale with model size** the way it does in NLP/natural images — *"smaller models perform on par with much larger models on most tasks"* ([benchmark, PMC12003829](https://pmc.ncbi.nlm.nih.gov/articles/PMC12003829/)). And on Owkin's external benchmark the encoders are near-parity: **Phikon-v2 0.874 AUC ≈ UNI 0.873 ≈ Prov-GigaPath 0.883 ≈ H-optimus-0 0.867** ([Phikon-v2 paper](https://arxiv.org/html/2409.09173v1)). **⟹ Restricting ourselves to non-gated models costs ≈ 0.01 AUC. There is essentially no accuracy penalty for full compliance.**

> **License note (flag for the user):** Phikon-v2 and Lunit are **non-commercial research** licenses. They are **non-gated and reproducible**, so they satisfy the stated compliance rule (which targets *gated/credentialed/private* weights). If the platform requires a *commercially*-permissive license, fall back to an ImageNet-pretrained ConvNeXt/ViT (Apache/MIT) with stronger augmentation (§4) — weaker, but fully permissive. **Recommendation: use Phikon-v2 + Lunit; confirm non-commercial license is acceptable for the Eris context.**

### 3.2 Recommended encoder stack
- **Primary:** Phikon-v2 (ViT-L, 1024-d) — strongest non-gated, pretrained DINOv2 on 460M tiles / 58k WSIs across 30+ cancer sites.
- **Diversity for the ensemble:** Lunit **DINO ViT-S/8** (384-d; the /8 patch size gives fine nuclear detail, useful for *grading*) and optionally Phikon-v1 (768-d, different SSL objective = iBOT).
- Concatenate the (L2-normalized) embeddings from the chosen encoders → one feature vector per crop. Ensemble diversity across *different SSL objectives/architectures* is the cheap, overfitting-free way to buy robustness on a tiny test set.

---

## 4. Turning a 1600×1200 patch (at 20x/40x) into encoder inputs

Foundation encoders expect **224×224**. Two failure modes to avoid: (i) resizing the whole 1600×1200 image to 224 destroys the nuclear/textural detail that *grading* depends on; (ii) a single crop is high-variance.

**Recipe — multi-crop, mean-pooled (deep-sets / MIL-lite):**
1. **Tile** each image into a grid of 224×224 crops (e.g. 5×4 = 20 tiles, or a sampled subset; tissue-only tiles via a simple H&E saturation/Otsu mask to drop white background).
2. Run the frozen encoder on each tile → **mean-pool** (and optionally also `max`-pool, concatenated) the tile embeddings into one image-level vector. Mean-pooling of tile features is the standard, robust aggregation for patch-bag problems and adds no parameters.
3. **Magnification handling.** Phikon-v2 is trained at **20x (0.5 mpp)**; our **40x** patches show cells ~2× larger than the encoder expects. Mitigate three ways, combined:
   - For 40x images, **also** generate a 0.5× downsampled view (≈ 20x field) and tile that, so the encoder sees a familiar scale; for 20x, optionally a 2× view. This makes the encoder **scale-consistent** across the magnification covariate.
   - **Append the magnification** (one-hot/binary) to the pooled feature vector fed to the head, so the head can condition on it.
   - Keep magnification in the **CV grouping/stratification** so we never validate on a leaked scale shortcut.

**Stain robustness (cross-patient generalization).** Test patients differ in staining. With **frozen** features the cheapest robust trick is **stain-augmented test-time augmentation**: extract features from a few **HED / RandStainNA**-perturbed ([RandStainNA, 2022](https://arxiv.org/abs/2206.12694)) versions of each image and average them. This bakes stain-invariance into the pooled feature without touching the encoder. (Optionally Macenko-normalize all inputs first for a common stain baseline; combining a light normalize + stochastic stain-augment TTA is standard and generalizes better than either alone.) Also add benign geometric TTA (flips/90° rotations — histology is rotation-invariant).

> Net: each image → a single pooled, stain/scale-robust feature vector (~1k–2k dims). Extraction for all 691 images across 2–3 encoders × multi-crop × a few TTA views is a few minutes on an A10G.

---

## 5. The head + calibration (calibration is *load-bearing*)

**Head.** A small, strongly-regularized classifier on the frozen features:
- **L2-regularized multinomial logistic regression** (primary — naturally well-calibrated, near-zero variance, ~instant) **and/or** a 1-hidden-layer MLP (dropout + weight decay + early stop on grouped CV). Average them.
- **Class imbalance:** class-weighted loss (inverse-frequency) so rare classes (aca_pd, scc_bd) are not ignored — aligned with the **macro** metric.
- Regularization strength `C` (logreg) / weight-decay (MLP) is the *only* capacity knob, chosen by patient-`GroupKFold` (§7). Keep it strong: with 30 patients the head must stay simple.

**Calibration — do not skip this.** Both decision-layer levers (§6) assume the posterior `q` is **calibrated**; my simulation shows that with *miscalibrated* `q`, the referral policy actively *loses* score. On <500 samples:
- Fit a **single global temperature** `T` on **out-of-fold** logits (1 parameter — the most data-efficient, reliable choice; [Guo et al. 2017](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf)).
- **Avoid** full Dirichlet/vector scaling here ([Dirichlet calib. 2019](https://arxiv.org/abs/1910.12656)) — too many parameters for 472 samples; it overfits the calibration set.
- Validate **ECE** and the macro metric on grouped CV before trusting the decode/referral.

**Ordinal/hierarchical heads — deliberately *not* used as the main model.** CORAL/CORN ordinal heads or hierarchical-softmax help when the *training loss* must encode order. Here the **downstream metric already grants hierarchy-aware partial credit**, and our **decode layer exploits the hierarchy explicitly** via the affinity matrix (§6). A flat 7-way *calibrated* classifier + metric-aware decode is **lower-variance** on tiny data than adding ordinal machinery. (Listed as an optional ablation, expected to be neutral-to-marginal.)

---

## 6. The decision layer — the differentiator (verified)

This converts a calibrated posterior `q` (7-vector) into the submission `(p, referral)` that **maximizes the published score in expectation**. It is pure math on a known function ⟹ **cannot overfit**.

### 6.1 Bayes-optimal belief `p` (closed form, proven exact)

Maximize `E_{t~q}[harm(p, t)]` over the simplex. Let `A[t,k] = affinity(t,k)` and `a = Aᵀ q` (so `a_k = Σ_t q_t·affinity(t,k)`).
- `E[hr] = Σ_k p_k a_k` (linear in `p`).
- `E[penalty] = q_nor·0.5·(1−p_nor)² + (1−q_nor)·1.0·p_nor²`.

The objective is **concave**; because the reward is linear in the malignant sub-mass, **all malignant probability goes to the single class `k* = argmax_{k≠nor} a_k`**, and the split between `nor` and `k*` (malignant mass `m = 1−p_nor`) solves a 1-D concave quadratic:

```
a_nor = 2·q_nor − 1
k*    = argmax over malignant k of a_k ;   a_mal = a_[k*]
m*    = clip( (a_nor − a_mal − 2·(1−q_nor)) / (q_nor − 2),  0, 1 )
p_nor = 1 − m* ;   p_[k*] = m* ;   else 0
```

**Verification (own code vs. the exact grader):** over **150 posteriors × ~450k candidate distributions**, `max(brute-force − closed-form) = 0.0` — the closed form is the exact optimum. Average leaderboard uplift vs. naive baselines (5000 random posteriors):

| Submitted belief | Raw harm | Leaderboard |
|---|---|---|
| posterior `q` (naive) | 0.136 | 0.7177 |
| argmax one-hot | 0.215 | 0.7435 |
| **Bayes-optimal decode** | **0.299** | **0.7711** |

⟹ **+0.053 vs. posterior, +0.028 vs. one-hot.**

**Emergent behaviors the decoder discovers automatically (verified examples):**
- **Unsure of *grade* within a subtype → bet the *middle* grade (md).** It is never "off by 2", so it maximizes expected partial credit (+0.5 to both neighbors vs. risking +0.2). The decoder finds this with no hand-coding.
- **Subtype tossup (aca vs scc) → commit to one** (cross-subtype reward is 0 either way); expected harm → 0.40 — exactly the referral margin.
- **Benign/malignant genuinely uncertain → split `nor` vs. best malignant**, yielding low expected harm → which flags it as a **referral candidate**, tying the two layers together.

### 6.2 Macro-average–aware refinement
The leaderboard macro-averages over true classes, so an error on a **rare** class costs `1/n_c` more. Make the decode macro-aware by weighting each true-class term by an estimated **inverse test-class frequency** (from the train prior): `a_k = Σ_t q_t·(1/n_t)·affinity(t,k)` and likewise for the penalty. This nudges mass toward rare classes — the right thing for a macro metric. Treat the prior as estimated; **validate on macro-averaged grouped CV** and back off if it doesn't help (it depends on the test prior matching train).

### 6.3 Referral policy (budget-aware selective prediction)
For each patch compute the optimal `p` and its expected harm `E[harm]`. Escalating replaces that with `S_REF = 0.40`, so the **expected uplift = max(0, 0.40 − E[harm])**. Rank patches by uplift and flag the **top ≈ K** (budget `K ≈ 43`) by setting `referral ∈ (0.5, 1.0]` in **descending uplift order** (encode the ranking in the magnitude); set `referral ≤ 0.5` for everyone else.

**Verified, and with an honest caveat:**
- With **calibrated** posteriors (true label sampled from `q`), referring the lowest-`E[harm]` patches gains **+0.017–0.018 leaderboard**, optimum spending most of the budget (~30–43 patches). Holds across strong/medium/weak models.
- With **miscalibrated** posteriors, the *same* policy **loses 0.01–0.03** — escalating a patch whose true (easy) class already scores ~1.0 drags down that class's macro mean toward 0.40.
- **⟹ `sample_submission.csv`'s `referral = 1.0` for every row is actively harmful.** Referral is only safe **after** calibration, and the **referral count is a hyperparameter tuned on the exact grader over patient-grouped CV** — if CV says it doesn't help, submit minimal/zero referral.

---

## 7. Validation protocol (how we tune without overfitting)

- **Patient `GroupKFold` by `patient_id`** so no patient spans train/val — this mirrors the real train/test split. Only 30 patients ⟹ **5 folds × ~6 patients**; **repeat** with several seeds and treat **CV variance as a first-class number** (report mean ± std).
- **Score every fold with the *exact* grader** (re-implemented from the description: referral allocation + per-true-class cap + macro-average + rescale), not a proxy. The whole pipeline (decode + referral) is selected to maximize *this*.
- **Tune only:** head regularization, temperature `T`, encoder/TTA choices, the macro-reweight on/off, and the referral count. Everything is chosen on grouped CV — **never** on the public leaderboard (treat public score as a coarse diagnostic only; no probing).
- **Out-of-fold logits** feed temperature fitting and an honest estimate of the final score.
- Sanity: confirm the hardest axis is **grade** (md/pd confusions) and watch the **benign↔malignant** confusion (the −1 affinity + miss penalty make it the most expensive error) — bias the operating point to avoid confident misses.

---

## 8. Ensembling & the full pipeline

```
For each image (train+test):
  tissue-mask → multi-crop 224 tiles (+ scale-normalized view for 40x)
  for each encoder in {Phikon-v2, Lunit ViT-S/8, [Phikon-v1]}:
     for each TTA view in {stain/HED perturb × flips/rot}:
        embed tiles → mean(+max) pool
  concat & L2-normalize → image feature vector  (+ magnification flag)

Train (patient GroupKFold):
  class-weighted L2-logreg  ⊕  small MLP (dropout/WD/early-stop)  → average → q_raw
  fit single global temperature T on OOF logits → calibrated q

Inference (per test patch):
  q → §6.1 closed-form Bayes-optimal p  (with §6.2 macro reweight)
  E[harm](p) → §6.3 referral ranking under budget K
  write p_nor..p_scc_pd, referral
```

All frozen-feature; the only trained parameters are a logreg/MLP head + one temperature. Comfortably within **A10G / 30 min** (feature extraction ≈ minutes; head ≈ seconds; repeated CV ≈ minutes).

---

## 9. Compliance & anti-overfitting checklist

- ✅ **No gated/credentialed models** — Phikon-v2 & Lunit are non-gated, `torch.hub`/HF-downloadable, reproducible offline (pre-stage weights as an offline asset for the locked runtime). Gated UNI/Virchow/GigaPath/H-optimus/Hibou explicitly avoided.
- ✅ **No external labeled data.** LungHist700's own labels are **never** used (the test patches are a subset of it — using them would be leakage); foundation weights are self-supervised on *other* corpora (no LungHist700 labels). The dataset paper is used for **methodology/expectations only**.
- ✅ **No metadata/`id`/filename shortcuts; no leaderboard probing; no hardcoded rows.** Magnification is a legitimate provided covariate, used as a model input only.
- ✅ **Generalization-first:** every hyperparameter chosen on **patient-grouped CV**; capacity kept tiny; gains come from features + decision theory, both of which transfer to unseen patients.
- ✅ **Determinism:** fixed seeds, no network at inference, valid submission schema (7 probs sum→1 within tol, `referral`∈[0,1], one row per test id).

---

## 10. Why this beats a standard AI baseline (and where each point comes from)

A typical baseline = fine-tune one ImageNet net on 472 resized images, submit softmax, naive referral. It overfits 30 patients and ignores the metric. Our expected **progression ladder** (each step validated on grouped CV with the exact grader):

| Step | Change | Why it helps | Overfit risk |
|---|---|---|---|
| B0 | ImageNet net, softmax, no referral | baseline | high (memorizes patients) |
| B1 | **Frozen pathology FM features + logreg** | cross-patient generalization on tiny data | low |
| B2 | **+ multi-crop pooling + stain/scale TTA** | preserves grading detail; stain-invariance | low |
| B3 | **+ temperature calibration** | unlocks the decision layer | low |
| B4 | **+ Bayes-optimal belief decode** | **+0.05 LB**, exact metric optimum | **none** |
| B5 | **+ budget-aware referral** | **+0.018 LB** (calibrated) | none (CV-tuned) |
| B6 | **+ encoder ensemble + macro-reweight** | variance reduction, macro alignment | low |

The B4/B5 gains are **proven against the grader** and independent of the data, so they hold on the private test set. B1–B3/B6 are the well-established small-data histopathology recipe (frozen FM + light head + stain aug + ensemble), chosen specifically because it **generalizes** rather than memorizes.

---

## 11. Risks & honest unknowns

- **CV variance is large** (30 patients). Decisions must clear mean ± std, not a lucky fold. Prefer simpler models when folds disagree.
- **Magnification scale gap** for Phikon-v2 at 40x — mitigated by scale-normalized views + mag covariate; verify per-magnification CV.
- **Macro-reweight (§6.2)** assumes test prior ≈ train prior; it's gated behind CV and dropped if it doesn't help.
- **Referral is calibration-sensitive** — ship it only if calibrated CV shows a gain; otherwise minimal referral.
- **Non-commercial license** on Phikon-v2/Lunit — compliant with the *gating* rule; confirm acceptable for the platform, else use the permissive-ImageNet fallback.
- Published LungHist700 baselines indicate **fine 7-class + grading is substantially harder than binary/subtype** — keep expectations calibrated and lean on the metric's partial-credit structure (the decoder already does).

---

## 12. Key references (verified)
- LungHist700: [PMC11455975](https://pmc.ncbi.nlm.nih.gov/articles/PMC11455975/), [Sci Data 2024](https://www.nature.com/articles/s41597-024-03944-3), [figshare](https://figshare.com/articles/dataset/LungHist700_A_Dataset_of_Histological_Images_for_Deep_Learning_in_Pulmonary_Pathology/25459174)
- Phikon-v2 (primary encoder): [HF](https://huggingface.co/owkin/phikon-v2), [paper](https://arxiv.org/html/2409.09173v1) · Lunit SSL: [GitHub](https://github.com/lunit-io/benchmark-ssl-pathology)
- Encoder size doesn't scale / near-parity: [PMC12003829](https://pmc.ncbi.nlm.nih.gov/articles/PMC12003829/), [Nat. BME 19-model benchmark](https://www.nature.com/articles/s41551-025-01516-3)
- Gated models avoided: [UNI](https://huggingface.co/MahmoodLab/UNI), [Virchow2](https://arxiv.org/pdf/2408.00738), [H-optimus-0](https://huggingface.co/bioptimus/H-optimus-0)
- Calibration: [Temperature scaling, Guo 2017](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf), [Dirichlet 2019](https://arxiv.org/abs/1910.12656)
- Selective prediction / reject option: [Geifman & El-Yaniv 2017](https://www.researchgate.net/publication/317100919_Selective_Classification_for_Deep_Neural_Networks), [SelectiveNet 2019](https://arxiv.org/pdf/1905.09509)
- Stain augmentation: [RandStainNA 2022](https://arxiv.org/abs/2206.12694)

---

### Appendix — recommended time-spent (submission form): **~12–16 hours**
(~3h data/metric analysis + decision-theory derivation & verification; ~4h feature pipeline & encoders; ~3h head/calibration + grouped-CV harness; ~2h referral tuning & ensembling; ~2h validation, ablation ladder, write-up.)
