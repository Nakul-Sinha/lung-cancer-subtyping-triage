Computer Vision Challenge, Scoring↑ Higher is better

GPU**A10G**  
  
Lung Cancer Subtyping & Grading
Overview
You are building the triage model for a digital-pathology lab. Given a single hematoxylin-and-eosin (H&E) tissue patch cropped from a lung whole-slide image, your model must do two things at once:

Diagnose, output a belief distribution over seven diagnostic classes (how much probability you place on each; the seven must sum to 1).
Triage, output a referral score saying how badly this patch needs a senior pathologist's review.
The lab has limited specialist capacity: only a fixed fraction of cases can be escalated. An escalated patch is resolved by the specialist (it receives a fixed score) but consumes scarce expert time; an auto-reported patch is scored on its diagnostic harm, where a confident mistake on the wrong side of the cancer boundary is heavily penalized.

The seven classes form a clinical hierarchy, and scoring follows that hierarchy rather than treating the classes as flat and interchangeable:

```
                 patch  
        ┌──────────┴───────────┐  
    benign (nor)            malignant  
                       ┌────────┴────────┐  
                  ACA subtype        SCC subtype  
                  bd < md < pd       bd < md < pd      ← ordinal differentiation grade  
```

Tier 1, triage: benign normal tissue (nor) vs. malignant carcinoma. Confusing the two is the clinically decisive error, a missed cancer (auto-reporting a malignant patch as normal) is the worst outcome; a false alarm is penalized too, but less.
Tier 2, subtype: among malignant patches, adenocarcinoma (aca) vs. squamous cell carcinoma (scc), these drive different treatments.
Tier 3, differentiation grade: an ordinal severity axis within each subtype, well-differentiated (bd), moderately (md), poorly (pd). Grade is the hardest, most subjective axis; the metric awards partial credit for being off by one grade and less for two.
Three things make this hard. First, data is scarce, under ~500 training patches. Second, the train/test split is patient-stratified: the patients in the training set never appear in the test set, so test scores reflect generalization across patients, staining, and morphology rather than memorization. Third, the score is a joint function of the belief distribution and the referral signal, both are part of every submission and both affect the result.

Compute budget: training and inference must complete on a single A10G GPU (24 GB) within 30 minutes.

Scenario context:

All images are H&E-stained lung tissue patches, fixed-size RGB crops from whole-slide images.
Each patch carries a magnification covariate (20x or 40x), available at training and inference time.
Classes are imbalanced (normal tissue is the largest class; scc_md is the smallest); the leaderboard macro-averages over the seven classes, weighting each class equally regardless of its size.
Use only the files provided in the competition package for training and inference.

Dataset
Public files
public/train.csv

Training data with ground-truth labels.
Columns: id, image, magnification, patient_id, superclass, subclass, label.
~480 rows (patient-stratified; patients disjoint from the test set).
public/test.csv

Test data without labels.
Columns: id, image, magnification.
~210 rows (patients never seen in training).
public/images/

All tissue patches for both train and test, named with opaque hash tokens (e.g. img_4f9a1c7b2e8d6a05.jpg). Original filenames are not recoverable.
public/sample_submission.csv

Example submission in the correct format: uniform 1/7 belief on every class, with referral signalled on a 20% slice of patches to exercise the referral column.
Private file (organizer only)
private/answers.csv
Columns: id, label.
Column descriptions
The train.csv and test.csv files contain the columns below (the columns patient_id, superclass, subclass, and label appear only in train.csv; submission columns are documented in the Submission format section):

Column	Type	Description
id	string	Unique patch identifier (the hash token, e.g. img_4f9a1c7b2e8d6a05). Used to join submissions to patches.
image	string	Filename of the tissue patch inside public/images/ (e.g. img_4f9a1c7b2e8d6a05.jpg).
magnification	string	Acquisition magnification of the patch: 20x or 40x. Available for both train and test patches.
patient_id	integer	Patient identifier (train only). Multiple patches share the same patient. Useful for building patient-stratified cross-validation folds.
superclass	string	High-level tissue category (train only): nor (normal), aca (adenocarcinoma), or scc (squamous cell carcinoma).
subclass	string	Differentiation grade within a cancer subtype (train only): bd (well-differentiated), md (moderately differentiated), pd (poorly differentiated); empty string for nor.
label	string	Fine-grained class (train only), one of nor, aca_bd, aca_md, aca_pd, scc_bd, scc_md, scc_pd. This is the target the submission places belief over.
Data example
Example from train.csv:

```text
id,image,magnification,patient_id,superclass,subclass,label  
img_4f9a1c7b2e8d6a05,img_4f9a1c7b2e8d6a05.jpg,40x,12,aca,md,aca_md  

Example from test.csv (no label columns):

undefined

id,image,magnification
img_8b3e0d61aa9c4f72,img_8b3e0d61aa9c4f72.jpg,20x

## Submission format Submit a CSV with one row per test patch: the seven class probabilities **and** a referral score. Required columns (in any order, header required): - `id` - `p_nor`, `p_aca_bd`, `p_aca_md`, `p_aca_pd`, `p_scc_bd`, `p_scc_md`, `p_scc_pd` - `referral` Rules: - Exactly one row per test patch (same count and same `id` set as `public/test.csv`), with a header row. - Each probability is a number in `[0, 1]`; the seven probabilities in a row must sum to `1.0` (a tolerance of ±0.02 is allowed; rows are renormalized before scoring). - `referral` is a number in `[0, 1]` indicating how strongly the patch is flagged for specialist escalation. A patch is considered for referral only if its `referral` exceeds `0.5`; among those, the highest scores are escalated first, up to the budget. A `referral` at or below `0.5` is never escalated. Values are used as a gate and a ranking, not as probabilities. - No duplicate `id`s; no missing or unknown `id`s; no empty or non-numeric values. ```csv ### Sample submission ```text id,p_nor,p_aca_bd,p_aca_md,p_aca_pd,p_scc_bd,p_scc_md,p_scc_pd,referral img_8b3e0d61aa9c4f72,0.40,0.10,0.10,0.10,0.10,0.10,0.10,0.0 img_2c7f91a4be03d8e6,0.05,0.05,0.60,0.15,0.05,0.05,0.05,0.8
Evaluation
Submissions are scored with a referral-gated hierarchical clinical-harm score, macro-averaged over the seven true classes. Higher is better.

Step 1, referral allocation. Referral is opt-in and capped. The total referral budget is K = int(0.20 · N) where N is the number of test patches in the ground-truth set being scored (derived from the ground-truth dictionary, not the submission row count). A patch is eligible for escalation only if its referral score exceeds REFERRAL_THRESHOLD = 0.5. Among eligible patches, sorted by descending referral score (ties broken by id ascending), patches are referred subject to a per-ground-truth-class cap of max(1, ceil(K / 7)). Since participants do not know the true class of test patches, this cap cannot be gamed; it prevents any single class from being disproportionately referred, which would distort the macro-average. Once K total referrals are reached or all eligible patches are processed, everything else is auto-reported. A submission with every referral at or below 0.5 escalates nothing.

Step 2, per-patch score.

A referred patch scores a flat S_REF = 0.40, modelling a case the specialist resolves correctly (≈ +1.0) net of the cost of expert time.
An auto-reported patch scores the hierarchical clinical-harm value below, with two terms.
Term 1, hierarchical affinity reward (expected, hierarchy-weighted agreement between belief and truth):

Relation of predicted class k to the true class t	affinity w(t, k)
exact class	+1.0
same subtype, differentiation grade off by 1 (e.g. aca_md ↔ aca_bd)	+0.5
same subtype, differentiation grade off by 2 (bd ↔ pd)	+0.2
both malignant but wrong subtype (aca ↔ scc)	0.0
crosses the benign ↔ malignant boundary	−1.0
Term 2, convex, asymmetric triage gate (penalizes confident wrong-side belief, accelerating quadratically):

if t is malignant: penalty = GAMMA_MISS · (p_nor)², mass on "normal" when the patch is cancer (missed diagnosis).
if t is benign (nor): penalty = GAMMA_FA · (Σ malignant p)², mass on cancer when the patch is normal (false alarm).
GAMMA_MISS = 1.0 and GAMMA_FA = 0.5: a confident missed cancer is penalized twice as hard as a confident false alarm.

CLASSES = ["nor", "aca_bd", "aca_md", "aca_pd", "scc_bd", "scc_md", "scc_pd"]  
GAMMA_MISS = 1.0          # true malignant, belief on 'nor'   (missed cancer)  
GAMMA_FA   = 0.5          # true benign,   belief on malignant (false alarm)  
REFERRAL_BUDGET = 0.20    # at most 20% of test patches may be referred  
REFERRAL_THRESHOLD = 0.5  # eligible for referral only if referral > 0.5  
S_REF = 0.40              # flat score for a referred patch  
SCORE_FLOOR = 0.02  
HARM_FLOOR  = -2.0  
HARM_CEILING = 1.0  
  
def subtype(c):  
    return c.split("_")[0]  
  
def grade_rank(c):  
    return {"bd": 1, "md": 2, "pd": 3}.get(c.split("_")[1], 0) if "_" in c else 0  
  
def affinity(t, k):  
    if k == t:                        return  1.0  
    if t == "nor" or k == "nor":      return -1.0  
    if subtype(t) != subtype(k):      return  0.0  
    return {1: 0.5, 2: 0.2}[abs(grade_rank(t) - grade_rank(k))]  
  
def harm_score(probs, gt_label):  
    hr = sum(probs[k] * affinity(gt_label, k) for k in CLASSES)  
    if gt_label != "nor":  
        penalty = GAMMA_MISS * probs["nor"] ** 2  
    else:  
        penalty = GAMMA_FA * sum(probs[k] for k in CLASSES if k != "nor") ** 2  
    return max(HARM_FLOOR, min(HARM_CEILING, hr - penalty))  
  
def grade(pred, gt):  
    n = len(gt)  
    k = int(REFERRAL_BUDGET * n)  
    max_per_gt_class = max(1, ceil(k / len(CLASSES)))  
  
    eligible = [i for i in pred if pred[i][1] > REFERRAL_THRESHOLD]  
    eligible.sort(key=lambda i: (-pred[i][1], i))  
  
    referred = set()  
    gt_class_counts = {c: 0 for c in CLASSES}  
    for pid in eligible:  
        if len(referred) >= k:  
            break  
        true_label = gt[pid]  
        if gt_class_counts[true_label] < max_per_gt_class:  
            referred.add(pid)  
            gt_class_counts[true_label] += 1  
  
    by_class = {c: [] for c in CLASSES}  
    for pid, label in gt.items():  
        probs, _ = pred[pid]  
        s = S_REF if pid in referred else harm_score(probs, label)  
        by_class[label].append(s)  
  
    class_means = {c: sum(v) / len(v) for c, v in by_class.items() if v}  
    macro = sum(class_means.values()) / len(class_means)  
    score = SCORE_FLOOR + (macro - HARM_FLOOR) / (HARM_CEILING - HARM_FLOOR) * (1.0 - SCORE_FLOOR)  
    return max(SCORE_FLOOR, min(1.0, score))  

The raw macro-average (over the seven true classes, each weighted equally) is linearly rescaled from its natural range [−2.0, +1.0] to [0.02, 1.0] for the leaderboard. The true classes are not provided in the test set.

Higher is better. The theoretical maximum is 1.0 (perfect predictions on every class); the minimum is 0.02.

What Not To Use
Keep the setup fair and leakage-free. Do not:

use any external data source that contains pre-computed labels, annotations, or diagnostic metadata for lung histopathology patches that overlaps with this test set.
hardcode outputs or referral decisions based on id, image filename, or any fixed test pattern.
apply preprocessing, augmentation, or split logic that leaks information between the train and test patients.
use tools, models, or workflows that tune directly against the private test labels (e.g. leaderboard probing).
Build and train your model only with the provided data, then run standard inference on the test patches.
```

