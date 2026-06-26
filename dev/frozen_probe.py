"""Diagnostic + robust baseline: FROZEN backbone features + logistic-regression probe.
Caches features so head experiments are instant. Run: python dev/frozen_probe.py
"""
import os, warnings, time
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd, torch, timm
from PIL import Image
import torchvision.transforms.functional as TF
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import CLASSES, CIDX, exact_grade, decode_belief, harm_score, referral_from_expected_harm

DATA = Path("G:/ml/data/runnerloaddataset/public")
BACKBONE = os.environ.get("BACKBONE","convnext_tiny.fb_in22k_ft_in1k")
CACHE = Path("dev/feat_cache"); CACHE.mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

def extract(df, tag, n_views=1):
    fp = CACHE/f"{tag}_{BACKBONE.split('.')[0]}.npy"
    if fp.exists(): return np.load(fp)
    cfg = timm.get_pretrained_cfg(BACKBONE); MEAN,STD = list(cfg.mean), list(cfg.std)
    m = timm.create_model(BACKBONE, pretrained=True, num_classes=0, global_pool="avg").to(DEV).eval()
    feats=[]; t0=time.time()
    with torch.no_grad():
        for n,(_,r) in enumerate(df.iterrows()):
            im = Image.open(DATA/"images"/r["image"]).convert("RGB"); im.thumbnail((512,512))
            views=[]
            for v in range(n_views):
                x = TF.resize(im,(224,224)) if v==0 else TF.resize(im,(256,256))
                x = TF.center_crop(x,(224,224)) if v>0 else x
                x = TF.normalize(TF.to_tensor(x),MEAN,STD).unsqueeze(0).to(DEV)
                views.append(m(x).cpu().numpy())
            feats.append(np.mean(views,0))
            if n%100==0: print(f"  {tag} {n}/{len(df)} ({time.time()-t0:.0f}s)")
    F=np.concatenate(feats,0); np.save(fp,F); print(f"cached {fp} {F.shape}"); return F

def main():
    tr=pd.read_csv(DATA/"train.csv"); y=tr["label"].map(CIDX).values; groups=tr["patient_id"].values
    mag=(tr["magnification"].astype(str).str.startswith("40")).astype(float).values.reshape(-1,1)
    F=extract(tr,"train")
    Fz=np.hstack([F/np.linalg.norm(F,axis=1,keepdims=True), mag])  # L2-norm + mag covariate
    cnt=np.bincount(y,minlength=7); w_macro=(1.0/np.maximum(cnt,1)); w_macro/=w_macro.mean()

    for C in [0.05,0.1,0.3,1.0]:
        oof=np.zeros((len(y),7))
        sgkf=StratifiedGroupKFold(5,shuffle=True,random_state=42)
        for tri,vai in sgkf.split(Fz,y,groups):
            lr=LogisticRegression(C=C,max_iter=2000,class_weight="balanced")
            lr.fit(Fz[tri],y[tri]); oof[vai]=lr.predict_proba(Fz[vai])
        acc=(oof.argmax(1)==y).mean()
        bacc=np.mean([(oof[y==c].argmax(1)==c).mean() for c in range(7) if (y==c).any()])
        # decode + referral score
        P=np.zeros_like(oof); Eh=np.zeros(len(y))
        for i in range(len(y)): P[i],_=decode_belief(oof[i],w_macro); Eh[i]=sum(oof[i,t]*harm_score(P[i],t) for t in range(7))
        best=-1; bestfr=0
        for fr in [0,.1,.15,.2,.25,.3]:
            s=exact_grade(P,referral_from_expected_harm(Eh,fr),y)
            if s>best: best,bestfr=s,fr
        print(f"C={C:<5} acc={acc:.3f} bacc={bacc:.3f}  decode+ref OOF={best:.4f} (fr={bestfr})")

if __name__=="__main__": main()
