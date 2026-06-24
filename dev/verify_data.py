"""Verify a (re-provided) dataset is intact before training.
Usage: python dev/verify_data.py [DATA_DIR]
Checks: image files vs unique contents, per-image label consistency, train/test content overlap."""
import sys, hashlib
from pathlib import Path
import pandas as pd

D = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("G:/Datacurve/eris/runnerloaddataset/public")
tr = pd.read_csv(D/"train.csv"); te = pd.read_csv(D/"test.csv")
def h(im): return hashlib.md5((D/"images"/im).read_bytes()).hexdigest()
tr["h"] = tr["image"].map(h); te["h"] = te["image"].map(h)
nfiles = len(list((D/"images").glob("*.jpg")))
allh = pd.concat([tr.h, te.h])
mult = tr.groupby("h")["label"].nunique()
overlap = len(set(tr.h) & set(te.h))

print(f"DATA DIR: {D}")
print(f"image files on disk : {nfiles}")
print(f"train: {len(tr)} rows -> {tr.h.nunique()} unique images")
print(f"test : {len(te)} rows -> {te.h.nunique()} unique images")
print(f"ALL  : {len(allh)} rows -> {allh.nunique()} unique images   (corrupted copy had 134)")
print(f"per-unique-image distinct labels: mean={mult.mean():.2f} (want ~1.00), single-label frac={ (mult==1).mean():.2f} (want ~1.00)")
print(f"train/test content overlap: {overlap}  (want 0)")

intact = (tr.h.nunique() >= 0.9*len(tr)) and (mult.mean() < 1.1) and (overlap == 0)
print("\nVERDICT:", "INTACT - ready to train" if intact else "STILL CORRUPTED - do not train")
sys.exit(0 if intact else 1)
